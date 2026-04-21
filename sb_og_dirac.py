#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 06:31:59 2025

@author: huisun
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import matplotlib.cm as cm  # Correct import for colormaps
# The 3D import is only needed for 3D plots
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

eps=1e-6 

class SDE_eps(object):
    def __init__(self,N,Xdata,eps=1e-6): 
        """
        x1-- the terminal variable
        a-- the initial variable
        x--  the input variable
        N -- the total number timesteps
        """
        self.X1=Xdata; 
        self.N=N 
        self.dt=torch.tensor(1.0/N)
        self.d=self.X1.shape[1]
        self.eps=eps

    def sigma2VE(self, t): 
        sig=t
        return sig

    def sigmaDiff(self,t):
        siguD=torch.tensor(1.0)
        return siguD

    def ftVE(self,x1,a,x,t): 
        ## alpha_0=0.0
        num=self.sigma2VE(t)
        ft_xx1=torch.exp(torch.sum((x1-a)**2, axis=1)/(2.0) -torch.sum((x1-x)**2,axis=1)/(2.0-2.0*num))
        ## This provides a weight vector
        return ft_xx1.unsqueeze(1)

    def uVE(self,a,X1,x,t): 
        ## defining the numerator
        num=(1.0-self.eps)*torch.mean((X1-x)*self.ftVE(X1,a,x,t),axis=0)
        ## defining the denominator
        deno=(1.0-self.eps)*torch.mean(self.ftVE(X1,a,x,t),axis=0)*(1.0-self.sigma2VE(t))
        return num/(deno+ ((1.0-t)**(self.d/2))*self.eps)

    def forward(self,a):
        ## we just want the terminal x1.
        ## a can be the [0,0] vector. 
        Xt=a.clone()
        
        ### simulating the SDE trajectory by trajectory. 
        t=torch.tensor(0.0)
        for i in range(self.N): 
            Xt=Xt+self.dt*self.uVE(a,self.X1,Xt,t)+torch.randn_like(Xt)*torch.sqrt(self.dt)#*self.sigmaDiff(t)
            t=t+self.dt
        return Xt
    
    
    

class SBG:
    def __init__(self, T, N, Xdata):
        self.T, self.N, self.Xdata = T, N, Xdata
        self.M_data, self.D = Xdata.shape
        self.dt = T / N
        self.tvec = np.linspace(0, T, N + 1)
        self.X_sq = np.sum(Xdata ** 2, axis=1) / (2.0 * T)

    def ftVE_batch(self, x_batch, t):
        diff = self.Xdata[None, :, :] - x_batch[:, None, :]  # (B, M_data, D)
        sq = np.sum(diff ** 2, axis=2)  # (B, M_data)
        Tm = max(self.T - t, 1e-8)
        p1 = sq / (2 * Tm)  # (B, M_data)
        p2 = self.X_sq[None, :]  # (1, M_data)
        logw = -p1 + p2
        logw -= np.max(logw, axis=1, keepdims=True)
        return np.exp(logw)[:, :, None]  # (B, M_data,1)

    def uVE_batch(self, x_batch, t):
        w = self.ftVE_batch(x_batch, t)  # (B, M_data,1)
        diff = self.Xdata[None, :, :] - x_batch[:, None, :]  # (B, M_data, D)
        num = np.sum(w * diff, axis=1) / self.M_data  # (B, D)
        denom = np.mean(w, axis=1) * (self.T - t)  # (B,1)
        denom = np.clip(denom, 1e-8, None)
        return num / denom  # (B, D)

    def sde_batch_with_snapshots(self, B, steps):
        x = np.random.randn(B, self.D)  # Initialize from standard normal distribution
        snapshots = {0: x.copy()}
        for i, t in enumerate(self.tvec[:-1], start=1):
            drift = self.uVE_batch(x, t) * self.dt
            noise = np.random.randn(B, self.D) * np.sqrt(self.dt)
            x = x + drift + noise
            if i in steps:
                snapshots[i] = x.copy()
        return snapshots
    
    
    
    
    
    
    
    