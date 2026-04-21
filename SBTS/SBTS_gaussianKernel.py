#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 24 09:42:48 2025

@author: huisun
"""

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
import time 
import os
import sys
import pandas as pd 

from scipy.stats import norm
import time as ttt
import iisignature as iisig
from tqdm import *


class timeSB_OG_Gaussian(object):
    def __init__(self,Xtensor,h,time,Ntime,Nstep): 
        self.Xtensor=Xtensor
        self.h=h
        self.Nstep=Nstep
        #self.tvec=np.array([0.0,1.0,2.0,3.0])/3.0
        self.tvec=np.linspace(0,time,Ntime)

    def mainsweep(self):
        tvec=self.tvec
        Klist=[]
        Mlist=[]
        xlist=[]

        ## The first iteration 
        Xvec0=self.Xtensor[:,0,]
        Xvec1=self.Xtensor[:,1,]

        ### Original input should be 0. 
        xin0=torch.zeros(self.Xtensor[0,0,].shape)
        xlist.append(xin0)  ### Attaching the current first value

        ## Obtain the mask and the kernel values 
        kvals=self.KernelGaussian(xin0, Xvec0)
        ### Data at preselected time intervals
        Klist.append(kvals) # t0 time kernel values
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            kvals=self.KernelGaussian(xin0,Xvec0)
            Klist.append(kvals) # ti time kernel values

            
            kvals_masked=self.cumulativeMK(Klist)
            
            Xvec0_new=Xvec0
            Xvec1_new=Xvec1
            
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_masked, Xvec0_new , Xvec1_new, xin0) # 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        res=torch.concatenate(xlist)

        return res #Xvec0,Mlist


    def cumulativeMK(self, klist): 

        k0=klist[0]
        for l in range(1, len(klist)):
            k0=k0*(klist[l])

        return k0  
            
    def OneStep(self,ti, ti1, kvals, Xti, Xti1, xin0): 
        nsteps=self.Nstep

        delt=(ti1-ti)/nsteps
        sqrt_delt=np.sqrt(delt)

        Nt=np.linspace(ti,ti1,nsteps+1)
        x0=torch.clone(xin0)

        for l in range(len(Nt)-1):
            x0=x0+self.Drift(ti,ti1, Nt[l],Xti, x0, Xti1, kvals)*delt+torch.randn_like(x0)*sqrt_delt        
        return x0
            

    def Drift(self,ti,ti1,t,Xti,x, Xti1, kvals):
        
        wi=self.Fi(ti,ti1,t,Xti,x, Xti1)
        wi_ext=torch.unsqueeze(wi*kvals,dim=1)
        
        num=torch.mean((Xti1-x)*wi_ext,dim=0) 
        denom=torch.mean(wi*kvals)

        res=num/(denom*(ti1-t))
        return res
  

    def Fi(self,ti,ti1,t,Xti,x, Xti1): 
        """
        Output is a tensor of shape B_{data}
        Now xi is not data
        """
        p1=-torch.norm(x-Xti1,dim=1)**2/(2.0*(ti1-t))
        p2=torch.norm((Xti1-Xti),dim=1)**2/(2.0*(ti1-ti))
        res=torch.exp(p1+p2)
        return res


    def KernelGaussian(self,xi,Xvec):
        """
        Returns the Mask, i.e. which elements to keep
        The kernel values
        """
        scaled_val=torch.norm(xi-Xvec,dim=1)/self.h
        out=torch.exp(-scaled_val**2/2.0)#/self.h; 
        return out



class timeSB_gaussianSig(object):
    def __init__(self,Xtensor,h,time,Ntime,Nstep): 
        self.Xtensor=Xtensor
        self.h=h
        self.tvec=np.linspace(0,time,Ntime)
        self.M=self.Xtensor.shape[0]
        self.Nstep=Nstep
        
        self.depth=3
        self.sig_xt=self.GenSignature_Xvec()

    def mainsweep(self):
        tvec=self.tvec
        Klist=[]
        Mlist=[]
        xlist=[]

        ## The first iteration 
        Xvec0=self.Xtensor[:,0,]
        Xvec1=self.Xtensor[:,1,]
        
        Xvec_sig0=self.sig_xt[:,0,]
        Xvec_sig1=self.sig_xt[:,1,]

        ### Original input should be 0. 
        xin0=torch.zeros(self.Xtensor[0,0,].shape)
        xlist.append(xin0)  ### Attaching the current first value
        
        xin01=xin0.unsqueeze(0)
        xin02=xin01.unsqueeze(0)   
        tvec1=np.reshape(self.tvec[0],(1,1,1)) 
        xvec_t0=np.concatenate((tvec1, xin02),axis=2)
        x0_sig=iisig.sig(xvec_t0,self.depth)
        


        ## Obtain the mask and the kernel values 
        kvals_sig=self.KernelGaussianSig(x0_sig, Xvec_sig0)
        Klist.append(kvals_sig) 
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals_sig, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]
            
            Xvec_sig0=self.sig_xt[:,i,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            xvec_tensor=torch.stack(xlist)
            tvec_temp=self.tvec[0:i+1]
            x0_sig=self.genSig_one(xvec_tensor, tvec_temp)
            
            kvals_sig=self.KernelGaussianSig(x0_sig,Xvec_sig0)
            Klist.append(kvals_sig) # ti time kernel values
            
            Xvec0_new=Xvec0
            Xvec1_new=Xvec1
            
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_sig, Xvec0_new , Xvec1_new, xin0) # 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        res=torch.cat(xlist) #torch.stack(xlist)#torch.concatenate(xlist)

        return res #Xvec0,Mlist


    def cumulativeMK(self, klist): 
        k0=klist[0]
   #     for l in range(1, len(klist)):
   #         k0=k0*(klist[l])
        return k0  
            
    def OneStep(self,ti, ti1, kvals_sig, Xti, Xti1, xin0): 
        nsteps=self.Nstep

        delt=(ti1-ti)/nsteps
        sqrt_delt=np.sqrt(delt)

        Nt=np.linspace(ti,ti1,nsteps+1)
        x0=torch.clone(xin0)

        for l in range(len(Nt)-1):
            x0=x0+self.Drift(ti,ti1, Nt[l],Xti, x0, Xti1, kvals_sig)*delt+torch.randn_like(x0)*sqrt_delt        
        return x0
            

    def Drift(self,ti,ti1,t,Xti,x, Xti1, kvals_sig):
        
        wi=self.Fi(ti,ti1,t,Xti,x, Xti1)
        wi_ext=torch.unsqueeze(wi*kvals_sig,dim=1)
        
        num=torch.mean((Xti1-x)*wi_ext,dim=0) 
        denom=torch.mean(wi*kvals_sig)

        res=num/(denom*(ti1-t))
        return res
  

    def Fi(self,ti,ti1,t,Xti,x, Xti1): 
        """
        Output is a tensor of shape B_{data}
        Now xi is not data
        """
        p1=-torch.norm(x-Xti1,dim=1)**2/(2.0*(ti1-t))
        p2=torch.norm((Xti1-Xti),dim=1)**2/(2.0*(ti1-ti))
        res=torch.exp(p1+p2)
        return res


    def KernelGaussianSig(self,xi_sig,Xvec_sig):
        """
        Returns the Mask, i.e. which elements to keep
        The kernel values
        """
        scaled_val=torch.norm(torch.tensor(xi_sig-Xvec_sig),dim=1)/self.h
        out=torch.exp(-scaled_val**2/2.0)#/self.h; 
        return out
        
    def genSig_one(self, xvec, tvec):
        """
        Input should be torch vector
        It should have at least two time_seq elements
        """
        xvec_stream=xvec.unsqueeze(0) ## np can deal with torch
        tvec_temp=np.expand_dims(tvec,axis=1)
        tvec_temp=np.expand_dims(tvec_temp,axis=0)
        xvec_t=np.concatenate((tvec_temp, xvec_stream),axis=2)
        isig_x=iisig.sig(xvec_t,self.depth)
        
        return isig_x

    def GenSignature_Xvec(self):
        t_temp=np.tile(self.tvec,(self.M, 1,1))
        t_temp=np.transpose(t_temp,(0,2,1))
        Xvect=np.concatenate((t_temp, self.Xtensor),axis=2)
        
        sig_list=[]
        for i in range(len(self.tvec)):
            sig_temp=iisig.sig(Xvect[:,0:i+1,:],self.depth)
            sig_list.append(sig_temp)
            
        sig_vec=np.stack(sig_list) 
        sig_vec=np.transpose(sig_vec,(1,0,2))
        return sig_vec
            

class timeSB_gaussian_VP(object):
    def __init__(self,Xtensor,h,tau,time,Ntime,Nstep): 
        self.Xtensor=Xtensor
        self.h=h
        self.tvec=np.linspace(0,time,Ntime)
        self.tau=tau
        self.Nstep=Nstep

    def mainsweep(self):
        tvec=self.tvec
        Klist=[]
        Mlist=[]
        xlist=[]

        ## The first iteration 
        Xvec0=self.Xtensor[:,0,]
        Xvec1=self.Xtensor[:,1,]

        ### Original input should be 0. 
        xin0=torch.zeros(self.Xtensor[0,0,].shape)
        xlist.append(xin0)  ### Attaching the current first value

        ## Obtain the mask and the kernel values 
        kvals=self.KernelGaussian(xin0, Xvec0)
        ### Data at preselected time intervals
        Klist.append(kvals) # t0 time kernel values
        #Mlist.append(mask)  # t0 time mask values

        ## Only the ones within range survives
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            kvals=self.KernelGaussian(xin0,Xvec0)
            Klist.append(kvals) # ti time kernel values

            
            kvals_masked=self.cumulativeMK(Klist) ## Don't have to call them kvals_masked, there is no mask. 
            Xvec0_new=Xvec0
            Xvec1_new=Xvec1

            ### What is the Xti in this case, the Xvec0 or the xi past
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_masked, Xvec0_new , Xvec1_new, xin0) # 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        res=torch.concatenate(xlist)
        return res 


    def cumulativeMK(self,klist): 
        k0=klist[0];
        for l in range(1,len(klist)):
            k0=k0*klist[l]
        return k0  
            
    def OneStep(self,ti, ti1, kvals, Xti, Xti1, xin0): 
        """
        what is the xti in this case, the given data or the simulated data along the way. 
        """
        nsteps=self.Nstep

        gamma_01in=self.Discount(ti,ti1)

        delt=(ti1-ti)/nsteps
        sqrt_delt=np.sqrt(delt)

        Nt=np.linspace(ti,ti1,nsteps+1)
        x0=torch.clone(xin0)

        for l in range(len(Nt)-1):
            gamma_t1in=self.Discount(Nt[l],ti1)
            beta_in=self.Beta(Nt[l])
            x0=x0+self.Drift(ti,ti1, Nt[l],Xti, x0, Xti1, kvals,gamma_01in, gamma_t1in, beta_in)*delt - 0.5*beta_in*x0*delt +\
            torch.randn_like(x0)*sqrt_delt*np.sqrt(beta_in)
        return x0
            

    def Drift(self,ti,ti1,t,Xti,x, Xti1, kvals, gamma_01, gamma_t1,beta_t):
        
        wi=self.Fi(ti,ti1,t,Xti,x, Xti1,gamma_01, gamma_t1)
        wi_ext=torch.unsqueeze(wi*kvals,dim=1)
        
        num=torch.mean((Xti1-x)*wi_ext,dim=0) 
        denom=torch.mean(wi*kvals)

        res=beta_t*gamma_t1*num/(denom*(1.0-gamma_t1**2))
        return res
  

    def Fi(self,ti,ti1,t,Xti,x, Xti1, gamma_01, gamma_t1): 
        """
        Output is a tensor of shape B_{data}
        Now xi is not data
        """
        #gamma_01=self.Discount(ti,ti1)
        #gamma_t1=self.Discount(t,ti1)
        
        p1=-torch.norm(x*gamma_t1-Xti1,dim=1)**2/(2.0*(1-gamma_t1**2))
        p2=torch.norm((Xti1-Xti*gamma_01),dim=1)**2/(2.0*(1-gamma_01**2))
        res=torch.exp(p1+p2)
        return res

    def Beta(self,t):
        beta_val=self.tau*np.exp(-self.tau*t)
        return beta_val

    def Discount(self,t0,t1): 
        temp0=np.exp(-self.tau*t0)-np.exp(-self.tau*t1)
        temp=np.exp(-0.5*temp0)
        return temp

    def KernelGaussian(self,xi,Xvec):
        """
        Returns a vector related to each input data point
        Even though h may get cancelled off, we still need it here for 
        scalability reasons
        """
        scaled_val=torch.norm(Xvec-xi,dim=1)/self.h
        out=torch.exp(-scaled_val**2/2.0)/self.h
        return out
        






