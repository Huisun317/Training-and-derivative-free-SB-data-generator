#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:03:46 2025

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


###########################################################################
#                          #Variance Exploding                            #
#                                                                         #
###########################################################################


#### The original timeseries  
class timeSB_OG_Quartic(object):
    def __init__(self, Xtensor,h,time,Ntime,Nstep): 
        """
        Xtensor: The input dataset should be of the shape B x Ntime x dim 
        h: the kernel radius 
        time: total time horizon
        Nstep: the total simulation time step for each data time interval 
        """
        self.Xtensor=Xtensor
        self.h=h
        self.Nstep=Nstep
        self.tvec=np.linspace(0,time,Ntime)

    def mainsweep(self):
        
        tvec=self.tvec
        Klist=[] ## The list of kernels, after each data time stepping, Klist should add one new kernel 
        Mlist=[] ## Masks. Note that this is due to we are using the Quartic kernel. The data far away from the input will be killed. 
        xlist=[] ## To track the simulated path. 

        ## The first iteration 
        Xvec0=self.Xtensor[:,0,]
        Xvec1=self.Xtensor[:,1,]

        xin0=torch.ones(self.Xtensor[0,0,].shape)*self.Xtensor[:,0,:].mean() ## initializing the first value to align it with the mean of the data
        xlist.append(xin0)  ### Attaching the current first value

        ## Obtain the mask and the kernel values 
        ## Masks are the indices of the survival data
        kvals,mask=self.KernelMask0(xin0, Xvec0)
        ### Data at preselected time intervals
        Klist.append(kvals) # t0 time kernel values

        ## Only the ones within range survives
        kvals=kvals[mask]
        Xvec0=Xvec0[mask]
        Xvec1=Xvec1[mask]
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            kvals,mask=self.KernelMask0(xin0,Xvec0)
            Klist.append(kvals) # the new ti time kernel values, attach it to the list since there is a growing product.  
            Mlist.append(mask)  # the new ti time mask values, attach it to the list since there is a growing product. 

            ## find the cumulative mask at time i: this corresponds to the product term in the drift
            kvals_masked, mask_vals=self.cumulativeMK(Mlist, Klist) 
            ## Find the X_{ti} vector to be used 
            Xvec0_new=Xvec0[mask_vals]
            ## Find the X_{t_{i+1}} vector to be used
            Xvec1_new=Xvec1[mask_vals]
            
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_masked, Xvec0_new , Xvec1_new, xin0) 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        ### Make it a time list. 
        res=torch.concatenate(xlist)

        return res #Xvec0,Mlist


    def cumulativeMK(self,mlist, klist): 
        m0=mlist[0];
        for l in range(1,len(mlist)):
            m0=m0*mlist[l]
            
        k0=klist[0][m0]
        for l in range(1, len(klist)):
            k0=k0*(klist[l][m0])

        return k0,m0   
            
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


    def KernelMask0(self,xi,Xvec):
        """
        Returns the Mask, i.e. which elements to keep
        The kernel values
        """
        scaled_val=torch.norm(xi-Xvec,dim=1)/self.h
        mask=(scaled_val)<=1;
        out=(1.0-scaled_val**2)**2#/self.h; 
        return out, mask

###########################################################################
#                          #Variance Preserving                           #
#                                                                         #
###########################################################################
class timeSB_VP_Quartic(object):
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
        
        xin0=torch.ones(self.Xtensor[0,0,].shape)*self.Xtensor[:,0,:].mean()
        xlist.append(xin0)  ### Attaching the current first value

        ## Obtain the mask and the kernel values 
        kvals,mask=self.KernelMask0(xin0, Xvec0)
        ### Data at preselected time intervals
        Klist.append(kvals) # t0 time kernel values
        Mlist.append(mask)  # t0 time mask values

        ## Only the ones within range survives
        kvals=kvals[mask]
        Xvec0=Xvec0[mask]
        Xvec1=Xvec1[mask]
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            kvals,mask=self.KernelMask0(xin0,Xvec0)
            Klist.append(kvals) # ti time kernel values
            Mlist.append(mask)  # ti time mask values

            
            kvals_masked, mask_vals=self.cumulativeMK(Mlist, Klist)
            Xvec0_new=Xvec0[mask_vals]
            Xvec1_new=Xvec1[mask_vals]
            
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_masked,Xvec0_new , Xvec1_new, xin0) # 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        res=torch.concatenate(xlist)

        return res #Xvec0,Mlist


    def cumulativeMK(self,mlist, klist): 
        m0=mlist[0];
        for l in range(1,len(mlist)):
            m0=m0*mlist[l]
            
        k0=klist[0][m0]
        for l in range(1, len(klist)):
            k0=k0*(klist[l][m0])
        return k0,m0   
            
    def OneStep(self,ti, ti1, kvals, Xti, Xti1, xin0): 
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


    def KernelMask0(self,xi,Xvec):
        """
        Returns the Mask, i.e. which elements to keep
        The kernel values
        """
        scaled_val=torch.norm(xi-Xvec,dim=1)/self.h
        mask=(scaled_val)<=1;
        out=(1.0-scaled_val**2)**2#/self.h; 
        return out, mask


###########################################################################
#                          #Signature as features                         #
#                                                                         #
###########################################################################


class timeSB_Sig_Quartic(object):
    def __init__(self,Xtensor,h,time,Ntime,Nstep): 
        self.Xtensor=Xtensor
        self.dim=Xtensor.shape[-1]
        
        self.h=h
        self.tvec=np.linspace(0,time,Ntime)
        self.depth=3
        self.Nstep=Nstep
        self.M_data=self.Xtensor.shape[0]
        self.sig_xt=self.GenSignature_Xvec()
        

    def mainsweep(self):
        tvec=self.tvec
        Klist=[]
        Mlist=[]
        xlist=[]

        ## The first iteration 
        Xvec0=self.Xtensor[:,0,]
        Xvec1=self.Xtensor[:,1,]

        ### Original input should be 0. 
        #xin0=torch.zeros(self.Xtensor[0,0,].shape)
        xin0=torch.ones(self.Xtensor[0,0,].shape)*self.Xtensor[:,0,:].mean()
        xlist.append(xin0)  ### Attaching the current first value

        ## Obtain the mask and the kernel values 
        kvals,mask=self.KernelMask0(xin0, Xvec0)
        ### Data at preselected time intervals
        Klist.append(kvals) # t0 time kernel values
        Mlist.append(mask)  # t0 time mask values

        ## Only the ones within range survives
        kvals=kvals[mask]
        Xvec0=Xvec0[mask]
        Xvec1=Xvec1[mask]
        
        xin1=self.OneStep(tvec[0], tvec[1], kvals, Xvec0, Xvec1, xin0) #Generate one sample xt1

        xlist.append(xin1)
        xin0=xin1 ## obtain next result xt1 and update

        ### First ignore this part and test. 
        
        for i in range(1,len(tvec)-1):#1, len(tvec)-1
            Xvec0=self.Xtensor[:,i,]
            Xvec1=self.Xtensor[:,i+1,]

            xin0=xlist[-1]

            ### Collecting the new mask and the total kernel values
            kvals,mask=self.KernelMask0(xin0,Xvec0)
            Klist.append(kvals) # ti time kernel values
            Mlist.append(mask)  # ti time mask values

            
            kvals_masked, mask_vals=self.cumulativeMK(Mlist, Klist)
            Xvec0_new=Xvec0[mask_vals]
            Xvec1_new=Xvec1[mask_vals]
            
            xin1=self.OneStep(tvec[i], tvec[i+1], kvals_masked,Xvec0_new , Xvec1_new, xin0) # 
    
            xlist.append(xin1)
            xin0=xin1         ## obtain next result xt1 and update

        res=torch.concatenate(xlist)

        return res #Xvec0,Mlist


    def cumulativeMK(self,mlist, klist): 
        m0=mlist[0];
        for l in range(1,len(mlist)):
            m0=m0*mlist[l]
            
        k0=klist[0][m0]
        for l in range(1, len(klist)):
            k0=k0*(klist[l][m0])

        return k0,m0   
            
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


    def KernelMask0(self,xi,Xvec):
        """
        Returns the Mask, i.e. which elements to keep
        The kernel values
        """
        scaled_val=torch.norm(xi-Xvec,dim=1)/self.h
        mask=(scaled_val)<=1;
        out=(1.0-scaled_val**2)**2#/self.h; 
        return out, mask
    
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
        """
        Generate signature features
        """
        t_temp=np.tile(self.tvec,(self.M_data, 1,1))
        t_temp=np.transpose(t_temp,(0,2,1))
        Xvect=np.concatenate((t_temp, self.Xtensor),axis=2)
        
        sig_list=[]
        for i in range(len(self.tvec)):
            sig_temp=iisig.sig(Xvect[:,0:i+1,:],self.depth)
            sig_list.append(sig_temp)
            
        sig_vec=np.stack(sig_list) 
        sig_vec=np.transpose(sig_vec,(1,0,2))
        return sig_vec
            