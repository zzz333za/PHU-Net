#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 01:57:42 2019

@author: z
"""

import torch.nn.functional as F
from torch import autograd
import torch
import numpy as np
DK=[]
for i in [0,1,2,3,5,6,7,8]:
    one=np.zeros([3,3])
    one[1,1]=-1
    one[i//3,i%3]=1
    DK.append(one)
def mopool(g2c,h,w):
    for i in range(w*h):
     
        a=np.zeros([h,w])
        a[i//w,i%w]=1
        filters =torch.from_numpy(a.reshape([1,1,h,w]))
        if i==0:
            gx=F.conv2d(g2c, filters.float(), padding=(h//2,w//2))
            continue    
        gx=torch.cat([gx,F.conv2d(g2c, filters.float(), padding=(h//2,w//2))],dim=1)
    
    s=torch.mode(gx,dim=1)[0]
    return s
def mepool(g2c,h,w):
    for i in range(w*h):
     
        a=np.zeros([h,w])
        a[i//w,i%w]=1
        filters =torch.from_numpy(a.reshape([1,1,h,w]))
        if i==0:
            gx=F.conv2d(g2c, filters.float(), padding=(h//2,w//2))
            continue    
        gx=torch.cat([gx,F.conv2d(g2c, filters.float(), padding=(h//2,w//2))],dim=1)
    
    s=torch.median(gx,dim=1)[0]
    return s
def pro(im,h1,w1,h2,w2):
    im=im.reshape([-1,1,256,256])
 
    for i in range(8):
        
        filters =torch.from_numpy(DK[i].reshape([1,1,3,3]))
        if i==0:
            g=F.conv2d(im, filters.float(), padding=1)
            continue
        g=torch.cat([g,F.conv2d(im, filters.float(), padding=1)],dim=1)
    g1=((torch.min((g),dim=1)[0]>0.1)).float()
    im[g1.reshape([-1,1,256,256])==1]=0
    g1=((torch.min((-g),dim=1)[0]>0.1)).float()
    im[g1.reshape([-1,1,256,256])==1]=0
    #im=mepool(im,3,3).reshape([-1,1,256,256])
    #im[km.reshape([-1,1,256,256])==1]=0
    for i in range(8):
        
        filters =torch.from_numpy(DK[i].reshape([1,1,3,3]))
        if i==0:
            g=F.conv2d(im, filters.float(), padding=1)
            continue
        g=torch.cat([g,F.conv2d(im, filters.float(), padding=1)],dim=1)
    s1=zong(im,g,h2,w2).reshape([-1,1,256,256])
    s2=heng(im,g,h1,w1).reshape([-1,1,256,256])
    img=torch.cat([im,s1,s2],dim=1).reshape([-1,256,256])
    return img
#g1=torch.cat([g[:,1]-g[:,5],g[:,1]-g[:,6],g[:,1]-g[:,7]],dim=0)
#g=g[:,1]-g[:,6]
#g=g.reshape([1,1,256,256])
#g1=torch.cat([g[:,6]-g[:,0],g[:,6]-g[:,1],g[:,6]-g[:,2]],dim=0)
#g1=((torch.max((g1),dim=0)[0]>0.95)).float()
def zong(im,g,h,w):
    g1=(g[:,3]>0.3).float()
    
    g1=g1.reshape([-1,1,256,256])
    gi=im<-0.3
    gi=gi.reshape([-1,1,256,256])
    #g1[gi==0]=0

    g2=(g[:,3]<-0.3).float()
    
    
    g2=g2.reshape([-1,1,256,256])
    gi=(im)>0.3
    gi=gi.reshape([-1,1,256,256])
    #g2[gi==0]=0
    
    gg=g2-g1
    
    g2=gg
    filters =torch.from_numpy(np.ones([5,5]).reshape([1,1,5,5]))
    g2=F.conv2d(g2, filters.float(), padding=2)
    g2=F.conv2d(g2, filters.float(), padding=2)
    #plt.imshow(g2.squeeze()>0)
    #plt.imshow(g2.squeeze()<0)
    g2=g2.float()
    
    filters =torch.from_numpy(np.array([[0,0,0],[-99,1,0],[0,0,0]]).reshape([1,1,3,3]))
    g41=F.conv2d((g2>0).float(), filters.float(), padding=1)>0
    g42=F.conv2d((g2<0).float(), filters.float(), padding=1)>0
    #g2=g2>1
    #plt.imshow(g41.float().squeeze()-g42.float().squeeze())
    g2=(g41).float()-(g42).float()

    #g2c=torch.cumsum(g2,dim=3)
    g2c=torch.cumsum(g2[:,:,:,128:],dim=3)
    g2c1=torch.cumsum(g2[:,:,:,:128],dim=3)
    g2c=torch.cat([(g2c1[:,:,:,127].reshape([-1,1,256,1]).repeat(1,1,1,128)-g2c1),g2c],dim=3)

    
    s=mopool(g2c,h,w).reshape([-1,1,256,256])
    #s=g2c
    #plt.imshow(s.squeeze())
    #s=mopool(s.reshape([-1,1,256,256]),h,w).reshape([-1,1,256,256])
    s[im==0]=0
    return s
def heng(im,g,h,w):
    g1=(g[:,1]>0.3).float()

    g1=g1.reshape([-1,1,256,256])
    gi=im<-0.3
    gi=gi.reshape([-1,1,256,256])
    #g1[gi==0]=0
    
   
    g2=(g[:,1]<-0.3).float()
    
    g2=g2.reshape([-1,1,256,256])
    gi=(im)>0.3
    gi=gi.reshape([-1,1,256,256])
    #g2[gi==0]=0
    
    gg=g2-g1
    
    g2=gg
    filters =torch.from_numpy(np.ones([5,5]).reshape([1,1,5,5]))
    g2=F.conv2d(g2, filters.float(), padding=2)
    g2=F.conv2d(g2, filters.float(), padding=2)
    
    g2=g2.float()
    
    filters =torch.from_numpy(np.array([[0,-99,0],[0,1,0],[0,0,0]]).reshape([1,1,3,3]))
    g41=F.conv2d((g2>0).float(), filters.float(), padding=1)>0
    g42=F.conv2d((g2<0).float(), filters.float(), padding=1)>0
    #g2=g2>1
    
    g2=(g41).float()-(g42).float()
    
    #g2c=torch.cumsum(g2,dim=2)
    g2c=torch.cumsum(g2[:,:,128:,:],dim=2)
    g2c1=torch.cumsum(g2[:,:,:128,:],dim=2)
    g2c=torch.cat([-(g2c1[:,:,127,:].reshape([-1,1,1,256]).repeat(1,1,128,1)-g2c1),g2c],dim=2)
    
    
    s=mopool(g2c,h,w).reshape([-1,1,256,256])
    
    #s=mopool(s.reshape([-1,1,256,256]),3,9).reshape([-1,1,256,256])
    s[im==0]=0
    return s