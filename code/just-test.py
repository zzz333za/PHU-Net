

import apex

from apex import amp
import os
import cv2
import glob
import pydicom
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from albumentations import PadIfNeeded,Compose, ShiftScaleRotate, Resize, CenterCrop, HorizontalFlip, RandomBrightnessContrast,RandomContrast,ShiftScaleRotate
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
import os
import random
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm as ttm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import warnings
from random import randint
warnings.filterwarnings("ignore")

import collections
from pprint import pprint
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import label,regionprops
#import lovasz_losses as L
import random
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from tor_pro import pro
from tor_pro import pro as pro1
from tor_pro001 import pro

def bc(img,th=0):
    x=img!=th
    x=x.sum(axis=0)
    x=(x>0)
    x1=list(x.squeeze()).index(1)
    x2=list(x[::-1].squeeze()).index(1)
    y=img!=th
    y=y.sum(axis=1)
    y=(y>0)
    y1=list(y).index(1)
    y2=list(y[::-1]).index(1)
    return img[y1:img.shape[0]-y2,x1:img.shape[1]-x2]
imgd=os.listdir('../ODIR-5K_Training_Dataset/')
l_train, l_valid =imgd[:6000],os.listdir('../valid/')




SMOOTH = 1e-6

n_epochs = 300
batch_size =48
os.environ['CUDA_VISIBLE_DEVICES']='0，1' 
device = torch.device("cuda:0")


  
#torch.save(model,'./model/newtor2.pth')  
#model=torch.load('./model/newtor2.pth')
model=torch.load('./model/tor221.pth')
from tor_pros import pro as pro1
from tor_pro001s import pro
from scipy.io import loadmat,savemat
n=9 
    
xi,xj,yi,yj=nm
s='s'+str(n)
path='./ISMRM2012/'+s+'.mat'
md=loadmat(path)
mf=md[s][0,0]['phasor']
path1='./ISMRM challenge/'+s[1:]+'.mat'
if n<10:
    path1='./ISMRM challenge/0'+s[1:]+'.mat'
md1=loadmat(path1)
mf1=md1['imDataParams']
mask=mf1[0][0][4]
#ff=ff[:,:,0]
#ma=ma*ma
ma=np.angle(mf)
#ma[np.abs(ff)<0.05*0.95*np.abs(ff).max()]=0
#ma=ma*mask
j=1
if n==15:
    j=2
mm=ma[:,:,j]
mask=mask[:,:,j]
plt.imshow(mm,cmap='gray')
plt.imshow(mm*mask,cmap='gray')

im=mm
img=im#-im.min()

resize_scale = width / max(img.shape[:2])
#img1=bc(img)
img1=img
a,b=img1.shape[0],img1.shape[1]


img1 = cv2.copyMakeBorder(img1,(256-a)//2,(256-(256-a)//2-a),(256-b)//2,(256-(256-b)//2-b),cv2.BORDER_CONSTANT,value=0)

img = img1.reshape([width,width,1])
img=img/(2*np.pi)
augmented =transform_test(image=img)
timg = augmented['image']

temp= pro1(timg,3,5,5,3,xi/10,yi/10)
timg = pro(timg,3,5,5,3,xj/10,yj/10)
timg=torch.cat([timg,temp[1:]],dim=0)
timg=timg.reshape([1,5,256,256]).to(device, dtype=torch.float)
#img=np.round(img*12)/12
with torch.no_grad():
    c=model(timg)#.cpu().numpy().squeeze()
    c=c[:,0]#+torch.tanh(c[:,1])
    c=c.cpu().numpy().squeeze()
    c[img.squeeze()==0]=0
#c=ac[0]
plt.imshow(img.squeeze(),cmap='gray')
plt.imshow(c.squeeze(),cmap='gray')
if '.' in str(yj):
    yj=int(yj*10)
plt.imshow(mask*(np.round(c.squeeze()-img.squeeze())+img.squeeze())[(256-a)//2:(256-a)//2+a,(256-b)//2:(256-b)//2+b],cmap='gray')
   
plt.imshow(np.floor(c.squeeze()-img.squeeze())+img.squeeze(),cmap='gray')
plt.imshow(np.round(c.squeeze()-img.squeeze()),cmap='gray')
plt.imshow((timg.squeeze().cpu()[0]).squeeze(),cmap='gray')
plt.imshow((timg.squeeze().cpu()[1]).squeeze(),cmap='gray')
plt.imshow((np.round(c[0])-np.round(c[1])+img)[0,:,:,0],cmap='gray')    


