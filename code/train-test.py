

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


width=256
x_train=l_train
X_valid=l_valid
q=np.zeros([width,width])
q1=np.zeros([width,width])
q2=np.zeros([width,width])
q3=np.zeros([width,width])
q4=np.zeros([width,width])
q5=np.zeros([width,width])
q6=np.zeros([width,width])
q7=np.zeros([width,width])
for i in range(width):
    for j in range(width):
        q[i,j]=(i+j)/2
        q1[i,j]=i
        q2[i,j]=j
        q3[i,j]=((i-127)**2+(j-127)**2)**0.5
        q4[i,j]=((i)**2+(j)**2)**0.5
        q5[i,j]=((width-1-i)**2+(j)**2)**0.5
        q6[i,j]=((i)**2+(width-j)**2)**0.5

qs=[q,q1,q2,q4,q5,q6,(q4+q5),(q4+q6),(q5+q6),q3+q4,q2+q4,q1+q4]

class VDataset(Dataset):

    def __init__(self, path, labels, transform=None):
        
        #self.path = path
        self.data = path
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, i):
        
        j=self.data[i]
        #j=random.choice(X_valid)
        img=np.load("../valid/"+j,allow_pickle=True)
        
        a=img[1]
        b=img[2]
        quchu=img[0].squeeze()==0
        b[quchu]=0
        a[quchu]=0
        b=b.reshape([width,width,1])
        #X1.append(a.reshape([width,width,1]))
        #X2.append(b.reshape([width,width,1]))
        a=a.reshape([width,width,1])
        kb=b[127,127]
        b=b-kb[0]
        b[quchu]=0
        b1=b.copy()
        b1[b<0]=0
        b2=-1*b.copy()
        b2[b2<0]=0  
        c=a+b
       
        augmented =self.transform(image=a)
        img = augmented['image']
        #img=img.repeat(3,1,1)
        timg=img
        temp= pro1(timg,3,5,5,3)
        timg = pro(timg,3,5,5,3)
        timg=torch.cat([timg,temp[1:]],dim=0)
        return {'image': timg, 'labels': [torch.tensor(c),torch.tensor(b1),torch.tensor(b2)]}    
        
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
			rotation_range=180,
			width_shift_range=0.2,
			height_shift_range=0.2,
		
			zoom_range=0.2,         
			horizontal_flip=True,
			vertical_flip=True,
            fill_mode="constant",cval=0
            #brightness_range=[0.7,1.3],
	
			#channel_shift_range=20.  
			)

class IntracranialDataset(Dataset):

    def __init__(self, path, labels, transform=None):
        
        #self.path = path
        self.data = path
        self.transform = transform
        self.labels = labels

    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, i):
        
        j=self.data[i]
        #j=random.choice(x_train)
        chk=randint(0,4)
        if chk<5:
            img=cv2.imread('../ODIR-5K_Training_Dataset/'+j,0)
            img1=bc(img)
            plt.imshow(img1)
      
            n=randint(180,400)
            m=randint(180,400)
            img1 = cv2.resize(img1,(m,n))
       
            a,b=img1.shape[0],img1.shape[1]
            if max(a,b)>=width:
                a1=randint(0,max(a-width,0))
                b1=randint(0,max(b-width,0))
                img1=img1[a1:min(a,a1+width),b1:min(b,b1+width)]
            
            a,b=img1.shape[0],img1.shape[1]
            if randint(0,5)>1:
                qu=img1==0
       
               
                dst=cv2.resize(random.choice(qs),(b,a))
                if randint(0,1)==1:
                    img1[img1>0]=30
                '''if randint(0,3)==0:
                    o1=randint(1,b-1)
                    o2=randint(1,a-1)
                    for l1 in range(b):
                        for l2 in range(a):
                            dst[l2,l1]=255-np.sqrt((l1-o1)**2+(l2-o2)**2)
                    dst=dst/randint(1,10)'''
                img1=(img1*dst)
                img1=(img1/img1.max())*255
                img1[qu]=0

            img = cv2.copyMakeBorder(img1.astype(int),(width-a)//2,(width-(width-a)//2-a),(width-b)//2,(width-(width-b)//2-b),cv2.BORDER_CONSTANT,value=0)

            #img = cv2.resize(img1,(width,width))
        
            #if randint(0,3)==3:
             #   img=cv2.equalizeHist(img.astype('uint8'))
            #img = cv2.resize(img1,(width,width))
            #img=cv2.equalizeHist(img)
            
          
            img = img.reshape([1,width,width,1])  
            #x1 = datagen.flow(img,batch_size=1).next()[0]
            #img=x1
            
          
            img = img.reshape([width,width,1])  
            img=255*(img/(img.max()-img.min()))
            #img[img<0]=0
            img=img.astype('uint8')
            
           
            if randint(0,5)>4:
                tem=np.zeros([width,width])
                tem=cv2.cv2.ellipse(tem,(randint(50,width-50),randint(50,width-50)),(randint(0,100),randint(0,100)),0,0,randint(0,360),1,-1)
                img[tem.astype(int)]=0
            augmented =transform_train (image=img)
            img = augmented['image'] 
            #img=img*255
       
    
            quchu=img.squeeze()==0
            quchu=quchu.reshape([256,256,1])
            #img=img.astype(int)
            img[quchu]=0
            a=randint(1,80)/4
    
            timg=((img)/a)%(2*np.pi)-np.pi
            timg=timg/(2*np.pi)
       
            timg[quchu]=0

            b=np.round((((img)/a-255/(2*a))/(2*np.pi))-timg)
            #print(np.unique(b))
    
            b[quchu]=0
            
            a=timg
            b=b.squeeze()
            a=timg.reshape([width,width,1])
            a=a.squeeze()
            kb=b[127,127]
            b=b-kb
       
            b[quchu.squeeze()]=0
            augmented =transform_test(image=b.reshape([width,width,1]))
            b = augmented['image']  
            b1=b.clone()
            b1[b<0]=0
            b2=-1*b.clone()
            b2[b2<0]=0
            #b=b.reshape([width,width,1])
            #nb=randint(10,25)
            nb=20
            nor=nb*np.random.randn(256,256)+1j*nb*np.random.randn(256,256)
            no=np.angle(nor)
            no[quchu.squeeze()]=0
            
            timg=a.reshape([width,width,1])
            timg0=timg.copy()
            if randint(0,2)<0:
                timg=(timg0*2*np.pi+no.reshape([width,width,1]))/(2*np.pi)
            if randint(0,2)<0:
                    timg[quchu]=0
            augmented =transform_test(image=timg)
            timg = augmented['image']
            augmented =transform_test(image=timg0)
            timg0 = augmented['image']
            c=timg+b
            #timg=timg.repeat(3,1,1)
            temp= pro1(timg,3,5,5,3)
            timg = pro(timg,3,5,5,3)
            timg=torch.cat([timg,temp[1:]],dim=0)
            c[timg[0].reshape([1,width,width])==0]=0
            return {'image': timg, 'labels': [c,torch.tensor(b1),torch.tensor(b2)]}    
     

    
transform_train = Compose([#Resize(width, width),#CenterCrop(200, 200),
                           #Resize(224, 224),
                           
                           HorizontalFlip(),
                           #RandomBrightnessContrast(),
    ShiftScaleRotate(rotate_limit=180,p=0.3,border_mode=cv2.BORDER_CONSTANT,value=0),
    #ToTensor()
])

transform_test= Compose([#Resize(width,width),#CenterCrop(200, 200),
                         #Resize(224, 224),
    ToTensor()
])


SMOOTH = 1e-6

n_epochs = 300
batch_size =48
os.environ['CUDA_VISIBLE_DEVICES']='0，1' 
device = torch.device("cuda:0")

aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation=None,      # activation function, default is None
    classes=1,                 # define number of output labels
)

model =smp.PAN('efficientnet-b5', classes=1,in_channels=5, activation=None,encoder_weights='imagenet')
#model.fc = torch.nn.Linear(2048, 6)
#model1=torch.load('./model/newtor.pth')
#model.load_state_dict(model1.state_dict())
#del model1
model.to(device)
#model=torch.load('../model/t2b.pth')
#weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0]).cuda()
criterion = torch.nn.L1Loss()
criterion1 =  torch.nn.BCEWithLogitsLoss()
plist = [{'params': model.parameters(), 'lr': 2e-4}]
optimizer = optim.Adam(plist, lr=2e-4)


cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
model.to('cuda')
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
model = torch.nn.DataParallel(model).cuda()
l0=[]
l1=[]
lo=[]
lm=[]
#model=torch.load('../model/2db.pth')
train_dataset = IntracranialDataset(
 path=x_train, transform=transform_train, labels=True)
data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
val_dataset = VDataset(
 path=X_valid, transform=transform_test, labels=True)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32) 
val_loss=0
va=10
for epoch in range(n_epochs):
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    model.train()    
    tr_loss = 0

    tk0 = tqdm(data_loader_train, desc="Iteration")
    n=0
    for step, batch in enumerate(tk0):
        #break
        if step%100==0:
            print(tk0)
        inputs = batch["image"]
        az=inputs[:,0,:,:].reshape([inputs.shape[0], -1, 256, 256] )
        labels = batch["labels"]
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels[0].to(device, dtype=torch.float).reshape([inputs.shape[0], -1, 256, 256] )
          
        out = model(inputs)

        out=out[:,0].reshape([-1,1,256,256])
        
        out[az==0]=0

       
        #out2=out
        #label1=((labels-out+1)%1).float()
        #out=out2+torch.tanh(out1*np.pi*2)*0.5+0.5

        az=az.to(device, dtype=torch.float).squeeze()
        la=labels.squeeze()-az
        la[az[:,0].squeeze()==0]=0
        oa=out.squeeze()-az
        oa[az[:,0].squeeze()==0]=0
        ph1=la[:,:-1,:]-la[:,1:,:]
        ph2=la[:,:,:-1]-la[:,:,1:]
        lk1=oa[:,:-1,:]-oa[:,1:,:]
        lk2=oa[:,:,:-1]-oa[:,:,1:]
        ph1=ph1.float()
        ph2=ph2.float()
        lk1=lk1.float()
        lk2=lk2.float()
        #loss =criterion1(out, la)+criterion(out, cl)+criterion(ph1,lk1)+criterion(ph2,lk2)        
        loss =criterion(out, labels)+criterion(ph1,lk1)+criterion(ph2,lk2)    #+criterion1(p(out),(p(labels)>0).float())
        #labels[az.reshape([inputs.shape[0], -1, 256, 256] )==0]=0
       
        tr_loss += loss.item()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))


    
    if epoch>=0:
        tk0 = tqdm(data_loader_val, desc="Iteration")
        model.eval()
        val_loss=0
        iou=0
        n=0
        for step, batch in enumerate(tk0):
            
            inputs = batch["image"]
            labels = batch["labels"]
            az=inputs[:,0,:,:].reshape([inputs.shape[0], -1, 256, 256] )
          
          
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels[0].to(device, dtype=torch.float).reshape([inputs.shape[0], -1, 256, 256] )
            with torch.no_grad():
                out = model(inputs)
       
                out=out[:,0].reshape([-1,1,256,256])
                
                out[az==0]=0
             
            #loss = criterion(out, labels)#+criterion1(out1, ((labels.reshape([-1,1,256,256])-out)>=1).float())+criterion(out+torch.sigmoid(out1), labels)
            #outputs=torch.sigmoid(outputs).long()
            az=az.to(device, dtype=torch.float).squeeze()
            la=labels.squeeze()-az
            la[az[:,0].squeeze()==0]=0
            oa=out.squeeze()-az
            oa[az[:,0].squeeze()==0]=0
            ph1=la[:,:-1,:]-la[:,1:,:]
            ph2=la[:,:,:-1]-la[:,:,1:]
            lk1=oa[:,:-1,:]-oa[:,1:,:]
            lk2=oa[:,:,:-1]-oa[:,:,1:]
            ph1=ph1.float()
            ph2=ph2.float()
            lk1=lk1.float()
            lk2=lk2.float()
            #loss =criterion1(out, la)+criterion(out, cl)+criterion(ph1,lk1)+criterion(ph2,lk2)        
            loss =criterion(out, labels)+criterion(ph1,lk1)+criterion(ph2,lk2) 

            val_loss += loss.item()
            
            n=n+1
        
        epoch_loss = val_loss / len(data_loader_val)
        #epoch_iou = np.mean(np.array(iou.tolist()) / len(x_train))
        print('val Loss: {:.4f}'.format(np.mean(epoch_loss)))
        if epoch%1==0 and np.mean(epoch_loss)<va:
            va=np.mean(epoch_loss)
            torch.save(model,'./model/zx001addtor- {:.1f}-{:.4f}.pth'.format(epoch,epoch_loss))
            aa='./model/zx001addtor- {:.1f}-{:.4f}.pth'.format(epoch,epoch_loss)
        lm.append(aa)
        lo.append(epoch_loss)
    
     
  
#torch.save(model,'./model/newtor2.pth')  
model=torch.load(aa)
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


