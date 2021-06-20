# PHU-Net
# The PHU-NET: A robust phase unwrapping method for magnetic resonance imaging based on deep learning 

# Data:
For simulation and training ODIR2019: https://odir2019.grand-challenge.org/  
Validation data ISMRM2012-challenge https://www.ismrm.org/workshops/FatWater12/data.htm

Pretrain_weight:https://pan.baidu.com/s/18wr-pRAWeKpT1T_hAihMNQ 
pwd：phun 

# Environment
python3
CUDA 10.1
pytorch 18.1
tensorflow 1.4.1 & keras 2.3.0 (just for data augmentation)
open-cv2
segmentation_models_pytorch
skimage
albumentations
pydicom

# How to train
1.prepare train/valid data
2.python train-test.py

# How to test
python just-test.py
