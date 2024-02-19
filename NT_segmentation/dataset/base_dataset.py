import numpy as np
import pandas as pd 
import os
import torch
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from tqdm.auto import tqdm
import cv2
import torch.nn.functional as F
import random
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import glob
from albumentations.core.composition import Compose, OneOf
import torchvision as tv
from monai.losses.dice import DiceLoss, one_hot
from monai.metrics import DiceMetric


class SegDataset(Dataset):
  def __init__(self,img_path,img_labels,mode=None,train_x_transform =None,train_y_transform = None,seed = None):
     self.img_path = img_path
     self.img_labels = img_labels
     self.train_x_transform = train_x_transform
     self.train_y_transform = train_y_transform
     self.mode= mode
     self.seed =seed

     path_train = self.img_path
     path_label = self.img_labels
     
     self.filenames_train = glob.glob(path_train + '/*.png')
     self.filenames_label = glob.glob(path_label + '/*.png')  
     a = []
     for i in range(len(self.filenames_label)):
          a.append(self.filenames_label[i][-30:])
     
     for i in range(len(self.filenames_train)):
        name = self.filenames_train[i][-30:]

  def __len__(self):
        return len(self.filenames_train)

  def __getitem__(self, idx):
    image = self.filenames_train[idx]  #하나씩 읽어 들려옴
    label = self.filenames_label[idx]
    
    images = Image.open(image).convert('L')#channel 1 
    labels = Image.open(label).convert('L')#각 이미지가 한장 씩 들어 온다 #정상 -> RGB
    

    #label의 이미지를 Class별로 나누어 주기 위하여
    img=np.array(labels)#이미지를 Numpy로 변환 정상torch.Size([1, 704, 1280])
    label_tensor = np.zeros_like(img[:, :])
    label_tensor[(img[:, :] == 0) ] = 0
    label_tensor[(img[:, :] == 1) ] = 1
    label_tensor[(img[:, :] == 2)  ] = 2
    #     label_tensor[(img[:, :] == 0)  ] = 0
    #     label_tensor[(img[:, :] == 64)  ] = 1
    #     label_tensor[(img[:, :] == 128)  ] = 2
    #     label_tensor[(img[:, :] == 32)  ] = 2


    label = torch.tensor(label_tensor)#tensor로 변경


    from typing import Optional #label-tensor를 One-Hot-encodeing을 시킨 부분
    def label_to_one_hot_label(labels: torch.Tensor,num_classes: int,eps: float = 1e-6,ignore_index=100 ,device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = torch.int64):
          shape = labels.shape
          labels = labels.type(torch.int64)
          one_hot = torch.zeros((num_classes,) + shape, device=device, dtype=dtype)
          labels = labels.type(torch.int64)
          ret = one_hot.scatter_(0,labels.unsqueeze(0),1.0)+eps
    
          return ret
    
    img = label_to_one_hot_label(label, num_classes=3)#torch.Size([4, 704, 1280])

    if self.train_x_transform: 
      torch.manual_seed(10)
      images = self.train_x_transform(images)
    if self.train_y_transform:
      torch.manual_seed(10)
      label = self.train_y_transform(img)

    return images,label