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

#GPU 확인
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

#-----------------------------------------------------------------------------CustomDataset-------------------------------------------------------------------------------------------
class SegDataset(Dataset):
  def __init__(self,img_path,img_labels,mode=None,transform_x =None,transform_y = None,seed = None):
     self.img_path = img_path
     self.img_labels = img_labels
     self.transform_x = transform_x
     self.transform_y = transform_y
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
    labels = Image.open(label).convert('RGB')#각 이미지가 한장 씩 들어 온다 #정상 -> RGB

    #label의 이미지를 Class별로 나누어 주기 위하여
    img=np.array(labels)#이미지를 Numpy로 변환 정상torch.Size([1, 704, 1280])
    label_tensor = np.zeros_like(img[:, :, 0])
    label_tensor[(img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0) ] = 0
    label_tensor[(img[:, :, 0] == 128) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0) ] = 1
    label_tensor[(img[:, :, 0] == 128) & (img[:, :, 1] == 128) & (img[:, :, 2] == 0) ] = 2
    label_tensor[(img[:, :, 0] == 0) & (img[:, :, 1] == 128) & (img[:, :, 2] == 0)] = 2 #RGB순으로 0,1,2

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

#     x = np.array(images)#Numpy로
#     images = x
#     images = torch.Tensor(images)
#     images = images.unsqueeze(0)

    if self.transform_x:
      torch.manual_seed(10)
      images = self.transform_x(images)
    if self.transform_x:
      torch.manual_seed(10)
      label = self.transform_y(img)

      images = images.permute(1,2,0)
      plt.imshow(images,cmap='gray')
      plt.show()
      label = label.permute(1,2,0)
      plt.imshow(label,cmap='gray')
      plt.show()

    return images,label
#----------------------------------------------------------------------------- Transform과 DataLoader ------------------------------------------------------------------------------

train_x_transforms= transforms.Compose([
    transforms.Resize((320,352)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.236356),(0.1891466))
     ])    

train_y_transforms= transforms.Compose([
    transforms.Resize((320,352)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.33333343 ,0.33333343, 0.33333343),(0.47129032, 0.47128868, 0.47129023)),
     ])    

val_x_transforms = transforms.Compose([
    transforms.Resize((320,352)),
    transforms.ToTensor(),
    transforms.Normalize((0.236356),(0.1891466))
#     transforms.Normalize((60.62533 ),(47.828457)),
     ])    

val_y_transforms = transforms.Compose([
    transforms.Resize((320,352)),
    transforms.ToTensor(),
    transforms.Normalize((0.33333343 ,0.33333343, 0.33333343),(0.47129032, 0.47128868, 0.47129023)),
     ])    


train_img_path = "D:/NT_GT_230720/train/NT_image/NT_VG"
train_img_label = "D:/NT_GT_230720/train/NT_label/label_VG"

train_dataset = SegDataset(train_img_path,train_img_label,'train',transform_x=train_x_transforms,transform_y=train_y_transforms)#batch 16일 때 이미지 1개씩 SegDataset에 들어감
train_loader = DataLoader(train_dataset,batch_size=10)

val_img_path = "D:/NT_GT_230720/val/NT_image/NT_VG"
val_img_label ="D:/NT_GT_230720/val/NT_label/label_NT_VG"

val_dataset = SegDataset(val_img_path,val_img_label,'val',transform_x=val_x_transforms,transform_y=train_y_transforms)
val_loader = DataLoader(val_dataset,batch_size=1)

test_img_path = "D:/NT_GT_230720/Test/NT_VG"
test_dataset = SegDataset(test_img_path,'test',transform_x=val_x_transforms,transform_y=train_y_transforms )
test_loader = DataLoader(test_dataset,batch_size=1)

#-----------------------------------------------------------------------------모델 정의--------------------------------------------------------------------------------------------
import segmentation_models_pytorch as smp #smp라는 라이브러리

model = smp.Unet( #Unet인데 encoder backbone이 resnet50
    #resnet50
    encoder_name="resnet18",        # choose encoder
    encoder_weights="imagenet" ,
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3
                   # use `imagenet` pre-trained weights for encoder initialization                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                   # model output channels (number of classes in your dataset)
)

model.to(device)


#-----------------------------------------------------------------------------Train--------------------------------------------------------------------------------------------------

# criterion = nn.CrossEntropyLoss()#loss
criterion = DiceLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
n_epochs = 30
train_loss = 0
valid_loss = 0
train_acc = 0
valid_acc = 0

loss_hist = []
loss_np =[]

cnt2 = 0
loss1 = []
loss2 = []
loss1_x = []
loss2_x = []
for e  in range(0,n_epochs):
      #train_model
      model.train()
      cnt = 0
      n = len(train_loader)
      for images,label in train_loader:
          cnt += 1
          print(cnt)
          images,labels = images.to(device), label.to(device)
          labels = labels.type(torch.IntTensor).to(device)
          optimizer.zero_grad()
          logits = model(images)
          softmax = nn.Softmax(dim=1)
          softmax_a = softmax(logits)
          output = softmax_a
          loss = criterion(output, labels)# #torch.Size([16, 2, 256, 256]),#torch.Size([16, 256, 256]) flaot32,torch.uint
          output = torch.argmax(output, dim=1, keepdim=True)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()

      temp = output.clone().cpu()
      label1 = labels.clone().cpu()
      image1 = images.clone().cpu()

      cnt2 = cnt2 + 1
      def multi_class_to_image(output_f):#label one-hot coding을 이미지로 변환
                  height = 320
                  width = 352
                  image1 = np.zeros((height, width, 3), dtype=np.uint8)

                  for k1 in range(320):
                        for l1 in range(352):

                              if output_f[0,0,k1,l1] == 0:
                                   image1[k1,l1,0] = 0
                                   image1[k1,l1,1] = 0
                                   image1[k1,l1,2] = 0

                              elif output_f[0,0,k1,l1] == 1:
                                   image1[k1,l1,0] = 128
                                   image1[k1,l1,1] = 0
                                   image1[k1,l1,2] = 0
                              
                              elif output_f[0,0,k1,l1] == 2:
                                   image1[k1,l1,0] = 128
                                   image1[k1,l1,1] = 128
                                   image1[k1,l1,2] = 0
          
                  return Image.fromarray(image1.astype('uint8'))
      
      
      def label_trans(label1):#label one-hot coding을 이미지로 변환
                height = 320
                width = 352
                image = np.zeros((height, width, 3), dtype=np.uint8)

            
                channel_1 = label1[0,0,:,:] 
                channel_2 = label1[0,1,:,:] 
                channel_3 = label1[0,2,:,:] 
      
                for k in range(320):
                  for l in range(352):
                      pixel_va1 = channel_1[k,l]
                      pixel_va12 = channel_2[k,l]
                      pixel_va13 = channel_3[k,l]
           
                      if pixel_va1 == 1:#channel 1
                            image[k,l] = [0,0,0]
                      elif pixel_va12 == 1 :
                           image[k,l] = [128,0,0]
                      elif pixel_va13 == 1 :
                           image[k,l] = [128,128,0]
                      
                return Image.fromarray(image.astype('uint8'))

      target = label_trans(label1)
      label_mask = multi_class_to_image(temp)

      label_mask.save('C:/Users/DaHaeLee/medical/resnet18_train_epoch/'+str(cnt2)+'epoch_train_img_label_.png','PNG')
      if cnt2 == 1:
            target.save('C:/Users/DaHaeLee/medical/resnet18_train_epoch/'+str(cnt2)+'epoch_train_target_label_.png','PNG')
      train_loss /= len(train_loader)
 
#--------------------------------------------------------------------------------------------------------------Validation----------------------------------------------------------------------------------------------------------------------
      cnt = 0
      with torch.no_grad():
            #validation
            model.eval() #검증
            cnt += 1
            for images, labels in val_loader:
                  images, labels= images.to(device), labels.to(device)
                  labels = labels.type(torch.FloatTensor).to(device)
                  logits = model(images)
                  # m= nn.Sigmoid()
                  softmax = nn.Softmax(dim=1)
                  softmax_a = softmax(logits)
                  output = softmax_a
                  loss = criterion(output, labels)
                  valid_loss += loss.item()  
      valid_loss /= len(val_loader)
      loss1.append(train_loss)
      loss2.append(valid_loss)
      loss1_x.append(e)
      loss2_x.append(e)
      print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(e, train_loss, valid_loss))
      torch.save(model.state_dict(),'2D_segmentation_normalize_epoch.pt')

# --------------------------------------------------------------------------------------------------------------Test----------------------------------------------------------------------------------------------
model.load_state_dict(torch.load("2D_segmentation_normalize_epoch.pt"))
pred_dict = {'input':[], 'target':[], 'output':[]}
cnt2 = 0
model.to('cpu')
with torch.no_grad():
        #validation
      model.eval() #검증
      cnt3 = 0
      for images, labels in test_loader:
            cnt2 += 1
            print(1)
            images = images.to('cpu')#images [1,3,256,256]
            labels = labels.type(torch.IntTensor).to('cpu')
            output = model(images).detach().cpu()#([1, 2, 256, 256]) 
            print(output)
            # print(output)
            # m = nn.Sigmoid()
            # output = m(output)
            softmax = nn.Softmax(dim=1)
            softmax_a = softmax(output)
            output = softmax_a
            output = torch.argmax(output, dim=1, keepdim=True)
            print(output)

 #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------           
         
            data1 = images.permute(0,2,3,1)#[256,256,3] ---->batch중 한 개 ---- permute() 통해 바꿔주고
            data_ = np.array(data1, dtype=np.uint8)

            output_ = output
            
            def multi_class_to_image(output_f):#label one-hot coding을 이미지로 변환
                  height = 320
                  width = 352
                  image1 = np.zeros((height, width, 3), dtype=np.uint8)

                  for k1 in range(320):
                        for l1 in range(352):

                              if output_f[0,0,k1,l1] == 0:
                                   image1[k1,l1,0] = 0
                                   image1[k1,l1,1] = 0
                                   image1[k1,l1,2] = 0

                              elif output_f[0,0,k1,l1] == 1:
                                   image1[k1,l1,0] = 128
                                   image1[k1,l1,1] = 0
                                   image1[k1,l1,2] = 0
                              
                              elif output_f[0,0,k1,l1] == 2:
                                   image1[k1,l1,0] = 128
                                   image1[k1,l1,1] = 128
                                   image1[k1,l1,2] = 0
                              
                  print(image1)
                  print(image1.shape)
                  return Image.fromarray(image1.astype('uint8'))
            
            
     
            def label_trans(label1):#label one-hot coding을 이미지로 변환
                height = 320
                width = 352
                image = np.zeros((height, width, 3), dtype=np.uint8)

                channel_1 = label1[0,0,:,:] 
                channel_2 = label1[0,1,:,:] 
                channel_3 = label1[0,2,:,:] 
      
                for k in range(320):
                  for l in range(352):
                      pixel_va11 = channel_1[k,l]
                      pixel_va12 = channel_2[k,l]
                      pixel_va13 = channel_3[k,l]
           
                      if pixel_va11 == 1:#channel 1
                             image[k,l] = [0,0,0]
                      elif pixel_va12 == 1  :
                           image[k,l] = [128,0,0]
                      elif pixel_va13 == 1:
                           image[k,l] = [128,128,0]
                      
                
                return Image.fromarray(image.astype('uint8'))

            
            label_mask = multi_class_to_image(output)

            # image_ex =  data_.copy()
            # img = cv2.merge([image_ex,image_ex,data_])
            # img2 = np.array(label_mask)
            # print(img)
            # print(img.shape)
            # img = img.squeeze(0)
            # img = img.squeeze(2)
            # label_mask = cv2.add(img2, img , dst=None, mask=None, dtype=None) 
            # label_mask = Image.fromarray(label_mask.astype('uint8'))

            cnt3 += 1
            label_mask.save('C:/Users/DaHaeLee/medical/resnet18_test/'+str(cnt3)+'epoch_test1_img_label.png','PNG')

#             pred_dict['input'].append(data_)
#             pred_dict['target'].append(target)
#             pred_dict['output'].append(label_mask)

# print(pred_dict['output'])
# print(pred_dict['target'])
#-----------------------------------------------------------------------------------------------------Visualization------------------------------------------------------------------------------------------------------------------------- label_mask.save('img_{} epoch.png'.format(cnt),'PNG')
# pred_dict['input'][0].save('img_input epoch.png','PNG')
# pred_dict['output'][0].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test1_img_label.png','PNG')
# pred_dict['target'][0].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test1_img_target.png','PNG')

# pred_dict['output'][1].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test2_img_label.png','PNG')
# pred_dict['target'][1].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test2_img_target.png','PNG')

# pred_dict['output'][2].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test3_img_label.png','PNG')
# pred_dict['target'][2].save('C:/Users/DaHaeLee/medical/test_epoch_img/epoch_test3_img_target.png','PNG')


x = loss1_x
y = loss1
plt.plot(x, y, marker="o")
plt.title("Training Loss")
plt.xlabel("epoch")
plt.savefig('Training_Loss.png')
plt.show()

x1 = loss2_x
y1 = loss2
plt.plot(x1,y1, marker="o")
plt.title("Validation Loss")
plt.xlabel("epoch")
plt.savefig('Validation_Loss.png')
plt.show()
