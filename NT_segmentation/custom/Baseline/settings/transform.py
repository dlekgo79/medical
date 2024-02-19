import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta
import torchvision.transforms as transforms
# https://albumentations-demo.herokuapp.com/

def getTransform():

    train_x_transform= transforms.Compose([                                                                                                                                                              
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.236356),(0.1891466)),
     ])    

    train_y_transform= transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.33333343 ,0.33333343, 0.33333343),(0.47129032, 0.47128868, 0.47129023)),
     ])    

    val_x_transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.236356),(0.1891466))
#     transforms.Normalize((60.62533 ),(47.828457)),
     ])    

    val_y_transforms = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        transforms.Normalize((0.33333343 ,0.33333343, 0.33333343),(0.47129032, 0.47128868, 0.47129023)),
     ])    
    

    return train_x_transform ,  train_y_transform


def getInferenceTransform():
    
    test_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.236356),(0.1891466))
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.33333343 ,0.33333343, 0.33333343),(0.47129032, 0.47128868, 0.47129023)),
     ])    
    
    return    test_transform



#   train_transform = A.Compose([
#                               A.Flip(),
#                               # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                               ToTensorV2(),
#                               ])

#   val_transform = A.Compose([
#                               # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                             ToTensorV2(),
#                             ])

#   return train_transform, val_transform


# def getInferenceTransform():

#   test_transform = A.Compose([
#                             # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                            ToTensorV2(),
#                            ])

#   tta_transform = tta.Compose(
#     [
#         tta.HorizontalFlip(),
#         tta.Rotate90(angles=[0, 180]),
#         tta.Scale(scales=[1, 2, 4]),
#         tta.Multiply(factors=[0.9, 1, 1.1]),        
#     ]
# )

                        
  # return test_transform, tta_transform