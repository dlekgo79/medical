import torch
import numpy as np
from torch import nn
# from utils.utils import add_hist, label_accuracy_score
# from utils.wandb_method import WandBMethod
# from utils.tqdm import TQDM
# from utils.save_helper import SaveHelper
from torch.cuda.amp import GradScaler, autocast
from utils.utils import add_hist, label_accuracy_score

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device,):
    n_class = 3
    
    train_loss = 0
    valid_loss = 0
    train_acc = 0
    valid_acc = 0

    for epoch in range(0,num_epochs):
        model.train()
        n = len(train_loader)
    
        for images,label in train_loader:
            images, label = images.to(device), label.to(device)
            model = model.to(device)
            optimizer.zero_grad()

            logits = model(images)
            softmax = nn.Softmax(dim=1)
            softmax_a = softmax(logits)
            output = softmax_a
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # hist = add_hist(hist, label, output, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

        scheduler.step() 
        train_loss /= len(train_loader)
        return train_loss

def validation(epoch, model, valid_loader, criterion, device,):
    model.eval()
    with torch.no_grad():
        n_class = 3
        cnt = 0
        

        for images,labels in valid_loader:
      
            images, labels = images.to(device), labels.to(device)            
            model = model.to(device)
            outputs = model(images)
            softmax = nn.Softmax(dim=1)
            softmax_a = softmax(outputs)
            output = softmax_a
            loss = criterion(outputs,labels)
            valid_loss += loss.item()  
    
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            cnt += 1
    valid_loss /= len(valid_loader)
    return valid_loss

