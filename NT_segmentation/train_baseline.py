import torch
import argparse
import os
import shutil
from importlib import import_module

from utils.train_method import train
from utils.set_seed import setSeed

from dataset.base_dataset import SegDataset  
def getArgument():
	# Custom 폴더 내 훈련 설정 목록을 선택
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str ,required=True)
	return parser.parse_known_args()[0].dir

def main(custom_dir):

	arg = getattr(import_module(f"custom.{custom_dir}.settings.arg"), "getArg")()#arg.py -> batch,epoch,경로 등 S

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(device)
	
	setSeed(arg.seed)

	train_x_transform,train_y_transform= getattr(import_module(f"custom.{custom_dir}.settings.transform"), "getTransform")()
	train_dataset = SegDataset(img_path= arg.train_img_path,img_labels=arg.train_img_label,mode='train',train_x_transform=train_x_transform,train_y_transform=train_y_transform)
	# train_dataset = SegDataset(data_dir=arg.image_root,image_root=arg.image_root, mode='train', transform=train_transform)
	val_dataset = SegDataset(img_path= arg.train_img_path,img_labels=arg.train_img_label, mode='val', train_x_transform=train_x_transform)

	trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.settings.dataloader"), "getDataloader")(
		train_dataset, val_dataset, arg.batch)

	model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getModel")()
	criterion = getattr(import_module(f"custom.{custom_dir}.settings.loss"), "getLoss")()

	optimizer, scheduler = getattr(import_module(f"custom.{custom_dir}.settings.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)

	outputPath = os.path.join(arg.output_path, arg.custom_name)

	#output Path 내 설정 저장
	# shutil.copytree(f"custom/{custom_dir}",outputPath)
	# os.makedirs(outputPath+"/models")

	train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device)


if __name__=="__main__":
	custom_dir = getArgument()
	main(custom_dir)

	




