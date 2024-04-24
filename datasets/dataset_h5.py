from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		f = h5py.File(self.file_path,'r')
		
		for key in f.keys():
			print(f[key].name)
			print(f[key].shape)
			print(f[key].value)
			
		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
			self.custom_transform_flag = False
		else:
			self.roi_transforms = custom_transforms
			self.custom_transform_flag = True

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('feature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)
		print('+++++++++++++++++++++++++++++++++++\n')

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		
		if self.custom_transform_flag == True:
		# img = self.roi_transforms(img).unsqueeze(0)  # img is a tensor, which shape is ([1,3,256,256])
			img = self.roi_transforms(img)
		else:
			img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]



""" This script modified from Whole_Slide_Bag, utlizing to pre train model from specific train set

Args:
	slide_id_path (list): Path to the selected .h5 file
	file_path (string): Path to the .h5 file containing patched data.
	pretrained (bool): Use ImageNet transforms
	custom_transforms (callable, optional): Optional transform to be applied on a sample
"""


		

class Generic_WSI_Patch(Dataset):
	def __init__(self,csv_path,
        slide_id_list,
        file_path,
        pretrained=False,
        custom_transforms=None,
        target_patch_size=-1,):

		self.slide_data = pd.read_csv(csv_path) # ./Step_3.csv
		self.gene_list = self.slide_data.columns.to_list()[2:] 
		self.slide_id_list =slide_id_list    			
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path
		self.length=len(slide_id_list)
		# with h5py.File(self.file_path, "r") as f:
		#     dset = f['imgs']
		#     self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx_):
		df = self.slide_data
		idx = df[df['slide_id'] == self.slide_id_list[idx_]].index.tolist()
		slide_id_set = df['slide_id'][idx] 


		label = self.slide_data[list(self.gene_list)].iloc[idx,:] # (sample, labels)
		# label = torch.tensor(label.values)
		label = torch.tensor(label.values.astype(np.float32))

		for	slide_id in slide_id_set:
			full_path = os.path.join(self.file_path,'patches','{}.h5'.format(slide_id)) 
			with h5py.File(full_path,'r') as hdf5_file:
				img = hdf5_file['imgs'][idx]
				coord = hdf5_file['coords'][idx]
				

		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		
		# TODO: what's the shape??
		img = self.roi_transforms(img).unsqueeze(0)

		return img, label, coord
