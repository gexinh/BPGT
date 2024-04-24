""" this script is used to extract feature via resnet, which is trained by ImageNet
	Update:
		1. add new model: ResNet50, ResNet101 and Xception
		2. can use clustered patch
		3. can extract patch for different magnification, patch size, and dataset 
	Illustration:
		h5_files:
		pt_files:
"""
import torch
import torch.nn as nn
from math import floor
import os


import pandas as pd
import argparse

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

def get_args_parser():
	parser = argparse.ArgumentParser('ConvNet', add_help=False)
	#path
	parser.add_argument('--output_dir', default="./checkpoints", type=str, help='Path to save logs and checkpoints.')
	parser.add_argument('--data_h5_dir', type=str, default='./data/cptac')
	parser.add_argument('--csv_path', type=str, default='./data/cptac/Step_2.csv')
	parser.add_argument('--feat_dir', type=str, default='./data/cptac/pretrained_feature')

	parser.add_argument('--if_ddp', type=int, default=False, help='if use the DDP')
	parser.add_argument('--select_gpu', type=int,default=0, help='if use select gpu')

	parser.add_argument('--task', type=str, choices=['cptac',  'tpca'], default='cptac')
	parser.add_argument('--label_frac', type=float, default=1.0,
						help='fraction of training labels (default: 1.0)')
	parser.add_argument('--slide_ext', type=str, default= '.svs')

	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--no_auto_skip', default=False, action='store_true')
	parser.add_argument('--target_patch_size', type=int, default=-1)

	parser.add_argument('--custom_downsample', type=int, default=1)

	parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac'], help='choose the dataset')

	parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
					help='choose the magnification of slides')
	parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
	parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')
	parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
	parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')

	parser.add_argument('--ori_patch_size', type=int, default=256, help='the original patch size of WSI')

	parser.add_argument('--pretrained_model', choices = ['dino', 'resnet', 'incep','hipt'], default='dino', help='choose pretrained model')
	return parser
parser = argparse.ArgumentParser('ConvNet', parents=[get_args_parser()], description='Feature Extraction')
args = parser.parse_args()

if not args.if_ddp:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.select_gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

    os.environ["OMP_NUM_THREADS"] = "6"  #
    os.environ["MKL_NUM_THREADS"] = "6"  # 
    import torch
    torch.set_num_threads(6)

import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from model.resnet_custom import ResNet_extractor, resnet50_baseline

from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# if __name__ == '__main__':

# select dataset
data = args.data
label_source = args.s
label_tag = args.atr

from utils.file_utils import param_selection_for_pretrain, custom_downsample_per_slide
args = param_selection_for_pretrain(data, label_source, label_tag, args)


print('initializing dataset')
csv_path = args.csv_path
if csv_path is None:
	raise NotImplementedError

# TODO: load dataset and add clustered-index 
bags_dataset = Dataset_All_Bags(csv_path)

os.makedirs(args.feat_dir, exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

print('loading model checkpoint')
# TODO: modify for different model
if args.pretrained_model == 'resnet':
	model = resnet50_baseline(pretrained=True)
	
elif args.pretrained_model == 'incep':
	# ref: https://rwightman.github.io/pytorch-image-models/models/inception-v4/
	from model.resnet_custom import Inception_extractor
	model = Inception_extractor(name='inception_v4', retrain=True, ckp_path=None)

elif args.pretrained_model == 'resnet101':
	model = ResNet_extractor(layers=101)
	
model = model.to(device)
# load weight

# print_network(model)
if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)
	
model.eval()
total = len(bags_dataset)

for bag_candidate_idx in range(total):
	""" construct dataset
	"""
	# TODO: modify this part
	slide_id = bags_dataset[bag_candidate_idx]
	h5_file_path = os.path.join(args.data_h5_dir,'patches','{}.h5'.format(slide_id))

	wsi_path = args.dir_path_df.loc[args.dir_path_df['slide_id']==slide_id,'full_path'].values.tolist()
	
	wsi_path = os.path.join(args.abs_path,  wsi_path[0].split('/',3)[-1])

	# wsi = openslide.open_slide(wsi_path[0])
	bag_name = slide_id + '.h5'

	# slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
	# bag_name = slide_id+'.h5'
	# h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name) # path of patch saving 

	# dir_path = bags_dataset.df.loc[bag_candidate_idx,'full_path'].rsplit('/',1)[0] # only split the last '/' and fetch first str 
	# slide_file_path = os.path.join(dir_path,slide_id+args.slide_ext) 
	# slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)

	print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
	print(slide_id)

	if not args.no_auto_skip and slide_id+'.pt' in dest_files:
		print('skipped {}'.format(slide_id))
		continue 
	output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

	#
	slide_mag = args.dir_path_df.loc[args.dir_path_df['slide_id']==slide_id,'mag']
	custom_downsample = custom_downsample_per_slide(slide_mag, args.mag)
	""" process dataset
	"""
	time_start = time.time()
	wsi = openslide.open_slide(wsi_path) # slide 
	output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
	model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
	custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
	
	time_elapsed = time.time() - time_start
	print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
	file = h5py.File(output_file_path, "r")

	features = file['features'][:]
	print('features size: ', features.shape)
	print('coordinates size: ', file['coords'].shape)   # [Patch, 2]
	features = torch.from_numpy(features)
	bag_base, _ = os.path.splitext(bag_name)
	torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))