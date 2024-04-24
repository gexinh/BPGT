from __future__ import print_function, division
from cProfile import label
import os
from pyexpat import features
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from scipy import stats

from torch.utils.data import Dataset
import h5py

class Generic_ML_Table(Dataset):
    """ customed pytorch dataset for multi-label data 
    """
    def __init__(self, csv_path, dataset, seed=1, shuffle=True, region_list=None):
        slide_data = pd.read_csv(csv_path) # ./Step_3.csv
        
        # slide_data = self.filter_df(slide_data, filter_dict) # filter
        # slide_data = self.df_prep(slide_data, self.label_col)   # concate
        
        #TODO: pick up region 
        # .....

        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data  # the data structure is dataframe
        # self.gene_list = self.slide_data.columns.values
        self.data_dir = None  # // pretrain_feature/

        self.data = dataset
        # self.gene_list = slide_data.columns.values

        self.region_list = region_list
        # for extracting features
        # self.use_h5 = use_h5
        # self.data_dir = data_dir
    
    #TODO: check the lenth of dataframe
    def __len__(self):
        return len(self.slide_data)

    def __getitem__(self, idx):
        return None

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        split = split.drop_duplicates(keep='first')
        
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
        #TODO: modify this part!
            split = Generic_Split(df_slice, dataset=self.data, data_dir=self.data_dir, region_list =self.region_list)
        else:
            split = None

        return split

    def return_splits(self, split_bool_path=None):
        """ split path
        """
        assert split_bool_path 
        all_splits = pd.read_csv(split_bool_path, dtype=self.slide_data['slide_id'].dtype)  # Without "dtype=self.slide_data['slide_id'].dtype", read_csv() will convert all-number columns to a numerical type. Even if we convert numerical columns back to objects later, we may lose zero-padding in the process; the columns must be correctly read in from the get-go. When we compare the individual train/val/test columns to self.slide_data['slide_id'] in the get_split_from_df() method, we cannot compare objects (strings) to numbers or even to incorrectly zero-padded objects/strings. An example of this breaking is shown in https://github.com/andrew-weisman/clam_analysis/tree/main/datatype_comparison_bug-2021-12-01.
        train_split = self.get_split_from_df(all_splits, 'train')
        val_split = self.get_split_from_df(all_splits, 'val')
        test_split = self.get_split_from_df(all_splits, 'test')	    		
        return train_split, val_split, test_split


class Generic_ML_Dataset(Generic_ML_Table):

    def __init__(self,
        data_dir, region_list=None,
        **kwargs):
        super(Generic_ML_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = False
        self.region_list = region_list
        #TODO: chech if it is correct
        

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx] 
        region = self.slide_data['region'][idx]
        label = self.slide_data[list(self.gene_list)].iloc[idx,:] # (sample, labels)
        # label = torch.tensor(label.values)
        label = torch.tensor(label.values.astype(np.float32))
        # label = self.slide_data['label'][idx]
        
        #TODO: following codes have been commented
        # if type(self.data_dir) == dict:
        #     source = self.slide_data['source'][idx]
        #     data_dir = self.data_dir[source]
        # else:
        
        #TODO: multi-region case
        if self.region_list is not None:
            region_name = self.region_list[int(region)][5:] if self.region_list[int(region)][0:4] == 'TCGA' else self.region_list[int(region)]
            cur_data_dir = re.split('/',self.data_dir) #./data/tcga/{region}/mag{}/{}feature --> ['.', 'data', 'tcga', '{}', 'mag{}', feature]
            cur_data_dir[-3] = region_name
            cur_data_dir = '/'.join(cur_data_dir)
        else:
            cur_data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                # TODO: modify this part
                full_path = os.path.join(cur_data_dir, 'pt_files', '{}.pt'.format(slide_id))
                features = torch.load(full_path)
                return features, label
            else:
                return slide_id, label

        else:
            full_path = os.path.join(cur_data_dir,'h5_files','{}.h5'.format(slide_id))
            with h5py.File(full_path,'r') as hdf5_file:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]

            features = torch.from_numpy(features)
            return features, label, coords


class Generic_Split(Generic_ML_Dataset):
    def __init__(self, slide_data, dataset, data_dir=None, region_list=None):

        self.use_h5 = False
        self.slide_data = slide_data
        self.region_list =region_list

        if dataset == 'tcga':
            self.start_point = 1
        elif dataset == 'cptac':
            self.start_point = 2
        self.gene_list = self.slide_data.columns.to_list()[self.start_point:]  # region, gene1, gene2, ...
        self.data_dir = data_dir

    def __len__(self):
        return len(self.slide_data)

    def get_labels(self):
        labels = self.slide_data.iloc[:,self.start_point:]
        return labels

#TODO: 4/10
# class Generic_WSI_Patch(Dataset):
#     def __init__(self,
#         csv_path,
#         slide_id_list,
#         file_path,
#         pretrained=False,
#         custom_transforms=None,
#         target_patch_size=-1,
#         ):
#         """ This script modified from Whole_Slide_Bag, utlizing to pre train model from specific train set

#         Args:
#             slide_id_path (list): Path to the selected .h5 file
#             file_path (string): Path to the .h5 file containing patched data.
#             pretrained (bool): Use ImageNet transforms
#             custom_transforms (callable, optional): Optional transform to be applied on a sample
#         """
#         self.slide_data = pd.read_csv(csv_path) # ./Step_3.csv
#         self.gene_list = self.slide_data.columns.to_list()[2:] 
#         self.slide_id_list =slide_id_list

#         self.pretrained=pretrained
#         if target_patch_size > 0:
#             self.target_patch_size = (target_patch_size, target_patch_size)
#         else:
#             self.target_patch_size = None

#         if not custom_transforms:
#             self.roi_transforms = eval_transforms(pretrained=pretrained)
#         else:
#             self.roi_transforms = custom_transforms

#         self.file_path = file_path

#         with h5py.File(self.file_path, "r") as f:
#             dset = f['imgs']
#             self.length = len(dset)

#         self.summary()
            
#     def __len__(self):
#         return self.length

#     def summary(self):
#         hdf5_file = h5py.File(self.file_path, "r")
#         dset = hdf5_file['imgs']
#         for name, value in dset.attrs.items():
#             print(name, value)

#         print('pretrained:', self.pretrained)
#         print('transformations:', self.roi_transforms)
#         if self.target_patch_size is not None:
#             print('target_size: ', self.target_patch_size)

#     def __getitem__(self, idx_):
#         df = self.slide_data
#         idx = df[df['slide_id'] == self.slide_id_list[idx_]].index.tolist()
#         slide_id = df['slide_id'][idx] 


#         label = self.slide_data[list(self.gene_list)].iloc[idx,:] # (sample, labels)
#         # label = torch.tensor(label.values)
#         label = torch.tensor(label.values.astype(np.float32))
        
#         full_path = os.path.join(self.file_path,'patches','{}.h5'.format(slide_id)) 
#         with h5py.File(full_path,'r') as hdf5_file:
#             img = hdf5_file['imgs'][idx]
#             coord = hdf5_file['coords'][idx]
        
#         img = Image.fromarray(img)
#         if self.target_patch_size is not None:
#             img = img.resize(self.target_patch_size)
#         img = self.roi_transforms(img).unsqueeze(0)

#         return img, coord









