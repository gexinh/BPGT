""" create splits based on Step_3.csv
"""
from __future__ import print_function, division
import os
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
import argparse
from skmultilearn.model_selection import iterative_train_test_split

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
   splits = [split_datasets[i] for i in range(len(split_datasets))]
   if not boolean_style:
      df = pd.concat(splits, ignore_index=True, axis=1)
      df.columns = column_keys
   else:
      df = pd.concat(splits, ignore_index = True, axis=0)
      index = df.values.tolist()
      one_hot = np.eye(len(split_datasets)).astype(bool)
      bool_array = np.repeat(one_hot, [len(splits[dset].values.tolist()) for dset in range(len(splits))], axis=0)
      df = pd.DataFrame(bool_array, index=np.squeeze(index), columns = ['train', 'val', 'test'])

   df.to_csv(filename)
   print()


parser = argparse.ArgumentParser(description='Creating splits for WSI gene prediction')
parser.add_argument('--label_frac', type=float, default= 1,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['cptac', 'tgca'], default='cptac')
parser.add_argument('--label_space', type=str, choices=['react','oncokb','final'],default='react',help='choose label space')

# add magnification, data, region, and 
parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')
parser.add_argument('--patch_size', default=256, type = int, help='choose the patch size of slides')

parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
            help='choose the magnification of slides')

parser.add_argument('--r', type = int, default=0,
         help='the index of specific region for tcga/cptac dataset')
parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')


parser.add_argument('--specific_region', type = int, default=True,
         help='if process the specific region for tcga/cptac dataset')
parser.add_argument('--mix_list', type=list, default=None, nargs='+', help='1234 -> [1,2,3,4]')
args = parser.parse_args()


def ml_stratification(data, region=None):
   """ stratification for the multi-label label space
   """
   # one-fold
   if region is not None:
      data = data[data['region'] == region]
      x = data[["slide_id"]].values
   else:
      x = data[["slide_id"]].values

   gene_list = list(data.columns[args.start_ind:].values)
   y = data[gene_list].values

   train_X,train_y,rest_x,rest_y  = iterative_train_test_split(x, y, test_size = 0.2)
   val_X,val_y,test_X,test_y  = iterative_train_test_split(rest_x, rest_y,test_size = 0.5)

   print('The number of one-patch training set, validation set, and test set is {}, {}, and {} respectively'.format(len(train_X),len(val_X),len(test_X)))

   train_X = np.squeeze(train_X)
   val_X = np.squeeze(val_X)
   test_X = np.squeeze(test_X)
   return train_X, val_X, test_X

for lf in label_fracs:
   # TODO: data path
   split_dir = os.path.join(args.split_dir, str(args.data) + '_{}'.format(int(lf * 100)))
   os.makedirs(split_dir, exist_ok=True)

   for i in range(args.k):

      if args.mix_list is not None:
         train_X = []
         val_X = []
         test_X = []
         for k in args.mix_list:
            region = 'TCGA-'+args.region_list[int(k)] if args.data=='tcga' else args.region_list(int(k))
            cur_train, cur_val, cur_test = ml_stratification(data, region)
            train_X.extend(cur_train)
            val_X.extend(cur_val)
            test_X.extend(cur_test)
      else:
        train_X, val_X, test_X = ml_stratification(data)
            
      print('The number of training set, validation set, and test set is {}, {}, and {} respectively'.format(len(train_X),len(val_X),len(test_X)))
      
      splits = [pd.DataFrame(train_X, columns=['slide_id']), pd.DataFrame(val_X, columns=['slide_id']), pd.DataFrame(test_X, columns=['slide_id'])]


      # save split data
      save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
      save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)

