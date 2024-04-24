""" Use to form the information table(csv) of dataset for extract_patch.py 

Usage: 
    Step 0: 
        python create_patches_fp.py \
        --source /data2/WSI_CPTAC/GBM/GBM \
        --save_dir ./toy_test \ 
        --patch_size 256 \
        --seg \
        --patch \
        --stitch 

    Step 1: run this code - python create_step_2_csv.py --csv_path --save path

    Step 2: use following command line, e.g.:
        CUDA_VISIBLE_DEVICES=5 python extract_features_fp.py \
        --data_h5_dir ./toy_test \
        --data_slide_dir /data2/WSI_CPTAC/GBM/GBM \
        --csv_path ./Step_2.csv \
        --feat_dir ./FEATURES_DIRECTORY \
        --batch_size 512 \
        --slide_ext .svs

        nohup CUDA_VISIBLE_DEVICES=5 python extract_features_fp.py &

    @ old version
    Step 3: use this code to generate the dataframe of training, validation, and test dataset, e.g.:
        python create_splits_seq.py \
        --task task_1_tumor_vs_normal \
        --seed 1 \
        --label_frac 0.75 \
        --k 10

    @ new version
    Step 3: use [1]_create_step_3_csv.py for further preprocessing

@Author: Gexin
"""

import os
from sqlite3 import register_converter 
import pandas as pd 
import argparse

parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')

parser.add_argument('--csv_path', type = str,
					help='path to folder containing csv of slided images'
					,default='./data/cptac/'
					)
parser.add_argument('--save_path', type = str,
					help='path to folder saving csv for extract_features_fp.py'
					,default='./data/cptac/'
					)
parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')

# TODO: additional 
parser.add_argument('--patch_size', default=256, type = int, help='choose the patch size of slides')
parser.add_argument('--mag', default=-1, type = int, choices=[-1, 40, 20, 10, 5], help='choose the magnification of slides')

# if __name__ == '__main__':
args = parser.parse_args()

if args.data == 'cptac':
    # Step 2: CUDA_VISIBLE_DEVICES=5 nohup python extract_features_fp.py > extract_feature.log 2>&1 &
    csv_path = './data/cptac/'
    save_path ='./data/cptac/'
    region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']

    args.region_list = region_list if not args.specific_region else [region_list[args.r]]	 
    save_path = './data/cptac' if not args.specific_region else './data/cptac/' + region_list[args.r]
    csv_path = save_path
    # csv_path = args.csv_path
    # save_path = args.save_path


elif args.data == 'tcga':
    # Step 2: CUDA_VISIBLE_DEVICES=5 nohup python extract_features_fp.py > extract_feature.log 2>&1 &
    csv_path = './data/tcga/'
    save_path = './data/tcga/'
    region_list = ['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',
        'LGG', 'PRAD', 'THCA', 'SKCM', 'STAD', 'BLCA', 
        'LIHC', 'CESC', 'KIRP', 'SARC', 'ESCA', 'PCPG',
        'READ', 'LAML', 'TGCT', 'THYM', 'ACC', 'MESO', 
        'UVM', 'KICH', 'UCS', 'DLBC', 'CHOL']
        
    args.region_list = region_list if not args.specific_region else [region_list[args.r]]	 
    save_path = './data/tcga' if not args.specific_region else './data/tcga/' + region_list[args.r]
    csv_path = save_path

if args.mag != -1:
    if args.mag == 20:
        args.custom_downsample = 2
    elif args.mag == 10:
        args.custom_downsample = 4
    elif args.mag ==5:
        args.custom_downsample = 8
    elif args.mag ==40:
        args.custom_downsample = 1
	# ratio = 40/ args.mag since the max/default magnification is 40 
    args.patch_size = args.patch_size * args.custom_downsample

    csv_path = csv_path + '/mag{}x_patch{}_fp'.format(args.mag, int(args.patch_size/(40/args.mag)))
else:
    csv_path= csv_path+ '/mag{}x_patch{}_fp'.format(40, int(args.patch_size))

df = pd.read_csv(csv_path+'/process_list_autogen.csv') 

# clean useless data, like those without magnification information
ids1 = [i[:-4] for i in df.slide_id] #.svs
ids2 = [i[:-3] for i in os.listdir(csv_path+'/patches/')] #.h5
df['slide_id'] = ids1
ids = df['slide_id'].isin(ids2)

df = df.loc[ids]
print('the number of sample is ', str(sum(ids)))
print('-----------------------')

if args.data == 'cptac':
    region_ind = [x.split('/')[-2] for x in df['full_path']]
elif args.data == 'tcga':
    region_ind = region_list[args.r]

df.insert(loc=0,column='region',value= region_ind)


# df.to_csv(save_path + '/Step_2.csv',index=False)

df.to_csv(csv_path + '/Step_2.csv',index=False)

""" Step 3 has been adding in [1]_create_step_3_csv.py
"""
## Step 3: nohup python create_splits_seq.py > split.log 2>&1
# df = pd.read_csv('./cohort.csv')
# df = df[['Case_ID','Slide_ID','Specimen_Type']]

# ids1 = [i for i in df.Slide_ID]
# ids2 = [i[:-3] for i in os.listdir(csv_path+'/patches/')]  #[:-3] 指的是去除 .h5 后缀

# ids = df['Slide_ID'].isin(ids2)  # choose mutual data

# print('the number of sample is ', str(sum(ids)))
# print('-----------------------')

# df = df.loc[ids]
# df.columns = ['case_id','slide_id','label']
# df.to_csv(save_path + '/Step_3.csv',index=False)
