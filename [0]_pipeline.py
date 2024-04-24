""" script for producing different magnification of dataset (according to a specific region)
    
    Args:
        mag: the magnification of slide: 5x, 10x, 20x, 40x
        data: tcga or cptac
        r: cancer region in terms of dataset, e.g., Breast, Lung, etc.
"""
import subprocess
import argparse
import os
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
from utils.general_utils import mkdir

parser = argparse.ArgumentParser(description='seg and patch')

parser.add_argument('--ori_patch_size', type = int, default=256,
					help='the patch size of WSIs')
parser.add_argument('--mag', default=-1, type = int, choices=[-1, 40, 20, 10, 5], help='choose the magnification of slides')
parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')
parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')
parser.add_argument('--mix_list', type=list, default=None, nargs='+', help='1234 -> [1,2,3,4]')

parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')

parser.add_argument('--model', default='dino', choices=['resnet', 'dino', 'hipt'], help='choose the source of tcga dataset label')
parser.add_argument('--inter', type = int, default=False,
					help='if interatively process the specific region for tcga/cptac dataset')
parser.add_argument('--gpu', type=int,default=0, 
                        help='specify the gpu for the model')
args = parser.parse_args()

#
"""
cptca - tcga
'BR' - 'BRCA',  
'CCRCC' - 'KIRC', 
'CO' - 'COAD',  
'GBM' - 'GBM',  
'HNSCC' - 'HNSC',  
'LSCC' - 'LUSC',  
'LUAD' - 'LUAD',  
'OV' - 'OV',  
'PDA' - 'PAAD',  
'UCEC' - 'UCEC'  
"""
selected_region_list ={'tcga':['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',],
                        'cptac':['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']}

# default setting
def default_setting(args):
    if args.data == 'cptac':
        region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']
        args.region_list = region_list if not args.specific_region else [region_list[args.r]]	 
        args.save_dir = './data/cptac' if not args.specific_region else './data/cptac/' + region_list[args.r]

    elif args.data == 'tcga':
        region_list = ['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',
            'LGG', 'PRAD', 'THCA', 'SKCM', 'STAD', 'BLCA', 
            'LIHC', 'CESC', 'KIRP', 'SARC', 'ESCA', 'PCPG',
            'READ', 'LAML', 'TGCT', 'THYM', 'ACC', 'MESO', 
            'UVM', 'KICH', 'UCS', 'DLBC', 'CHOL']
        args.region_list = region_list if not args.specific_region else [region_list[args.r]]	 
        args.save_dir = './data/tcga' if not args.specific_region else './data/tcga/' + region_list[args.r]

    else:
        args.region_list == None

    # dataset
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

        args.patch_size = args.ori_patch_size * args.custom_downsample
        args.step_size = args.patch_size
        args.actual_patch_size = int(args.patch_size/(40/args.mag))
        args.save_dir = args.save_dir + '/mag{}x_patch{}_fp'.format(args.mag, args.actual_patch_size)


    else:
        args.save_dir = args.save_dir + '/mag{}x_patch{}_fp'.format(40, int(args.patch_size))
        args.custom_downsample = 1
        args.actual_patch_size = args.patch_size 

    mkdir(args.save_dir)
    return args

def pipiline(args, flag_list=[0,0,0,0,1,0,0]):
    pre_process, create_step2, create_step3, create_label_dist, create_split, create_graph, train = flag_list
    """
    # pre-procecss the dataset
    """
    if pre_process == True:
        if os.path.exists((args.save_dir+'/patch_slides.log')):
            print('Dataset of Region {} Has Been Processed'.format(args.region_list))            
        else:
            subprocess_param = ["python", "[0][0]_create_patches_fp.py",
                                "--data", args.data, 
                                "--patch_size", str(args.patch_size),"--step_size", str(args.step_size),
                                "--mag", str(args.mag),
                                "--patch","--seg","--stitch",
                                "--specific_region", str(args.specific_region),
                                "--r", str(args.r)
                                ]

            log_path = os.path.join(args.save_dir, 'patch_slides.log').replace('\\','/')
            with open(log_path,'w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)

    """
    # create step_2.csv
    """
    if create_step2 == True:
        if os.path.exists((args.save_dir+'/Step_2.csv')):
            print('Step 2 of Region {} Has Been Processed'.format(args.region_list))            
        else:
            subprocess_param = ["python", "[0][1]_create_step_2_csv.py",
                                "--data", args.data, 
                                "--patch_size", str(args.actual_patch_size),
                                "--mag", str(args.mag),
                                "--specific_region", str(args.specific_region),
                                "--r", str(args.r)
                                ]
            with open(args.save_dir+'/step_2.log','w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)                    

    """
    # create step_3.csv
    """
    if create_step3 == True:
        if os.path.exists((args.save_dir+'/step_3.log')):
            print('Step 3 of Region {} Has Been Processed'.format(args.region_list))            
        else:
            subprocess_param = ["python", "[1]_create_step_3_csv.py",
                                "--data", args.data, 
                                "--iter", "1",
                                "--r", str(args.r),
                                "--patch_size", str(args.actual_patch_size),
                                "--mag", str(args.mag),
                                ]
            with open(args.save_dir + '/step_3.log','w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)                    

    """
    """
    if create_label_dist == True:
        subprocess_param = ["python", "plot_distribution.py",
                            "--data", args.data, 
                            "--r", str(args.r),
                            "--patch_size", str(args.actual_patch_size),
                            "--mag", str(args.mag),
                            "--s",args.s,
                            "--atr",args.atr,
                            "--specific_region", "1",
                            ]
        with open(args.save_dir+'/plot.log','w') as f:
            completed = subprocess.run(subprocess_param, stdout=f,shell=False)                    

    """
    # create split
    """
    if create_split == True:
        if os.path.exists((args.save_dir+'/splits')):
            print('Splits of Region {} Has Been Processed'.format(args.region_list))            
        else:
            subprocess_param = ["python", "[2]_create_splits.py",
                                "--data", args.data, 
                                "--label_frac","1",
                                "--k", str(args.k),
                                "--r", str(args.r),
                                "--patch_size", str(args.actual_patch_size),
                                "--mag", str(args.mag),
                                "--s",args.s,
                                "--atr",args.atr,
                                ]
            with open(args.save_dir+'/split.log','w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)                    

    """ create graph
    """
    if create_graph == True:
        if os.path.exists((args.save_dir+'/graph.log')):
            print('Graphs of Region {} Has Been Processed'.format(args.region_list))            
        else:
            subprocess_param = ["python", "graph_construction.py",
                                "--data", args.data, 
                                "--s",args.s,
                                "--atr",args.atr,
                                "--specific_region", "1",
                                "--r", str(args.r),
                                "--patch_size", str(args.actual_patch_size),
                                "--mag", str(args.mag),
                                ]
                                
            with open(args.save_dir+'/graph.log','w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False) 

    """
    # extract feature
    """
    if train == True:
        if model == 'dino':
            log_path = os.path.join(args.save_dir, 'extract_feature_dino.log').replace('\\','/')
            if train_model == 1:
                subprocess_param = ["python", 
                                "-m", "torch.distributed.launch", 
                                "--nproc_per_node=4",
                                # "--master_port", "29502",
                                "[0][3]_extract_feature_dino.py",
                                "--custom_downsample", str(args.custom_downsample),
                                "--data", args.data,
                                "--mag", str(args.mag),
                                "--specific_region", str(args.specific_region),
                                "--r", str(args.r),
                                "--train_model", '1',
                                "--ori_patch_size", str(args.actual_patch_size),
                                "--s",args.s,
                                "--atr",args.atr,
                                "--if_ddp",'1',
                                ]
            elif train_model == 0:
                subprocess_param = ["python", 
                                # "-m", "torch.distributed.launch", 
                                # "--nproc_per_node=4",
                                # "--master_port", "29502",
                                "[0][3]_extract_feature_dino.py",
                                "--custom_downsample", str(args.custom_downsample),
                                "--data", args.data,
                                "--mag", str(args.mag),
                                "--specific_region", str(args.specific_region),
                                "--r", str(args.r),
                                "--train_model", '0',
                                "--ori_patch_size", str(args.actual_patch_size),
                                "--s",args.s,
                                "--atr",args.atr,
                                "--if_ddp",'0',
                                "--select_gpu",str(args.gpu),
                                ]
            with open(log_path,'w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)

        elif model == 'resnet':
            log_path = os.path.join(args.save_dir, 'extract_feature_resnet.log').replace('\\','/')
            subprocess_param = ["python", 
                                "[0][2]_extract_features_fp.py",
                                "--pretrained_model", 'resnet',
                                "--custom_downsample", str(args.custom_downsample),
                                "--data", args.data,
                                "--mag", str(args.mag),
                                "--specific_region", str(args.specific_region),
                                "--r", str(args.r),
                                "--ori_patch_size", str(args.actual_patch_size),
                                "--s",args.s,
                                "--atr",args.atr,
                                "--if_ddp",'0',
                                "--select_gpu",str(args.gpu),
            ]
            with open(log_path,'w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)

        elif model == 'hipt':
            log_path = os.path.join(args.save_dir, 'extract_feature_hipt.log').replace('\\','/')   
            subprocess_param = ["python", 
                        # "-m", "torch.distributed.launch", 
                        # "--nproc_per_node=4",
                        # "--master_port", "29502",
                        "[0][3]_extract_feature_hipt.py",
                        "--custom_downsample", str(args.custom_downsample),
                        "--data", args.data,
                        "--mag", str(args.mag),
                        "--specific_region", str(args.specific_region),
                        "--r", str(args.r),
                        "--train_model", '0',
                        "--ori_patch_size", str(args.actual_patch_size),
                        "--s",args.s,
                        "--atr",args.atr,
                        "--batch_size",str(1),
                        ]
            with open(log_path,'w') as f:
                completed = subprocess.run(subprocess_param, stdout=f,shell=False)


# iretatively process new region
""" python pipeline.py --inter 1
"""
if args.inter == True: 
    selected_region_list[args.data][-1]
    for idx, val in enumerate(selected_region_list[args.data]):
        args.r = idx
        args = default_setting(args)
        flag_list = [1,1,1,0,1,1,0] # pre_process, create_step2, create_step3, create_label_dist, create_split, create_graph, train
        pipiline(args, flag_list)

else:

    args = default_setting(args)
    flag_list = [1,1,1,0,1,0,1]
    pipiline(args, flag_list)