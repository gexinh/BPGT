""" A pipeline for extracting features from Dino 
        python -m torch.distributed.launch --nproc_per_node=8 train_test.py
"""


""" import packages
"""
# WSI relavent
from ast import arg
import numpy as np
import os

import argparse

import datetime
import json
import math
import sys
import time
from pathlib import Path

import h5py
import openslide
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import model.vision_transformer as vits
import utils.utils_dino as utils
from datasets.dataset_h5 import (Generic_WSI_Patch, Whole_Slide_Bag,
                                 Whole_Slide_Bag_FP)
from model.vision_transformer import DINOHead, vit_small
from utils.file_utils import save_hdf5
from utils.utils import collate_features, print_network
from utils.utils_dino import DataAugmentationDINO, DINOLoss

# Dino
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

""" Arg parser
"""
def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # key parameters
    parser.add_argument('--train_model', default=False,type=int, help='if train the model or juse use the pretrained one')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.') # default =100
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.") # default = 10
    
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac'], help='choose the dataset')
    parser.add_argument('--if_ddp', type=int, default=False, help='if use the DDP')
    parser.add_argument('--select_gpu', type=int,default=0, help='if use select gpu')
    # WSI
    parser.add_argument('--data_h5_dir', type=str, default='./data/cptac')
    parser.add_argument('--data_root_dir', type=str, default='./data/cptac')
    parser.add_argument('--ori_patch_size', type=int, default=256, help='the original patch size of WSI')
    parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--task', type=str, choices=['cptac',  'tpca'], default='cptac')
    parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
    parser.add_argument('--slide_ext', type=str, default= '.svs')

    # extract features
    parser.add_argument('--feat_dir', type=str, default='/')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")

    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    parser.add_argument('--output_dir', default="./checkpoints", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", choices=["env://", "tcp://localhost:23456" ],type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # for dataset
    parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
					help='choose the magnification of slides')
    parser.add_argument('--specific_region', type = bool, default=True,
					help='if process the specific region for tcga/cptac dataset')
    parser.add_argument('--r', type = int, default=0,
					help='the index of specific region for tcga/cptac dataset')
    parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
    parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')
    
    return parser


# load parser
parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
args = parser.parse_args()
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

if not args.if_ddp:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.select_gpu)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'

    os.environ["OMP_NUM_THREADS"] = "6"  
    os.environ["MKL_NUM_THREADS"] = "6"  
    import torch
    torch.set_num_threads(6)


# new version
""" data info 
"""
if args.data == 'cptac':
    args.region_list = ['BR', 'UCEC', 'LUAD', 'LSCC', 'CCRCC', 'CO', 'OV', 'GBM', 'HNSCC',  'PDA']
   
    args.data_root_dir = './data/cptac' if not args.specific_region else './data/cptac/' + args.region_list[args.r]
    args.data_h5_dir = './data/cptac' if not args.specific_region else './data/cptac/' + args.region_list[args.r]
    
    ckpt_path = os.path.join(args.output_dir, 'checkpoint0020.pth')

    args.feat_dir = 'pretrained_dino_feature'
    abs_path = '/data2/WSI_CPTAC'
    csv_path = args.data_root_dir + '/Step_3_.csv'

elif args.data == 'tcga':

    if args.s == 'pancan':
        source = 'pan'
    else:
        source = 'per'

    if args.atr == 'freq':
        tag = 'f'
    else:
        tag = 'd'
    
    args.region_list = ['BRCA', 'UCEC', 'LUAD','LUSC', 'KIRC', 'COAD', 'OV', 'GBM', 'HNSC', 'PAAD',
        'LGG', 'PRAD', 'THCA', 'SKCM', 'STAD', 'BLCA', 
        'LIHC', 'CESC', 'KIRP', 'SARC', 'ESCA', 'PCPG',
        'READ', 'LAML', 'TGCT', 'THYM', 'ACC', 'MESO', 
        'UVM', 'KICH', 'UCS', 'DLBC', 'CHOL']
    # args.region = region_list if not args.specific_region else [args.region_list[args.r]]

    args.data_root_dir = './data/tcga' if not args.specific_region else './data/tcga/' + args.region_list[args.r]
    args.data_h5_dir = './data/tcga' if not args.specific_region else './data/tcga/' + args.region_list[args.r]
    
    ckpt_path = os.path.join(args.output_dir, 'vits_tcga_brca_dino.pt')

    args.feat_dir = 'pretrained_dino_feature'


    abs_path = '/data2/WSI_TCGA'
    csv_path = args.data_root_dir + '/{}_{}_Step_3.csv'.format(source, tag)



if args.mag != -1:
    # ratio = 40/ args.mag since the max/default magnification is 40 
    add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))
else:
    args.mag = 40
    add_path = '/mag{}x_patch{}_fp'.format(args.mag, int(args.ori_patch_size))


args.split_dir = args.split_dir = os.path.join(args.data_root_dir + add_path + '/splits', args.data+'_{}'.format(int(args.label_frac*100)))
args.data_h5_dir = args.data_h5_dir + add_path
args.feat_dir = os.path.join(args.data_h5_dir, args.feat_dir)

csv_path = args.data_root_dir + add_path +'/{}_{}_Step_3.csv'.format(source, tag) if args.data == 'tcga' else args.data_root_dir + add_path +'/Step_3.csv'
dir_path_df = pd.read_csv(os.path.join(args.data_root_dir + add_path + '/Step_2.csv')) 



from data.data_loader import (Generic_ML_Dataset, Generic_ML_Table,
                              Generic_Split)

total_dataset = Generic_ML_Table(csv_path = csv_path,
                            dataset=args.data,
                            shuffle = False, 
                            seed = args.seed,
                            region_list = args.region_list
                            )

def custom_downsample_per_slide(slide_mag, expect_mag, default_mag=40, if_patch=False):
    """ custom script for downsample WSI with various magnifications

        Args:
            if_patch: when patching slides, we need to 

            default_mag: the default magnification of WSI, data with '.svs' format is 40x

    """
    if if_patch:
        expect_ratio = int(default_mag/expect_mag)
        factual_ratio = int(default_mag/slide_mag)
        custom_downsample = factual_ratio if factual_ratio<=expect_ratio else expect_ratio

    else:
        expect_ratio = int(slide_mag/expect_mag)
        custom_downsample = 1 if expect_ratio <= 1 else expect_ratio
    return custom_downsample


def obtain_data_loader(train_dataset, args):
    train_dataset_id = train_dataset.slide_data['slide_id'].values.tolist()
    dataloader_set = []
    coor_full_path_set = []
    wsi_path_set = []
    total_length = 0
    cur_length = [0]
    for bag_idx, slide_id in enumerate(train_dataset_id):        
        # if len(dataloader_set) != len(train_dataset_id) :

        # TODO: need modify this part based on different region data
        if args.mix_list is not None:
            region_name = train_dataset.slide_data['region'][bag_idx][5:] if args.data=='tcga' else train_dataset.slide_data['region'][bag_idx]
            cur_data_h5_dir = args.data_h5_dir.split('/')
            cur_data_h5_dir[3] = region_name
            cur_data_h5_dir = '/'.join(cur_data_h5_dir)
        else:
            cur_data_h5_dir = args.data_h5_dir
        #
        coor_full_path = os.path.join(cur_data_h5_dir,'patches','{}.h5'.format(slide_id))
        wsi_path = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'full_path'].values.tolist()
        wsi = openslide.open_slide(wsi_path[0])

        # TODO: custom downsample according to magnification
        slide_mag = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'mag']
        custom_downsample = custom_downsample_per_slide(slide_mag, args.mag)

        
        """ initial loader
        """
        transform = DataAugmentationDINO(
            args.global_crops_scale,
            args.local_crops_scale,
            args.local_crops_number,
        )
        # dataset = Whole_Slide_Bag(file_path=coor_full_path)
        dataset = Whole_Slide_Bag_FP(file_path=coor_full_path, wsi=wsi, pretrained=False, 
                custom_downsample=custom_downsample, target_patch_size=-1,custom_transforms=transform)
        """ data augmentation
        """                
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Data loaded: there are {len(dataset)} images.")
        print(f"Data loaded: The length of dataloader is {len(data_loader)}.")

        """ save them 
        """
        dataloader_set.append(data_loader)
        coor_full_path_set.append(coor_full_path)
        wsi_path_set.append(wsi_path)
        
        total_length += len(data_loader)
        cur_length.append(total_length)
    return dataloader_set, coor_full_path_set, wsi_path_set, total_length, cur_length

def train(args,total_dataset, i):
    """ build model
    """
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    """ ============ building student and teacher networks ... ============
    """
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim

    """ === multi-crop wrapper handles forward with inputs of different resolutions
    """
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # TODO: test DDP
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        # teacher = nn.parallel.DistributedDataParallel(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher

    # student = nn.parallel.DistributedDataParallel(student)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    """ initialize loss, optimizer, and scheduler
    """
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()


    print(f"Loss, optimizer and schedulers ready.")


    """ create empty dataloader set and their corresponding data augmentation
    """
    train_dataset, val_dataset, test_dataset = total_dataset.return_splits(split_bool_path='{}/splits_{}.csv'.format(args.split_dir, i))

    # TODO: debug
    # train_dataset_id = train_dataset_id[0:2]
    # initialize set
    dataloader_set, coor_full_path_set, wsi_path_set, total_length, cur_bag_length = obtain_data_loader(train_dataset, args)

    """ scheduler
    """
    # ============ init schedulers ... ============
    # learning rate
    # weight decay
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, total_length,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, total_length,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                                args.epochs, total_length)    
    print("Schedulers have been initialized")
    # # initialize set
    # dataloader_set = []
    # lr_schedule_set = []
    # wd_schedule_set = []
    # momentum_schedule_set = []
    # coor_full_path_set = []
    # wsi_path_set = []

    """ traning epoch loop
    """
    from datasets.dataset_h5 import Whole_Slide_Bag
    start_time = time.time()
    print("\n====================================================")
    print("Starting DINO training !")
    for epoch in range(args.epochs):

        print("\n current epoch is {}".format(epoch))
        print("====================================================")
        """ slide loop
        """

        for bag_idx, slide_id in enumerate(train_dataset_id):        

            data_loader = dataloader_set[bag_idx]
            data_loader.sampler.set_epoch(epoch)
            # ============ training one epoch of DINO ... ============
            # train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            #     data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            #     epoch, fp16_scaler, args)

            # one epoch
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Epoch: [{}/{}], Slide: {}'.format(epoch, args.epochs, bag_idx)    

            for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
                # update weight decay and learning rate according to their schedule
                
                it = total_length * epoch + cur_bag_length[bag_idx] + it  # global training iteration
                # print("the iteration / the length of scheduler: [{}/{}]".format(it,len(momentum_schedule)))
                # it = len(data_loader) * epoch + it  # global training iteration 
                
                for i, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = lr_schedule[it]
                    if i == 0:  # only the first group is regularized
                        param_group["weight_decay"] = wd_schedule[it]

                # move images to gpu
                images = [im.cuda(non_blocking=True) for im in images]
                # teacher and student forward passes + compute dino loss
                with torch.cuda.amp.autocast(fp16_scaler is not None):
                    # TODO: maybe need to modify this part    
                    teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
                    student_output = student(images)
                    loss = dino_loss(student_output, teacher_output, epoch)

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()), force=True)
                    sys.exit(1)

                # student update
                optimizer.zero_grad()
                param_norms = None
                if fp16_scaler is None:
                    loss.backward()
                    if args.clip_grad:
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                    args.freeze_last_layer)
                    optimizer.step()
                else:
                    fp16_scaler.scale(loss).backward()
                    if args.clip_grad:
                        fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                        param_norms = utils.clip_gradients(student, args.clip_grad)
                    utils.cancel_gradients_last_layer(epoch, student,
                                                    args.freeze_last_layer)
                    fp16_scaler.step(optimizer)
                    fp16_scaler.update()

                # EMA update for the teacher
                with torch.no_grad():                    
                    m = momentum_schedule[it]  # momentum parameter
                    for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                # logging
                torch.cuda.synchronize()
                metric_logger.update(loss=loss.item())
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            print("Traning slide image {} finish!".format(bag_idx))
            print("=========================================================\n")
            # return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

            # for patch_ind, patch_data in enumerate(data_loader):
        """ record dictionary
        """
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            # 'epoch': epoch + 1 if bag_idx==0 else epoch,
            # 'data_index': bag_idx,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if args.train_model == True:
    train(args, total_dataset, i)

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=4, pretrained=False, 
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

""" eval pretrained feature for train, val, and test set.
"""
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# build model
model = vit_small(patch_size=args.patch_size)
# model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
for p in model.parameters():
    p.requires_grad = False

# if torch.cuda.device_count() > 1:
#     	model = nn.DataParallel(model)
# utils.init_distributed_mode(args)
model.to(device)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
model.eval()
""" load weight
"""
# ckpt_path = os.path.join(assets_dir, enc_name+'.pt')
args.pretrained_weights = args.output_dir 


assert os.path.isfile(ckpt_path)
state_dict = torch.load(ckpt_path, map_location="cpu")
if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
    print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
    # state_dict = torch.load(ckpt_path, map_location="cpu")['teacher']
    state_dict = torch.load(ckpt_path, map_location="cpu")[args.checkpoint_key]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # remove `module.` prefix
state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()} # remove `backbone.` prefix induced by multicrop wrapper
msg = model.load_state_dict(state_dict, strict=False)
print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
# missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)


# create folders
os.makedirs(args.feat_dir, exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

# TODO: load data respectively, which is due to the code of training model 
train_dataset, val_dataset, test_dataset = total_dataset.return_splits(split_bool_path='{}/splits_{}.csv'.format(args.split_dir, i))
data_set_ind = []


train_dataset_id = train_dataset.slide_data['slide_id'].values.tolist()
val_dataset_id = val_dataset.slide_data['slide_id'].values.tolist()
test_dataset_id = test_dataset.slide_data['slide_id'].values.tolist()

train_dataset_id = list(set(train_dataset_id))
val_dataset_id = list(set(val_dataset_id))
test_dataset_id = list(set(test_dataset_id))

print('The number of Train, Val, and Test set is {}, {}, and {}'.format(len(train_dataset_id),len(val_dataset_id),len(test_dataset_id)))

data_set_ind = [train_dataset_id, val_dataset_id, test_dataset_id]

for data in data_set_ind:
    total =len(data)

    for bag_candidate_idx in range(total):
        # confirm related url
        slide_id = data[bag_candidate_idx]
        h5_file_path = os.path.join(args.data_h5_dir,'patches','{}.h5'.format(slide_id))
        
        # modify this part for different sever
        #/data2/WSI_CPTAC/BR/BR/01BR001-0684a407-f446-486d-9160-b483cb.svs

        wsi_path = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'full_path'].values.tolist()
        
        wsi_path = os.path.join(abs_path,  wsi_path[0].split('/',3)[-1])
        wsi = openslide.open_slide(wsi_path)

        # wsi = openslide.open_slide(wsi_path[0])
        bag_name = slide_id + '.h5'

        print('\nprogress: {}/{}'.format(bag_candidate_idx+1, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue 

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)

        #
        slide_mag = dir_path_df.loc[dir_path_df['slide_id']==slide_id,'mag']
        custom_downsample = custom_downsample_per_slide(slide_mag, args.mag)
        
        # load model 
        time_start = time.time()
        output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
        model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
        custom_downsample=custom_downsample, target_patch_size=args.target_patch_size)

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
        file = h5py.File(output_file_path, "r")

        features = file['features'][:]
        print('features size: ', features.shape)
        print('coordinates size: ', file['coords'].shape)
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))