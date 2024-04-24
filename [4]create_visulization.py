#
from __future__ import print_function
import imp
from typing_extensions import dataclass_transform

import numpy as np
import argparse
import os

import torch
import torch.nn as nn
import pdb

import pandas as pd

from math import floor
from types import SimpleNamespace
from sklearn.preprocessing import Binarizer
from collections import namedtuple
import h5py
import yaml

from utils.utils import *
from utils.eval_utils import initiate_model as initiate_model
from utils.file_utils import save_hdf5
from utils.file_utils import root_arg, param_selection_for_dataset, get_args_parser_for_dataset

from wsi_core.batch_process_utils import initialize_df
from wsi_core.wsi_utils import sample_rois
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches, compute_from_patches_for_ML

# import model
from model.base_model import CLAM_SB, MLDecoder, query2label, transformer_decoder,graphormer_decoder
from model.resnet_custom import resnet50_baseline

# paramters
"""
   args:
        patch_size
        mag
        custom_downsample
        
        save_path
        data_root: 
        add_path: mag_{}_patch_{}
        
        data
        source
        atr
        region
        
"""
def ArgsParse():
    parser = argparse.ArgumentParser(description='Heatmap inference script', parents=[get_args_parser_for_dataset()])
    parser.add_argument('--save_exp_code', type=str, default=None,
                        help='experiment code')
    parser.add_argument('--overlap', type=float, default=None, help='the overlap of heatmap')
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    
    parser.add_argument('--ckpt_path', type=str, default="checkpoint path")
    

    args = parser.parse_args()

    args.mag = 20
    args.pretrained_model = 'dino'
    args.downstream_model = 'ml_decoder'
    args.agg_method = 'cor_mil'
    args.model_type = 'att_ml'
    args.model_size = 'dino' 
    args.graph_module = 'graphormer'
    args.specific_region = False
    args.mix_list = '012345678'

    return args

args = ArgsParse()
args.label_space = 'final'
args.count = 0
args = param_selection_for_dataset(args.data, args.s, args.atr, args)
args.k_start = 0  # choose different folds

sample_rate = 0.2  # sample rate to visualize samples
sample_mode = 'random' # random / accord_label
# sample_mode = 'accord_label' # random / accord_label
custom_roi_size = (64, 64)
custom_best_level = 1  # vis downsample rate

args.weight_adj = True
# args.exp_count = 36
# args.exp_count = 0
args.exp_count = 1
args.seed = 1
# add parameter
args.pretrained_model = 'dino'
args.model_type = 'att_ml'
args.downstream_model = 'ml_decoder'


# load gene_list
f = pd.read_csv(args.csv_2_path)
args.gene_list = f.columns.values.tolist()[args.start_ind:]





""" model parameter
"""
arg_key = []
for i in args._get_kwargs():
    arg_key.append(i[0])
if 'pretrained_model' in arg_key:
    if args.pretrained_model == 'dino': 
        args.feat_file = 'pretrained_dino_feature'

    elif args.pretrained_model == 'resnet':
        args.feat_file = 'pretrained_resnet_feature'
        args.model_size = '1024_min'
    
    elif args.pretrained_model == 'hipt':
        args.feat_file = 'pretrained_hipt_feature'
        args.model_size = 'hipt'
        # args.patch_size = 4096
else:
    args.feat_file = 'pretrained_dino_feature'

#path need to modify
if 'model_type' in arg_key:
    if args.model_type == 'att_ml':
    #     args.exp_name = 'model_{}_task_{}_data_{}_region_{}'.format(args.model_type, 'mlabel', args.data,args.region)
    # else:
    #     gene = args.gene_list[args.label_ind]
    #     args.exp_name = 'model_{}_task_{}_data_{}_region_{}'.format(args.downstream_model, gene, args.data,args.region)
        if args.downstream_model == 'mlp':
            args.exp_name = 'type_{}_model_{}_data_{}_region_{}'.format(args.model_type, args.downstream_model, args.data, args.region)
        elif args.downstream_model == 'ml_decoder':
            args.exp_name = 'type_{}_model_{}_graph_{}_data_{}_region_{}'.format(args.model_type, args.downstream_model, args.graph_module, args.data,args.region)
        elif args.downstream_model == 'exp_pipe':
            args.exp_name = 'model_{}/graph_{}_data_{}_region_{}'.format(args.model, args.graph_module, args.data, args.region)
    

args.result_path = './heatmaps/' + args.exp_name

args.results_dir = './experiments/' + args.exp_name
args.json_dir = args.results_dir
args.results_dir = os.path.join(args.results_dir, 'exp_{}'.format(args.exp_count))
args.results_dir = os.path.join(args.results_dir, 'seed_{}'.format(args.seed))


""" Obtain hyper-parameters
"""
from utils.general_utils import Params,mkdir
json_path = os.path.join(args.json_dir, 'exp_{}'.format(args.exp_count),'params.json').replace('\\', '/') 
assert os.path.isfile(
    json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

for j, k in enumerate(params.__dict__):
    if k != 'patch_size':
        v = params.__dict__[k]
        args.__dict__[k] = v      

print(args.model_type )

# need to modify
if args.model_type == 'att_ml':
    #
    modify_path = args.csv_2_path[:-4] + 'm' + '.csv'
    modify_data_list = pd.read_csv(modify_path)
    select_gene = modify_data_list.columns.values.tolist()[args.start_ind:-1]

    #TODO: for graphormer
    args.df = modify_data_list

    # adj
    args.list_dir = './data/processed data'
    graph_dir = './data/processed data/graph/'
    if args.data == 'tcga':
        graph_atr = args.region+'_'+args.s+'_'+args.atr if args.label_category =='react' else args.region + '_oncokb'
    else:
        graph_atr = args.region
    # graph_atr = args.region+'_'+args.s+'_'+args.atr

    import networkx as nx
    adj = pd.read_csv(graph_dir + graph_atr+'_adj.csv',index_col=0)
    w_adj = pd.read_csv(graph_dir + graph_atr+'_w_adj.csv',index_col=0)
    adj = adj.loc[select_gene,select_gene]
    w_adj = w_adj.loc[select_gene, select_gene]

    Gs = nx.from_pandas_adjacency(adj, create_using=nx.Graph)
    Gw = nx.from_pandas_adjacency(w_adj, create_using=nx.Graph)

    if args.weight_adj:
        adj_mx = nx.adjacency_matrix(Gw).todense()
    else:
        adj_mx = nx.adjacency_matrix(Gs).todense()
    #                             #
    # normalized Laplacian matrix #
    #                             #
    from scipy.sparse import csgraph

    # add new row for adjacency matrix due to the all-zero label 
    adj_mx = np.append(adj_mx, np.zeros((1, adj_mx.shape[1])), axis=0)
    adj_mx = np.append(adj_mx, np.zeros((adj_mx.shape[0],1)), axis=1)
    args.un_adj = torch.tensor(adj_mx,dtype=torch.float32) # unprocess

    # normalized laplacian matrix
    adj_mx = csgraph.laplacian(adj_mx, normed=True)
    # add self loop for the zero entry on the diagnal line
    for i in range(adj_mx.shape[0]):
        if any(adj_mx[i,:])==0:
            adj_mx[i,i] = 1

    # to tensor
    adj_mx = torch.tensor(adj_mx,dtype=torch.float32)
    # save in args
    args.process_gene_list = select_gene
    args.adj = adj_mx


    # n_label, num_region
    n_label = len(modify_data_list.columns.values.tolist()[args.start_ind:])
    region = modify_data_list.iloc[:,args.start_ind-1].tolist()
    unique, counts = np.unique(region, return_counts=True)  
    num_region = len(unique)

    # 
    args.n_label = n_label
    #   
    args.num_region = num_region

    from utils.utils import load_embedd
    model_name = args.word_embedd_model
    if model_name != 'one_hot':
        # save_path = os.path.join(args.list_dir+'/list', 'model_{}_encoded_gene_card_embed.h5'.format(model_name))
        save_path = os.path.join(args.list_dir+'/list','{}_{}_model_{}_encoded_gene_card_embed.h5'.format(args.data, args.label_category, model_name))
        word_list = select_gene
        word_embedd = load_embedd(save_path, word_list, covert_pt=True)

        # add a row with average pooling for 'non-mutation' node
        add_row = torch.sum(word_embedd,dim=0,keepdim=True)
        args.word_embedd = torch.cat( (word_embedd, add_row), dim=0)
    else:
        args.word_embedd = None


"""
    exp_arguments:
        # number of classes
        n_classes: 2
        # name tag for saving generated figures and assets
        save_exp_code: HEATMAP_OUTPUT 
        # where to save raw asset files
        raw_save_dir: heatmaps/heatmap_raw_results
        # where to save final heatmaps
        production_save_dir: heatmaps/heatmap_production_results
        batch_size: 384

    data_arguments: 
        # where is data stored; can be a single str path or a dictionary of key, data_dir mapping
        data_dir: heatmaps/demo/slides/
        # column name for key in data_dir (if a dict mapping is used)
        data_dir_key: source
        # csv list containing slide_ids (can additionally have seg/patch paramters, class labels, etc.)
        process_list: heatmap_demo_dataset.csv
        # preset file for segmentation/patching
        preset: presets/bwh_biopsy.csv
        # file extention for slides
        slide_ext: .svs
        # label dictionary for str: interger mapping (optional)
        label_dict:
            LUAD: 0
            LSCC: 1   

    patching_arguments:
        # arguments for patching
        patch_size: 256
        overlap: 0.5
        patch_level: 0
        custom_downsample: 1

    model_arguments: 
        # arguments for initializing model from checkpoint
        ckpt_path: heatmaps/demo/ckpts/s_0_checkpoint.pt
        model_type: clam_sb # see utils/eval_utils/
        initiate_fn: initiate_model # see utils/eval_utils/
        model_size: small
        drop_out: true
"""

""" useful functions
"""
def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code

    if args.result_path is not None: 
        """ './heatmaps/exp_name/raw(product)/gene_{}/slide_id'
        """ 
        config_dict['exp_arguments']['raw_save_dir'] = 'raw_result' 
        config_dict['exp_arguments']['production_save_dir'] =  'production'

    if args.abs_path is not None:
        config_dict['data_arguments']['data_dir'] = os.path.join(args.abs_path,args.region)

    if args.save_path:
        config_dict['data_arguments']['data_root_dir'] = args.save_path

    # process_list


    # data_dir_key
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    if args.custom_downsample is not None:
        config_dict['patching_arguments']['custom_downsample'] = args.custom_downsample
    if args.patch_size is not None:
        config_dict['patching_arguments']['patch_size'] = args.patch_size
        

    # model 
    if args.ckpt_path is not None:
        config_dict['model_arguments']['ckpt_path'] = args.ckpt_path
    if args.model_type is not None:
        config_dict['model_arguments']['model_type'] = args.model_type
    if args.model_size is not None:
        config_dict['model_arguments']['model_size'] = args.model_size
    if args.drop_out is not None:
        config_dict['model_arguments']['drop_out'] = args.drop_out  

    #model_type
    #model_size

    return config_dict


def infer_single_slide_for_ml(model, features):
    features = features.to(device)
    with torch.no_grad():
        if isinstance(model, (query2label, transformer_decoder, graphormer_decoder)):            
            logits, probs, results_dict = model(features)
            binarizer = Binarizer(threshold=0.5)  
            Y_hat = binarizer.transform(probs.detach().cpu().numpy()) # [1, N_label]
            A = results_dict['a_score']

            # TODO: new feature map


            if len(A.size()) == 3:
                A = A.squeeze()
            A = A.detach().cpu().numpy()  # [batch, N_label, patch]
            
            if isinstance(model, graphormer_decoder):
                Y_attn = results_dict['Y_attn']
                Y_emb = results_dict['Y_emb']
                
                Y_emb_p = results_dict['Y_emb_p']
                one_emb_p = results_dict['one_emb_p']

                Y_init = results_dict['Y_init']

                if len(Y_attn.size()) == 3:
                    Y_attn = Y_attn.squeeze()
                if len(Y_emb.size()) == 3:
                    Y_emb = Y_emb.squeeze()

                Y_attn = Y_attn.detach().cpu().numpy()
                Y_emb = Y_emb.detach().cpu().numpy()

                Y_emb_p = Y_emb_p.detach().cpu().numpy()
                one_emb_p = one_emb_p.detach().cpu().numpy()

                Y_init = Y_init.detach().cpu().numpy()

                A = [A, Y_attn, Y_emb, Y_emb_p, one_emb_p, Y_init]
        else:
            raise NotImplementedError

    return Y_hat, A


""" Begin 
"""

# config path
config_path = os.path.join('heatmaps/configs', args.config_file)
config_dict = yaml.safe_load(open(config_path, 'r'))
config_dict = parse_config_dict(args, config_dict)

# print
for key, value in config_dict.items():
    if isinstance(value, dict):
        print('\n'+key)
        for value_key, value_value in value.items():
            print (value_key + " : " + str(value_value))
    else:
        print ('\n'+key + " : " + str(value))

# Interaction
# decision = input('Continue? Y/N ')
# if decision in ['Y', 'y', 'Yes', 'yes']:
#     pass
# elif decision in ['N', 'n', 'No', 'NO']:
#     exit()
# else:
#     raise NotImplementedError

# construct data path
patch_save_dir = os.path.join(args.data_root_dir, 'patches')
mask_save_dir = os.path.join(args.data_root_dir, 'masks')
stitch_save_dir = os.path.join(args.data_root_dir, 'stitches')

#  Parse parameter
multi_args = config_dict

patch_args = argparse.Namespace(**multi_args['patching_arguments']) 

data_args = argparse.Namespace(**multi_args['data_arguments'])

model_args = multi_args['model_arguments']
# model_args.update({'n_classes': args['exp_arguments']['n_classes']})
model_args = argparse.Namespace(**model_args)

exp_args = argparse.Namespace(**multi_args['exp_arguments'])

heatmap_args = argparse.Namespace(**multi_args['heatmap_arguments'])

sample_args = argparse.Namespace(**multi_args['sample_arguments'])

# TODO: modify
patch_args.overlap = 0

patch_size = tuple([patch_args.patch_size for i in range(2)])
step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))



####### use preset segmentation setting  ###############
preset = data_args.preset  # tcga.csv
# def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                    # 'keep_ids': 'none', 'exclude_ids':'none'}
# def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}

def_seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				'keep_ids': 'none', 'exclude_ids': 'none'}
def_filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}

def_vis_params = {'vis_level': -1, 'line_thickness': 250}
def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

if preset is not None:
    preset_df = pd.read_csv(preset)
    for key in def_seg_params.keys():
        def_seg_params[key] = preset_df.loc[0, key]

    for key in def_filter_params.keys():
        def_filter_params[key] = preset_df.loc[0, key]

    for key in def_vis_params.keys():
        def_vis_params[key] = preset_df.loc[0, key]

    for key in def_patch_params.keys():
        def_patch_params[key] = preset_df.loc[0, key]


##### load data_generate.csv ############
if data_args.process_list is None:
    from utils.file_utils import get_img_path
    full_path, slides = get_img_path(data_flag=args.data, source_path=data_args.data_dir, region_list=[args.region])    
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, full_path=full_path, use_heatmap_args=False)
    
else:
    df = pd.read_csv(os.path.join(data_args.data_root_dir, 'Step_2.csv')) 
    # df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

###### 
# TODO: modify for checking the status
mask = df['status'] == 'processed'

# load checkpoint of pretrained model
df = df[mask].reset_index(drop=True)

# # select gene
# gene = args.gene_list[args.label_ind]

gene_list = args.process_gene_list
#
blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

# checkpoint path
if args.k_start == -1:
    start = 0
else:
    start = args.k_start

    # if args.k_end == -1:
    #     end = args.k
    # else:
    #     end = args.k_end

    end = start + 1
folds = np.arange(start, end)
for cur in folds:
    args.ckpt_path = os.path.join(args.results_dir,'s_{}_checkpoint.pt'.format(str(cur)))
    print('\ninitializing model from checkpoint')
    ckpt_path = args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    # initiate
    if model_args.initiate_fn == 'initiate_model':
        model =  initiate_model(args, ckpt_path)
    else:
        raise NotImplementedError

    """# loop for dataset
    """    
    # datasplit = ['train', 'val', 'test']
    datasplit = ['train']
    # datasplit = ['val', 'test', 'train']
    for q in range(len(datasplit)):    
        # q = 0
        process_stack = df  # total dataset

        ## select from splits
        split = pd.read_csv(os.path.join(data_args.data_root_dir, 'splits',args.data+'_100','splits_{}.csv'.format(cur)))
        
        # choose for dataset
        ids = process_stack['slide_id'].isin(split[datasplit[q]])
        # ids = process_stack['slide_id'].isin(split[datasplit[q]]+data_args.slide_ext)
        process_stack = process_stack.loc[ids].reset_index(drop=True)

        # select for dataset
        sum(modify_data_list['slide_id'].isin(split[datasplit[q]]))
        ids_1 = process_stack['slide_id'].isin(modify_data_list['slide_id'])

        process_stack = process_stack.loc[ids_1].reset_index(drop=True)
        label_stack = modify_data_list

        total = len(process_stack)

        # # original data info
        # cur_data_info = process_stack

        # randomly select
        if sample_mode == 'random':
            if datasplit[q] == 'train':
                # fix number
                process_stack = process_stack.sample(n=1, replace=False, random_state=1).reset_index(drop=True)

            else:     
                process_stack = process_stack.sample(n=1, replace=False, random_state=1).reset_index(drop=True)    
                # process_stack = process_stack.sample(frac=sample_rate, replace=False, random_state=1).reset_index(drop=True)

        # choose positive label 
        elif sample_mode == 'accord_label':
            
            select_sld = {'tp':{},'tn':{}} # true positive, true negative
            select_idx = []
            s_idx = modify_data_list['slide_id'].isin(process_stack['slide_id'])
            select_data_list = modify_data_list.loc[s_idx].reset_index(drop=True)

            for j, gene in enumerate(gene_list):                
                # positive
                pos_select_sld = select_data_list.loc[select_data_list[gene] == 1]
                neg_select_sld = select_data_list.loc[select_data_list[gene] == 0]

                if len(pos_select_sld) >= 1:
                    p_list = [pos_select_sld['slide_id'].reset_index(drop=True)[i] for i in range(len(pos_select_sld))]
                    p_flag = 1
                    for k in p_list:
                        p_flag = p_flag*(k in select_idx)
                    p_bool_list = [k for k in p_list if not k in select_idx] 

                    while True:
                        cur_pos_select_idx = pos_select_sld.sample(n=1, replace=False, random_state=1)['slide_id'].reset_index(drop=True)[0]                        
                        if not cur_pos_select_idx in select_idx:                       
                            break
                        else:
                            if not p_flag:
                                cur_pos_select_idx = p_bool_list[0]
                                break
                            else:
                                break
                    
                    select_idx.append(cur_pos_select_idx)                    
                else:
                    cur_pos_select_idx = 'empty'

                if len(neg_select_sld) >= 1 :
                    n_list = [neg_select_sld['slide_id'].reset_index(drop=True)[i] for i in range(len(neg_select_sld))]
                    n_flag = 1
                    n_bool_list = [k for k in n_list if not k in select_idx] 
                    for k in n_list:
                        n_flag = n_flag*(k in select_idx)

                    while True:
                        cur_neg_select_idx = neg_select_sld.sample(n=1, replace=False, random_state=1)['slide_id'].reset_index(drop=True)[0]
                        if cur_neg_select_idx in select_idx:
                            break
                        else:
                            if not p_flag:
                                cur_neg_select_idx = n_bool_list[0]
                                break
                            else:
                                break

                    select_idx.append(cur_pos_select_idx)
                else:
                    cur_neg_select_idx = 'empty'
                    
                select_sld['tp'][gene] = cur_pos_select_idx
                select_sld['tn'][gene] = cur_neg_select_idx

            # sort
            select_idx = list(sorted(set(select_idx)))
            select_idx = process_stack['slide_id'].isin(select_idx)
            process_stack = process_stack.loc[select_idx].reset_index(drop=True)
            


        print('\nNumber of slides to process: ')
        print(len(process_stack))
         
        print('\nlist of slides to process: ')
        print(process_stack['slide_id'].head(len(process_stack)))

        # data path
        heatmap_path = os.path.join(args.result_path,datasplit[q])
        os.makedirs(heatmap_path, exist_ok=True)    

        # save process_list
        process_stack.to_csv(os.path.join(heatmap_path,'{}_process_list.csv'.format(datasplit[q])))

        # os.makedirs(exp_args.production_save_dir, exist_ok=True)
        # os.makedirs(exp_args.raw_save_dir, exist_ok=True)

        blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}


        """ macro and micro
        """
        # visual_level = ['macro', 'micro']
        visual_level = ['macro']

        # visual_level = ['heatmap','macro']
        for lvl in visual_level:
            if lvl == 'heatmap':
                vis_path = os.path.join(heatmap_path, 'macro')
            else:
                vis_path = os.path.join(heatmap_path, lvl)
            os.makedirs(vis_path, exist_ok=True)
            """ slide process
            """
            for i in range(len(process_stack)):
                slide_name = process_stack.loc[i, 'slide_id']
                cur_region = process_stack.loc[i, 'region']

                if data_args.slide_ext not in slide_name:
                    slide_name += data_args.slide_ext
                print('\nprocessing: ', slide_name)    

                ########## path #####################
                # extract feature
                slide_path = process_stack.loc[i, 'full_path']

                slide_id = slide_name.replace(data_args.slide_ext, '')
                Y_label = label_stack.loc[label_stack['slide_id'].isin([slide_id])].iloc[:, args.start_ind:]

                # produced data path
                slide_save_dir = os.path.join(vis_path, slide_id)

                p_slide_save_dir = os.path.join(slide_save_dir, exp_args.production_save_dir)
    
                # raw data path: save data
                r_slide_save_dir = os.path.join(slide_save_dir, exp_args.raw_save_dir)

                if lvl in ['macro', 'heatmap']:
                    os.makedirs(slide_save_dir, exist_ok=True)
                    os.makedirs(p_slide_save_dir, exist_ok=True)
                    os.makedirs(r_slide_save_dir, exist_ok=True)

                # TODO: block_map_save_path is the score map path    
                mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
                block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
                mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))

                #TODO: modify for multi-region data
                f_path = args.data_root + '/' + cur_region + args.add_path    
                features_path = os.path.join(f_path, args.feat_file, 'pt_files',slide_id+'.pt')
                features_h5_path = os.path.join(f_path, args.feat_file, 'h5_files',slide_id+'.h5')
                h5_path = os.path.join(f_path, 'patches',slide_id+'.h5')
                # features_path = os.path.join(data_args.data_root_dir, args.feat_file, 'pt_files',slide_id+'.pt')
                # features_h5_path = os.path.join(data_args.data_root_dir, args.feat_file, 'h5_files',slide_id+'.h5')
                # h5_path = os.path.join(data_args.data_root_dir, 'patches',slide_id+'.h5')
                ############ parameter ####################
                if heatmap_args.use_roi:
                    x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
                    y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
                    top_left = (int(x1), int(y1))
                    bot_right = (int(x2), int(y2))
                else:
                    top_left = None
                    bot_right = None
                
                print('slide id: ', slide_id)
                print('top left: ', top_left, ' bot right: ', bot_right)

                

                # Load segmentation and filter parameters
                seg_params = def_seg_params.copy()
                filter_params = def_filter_params.copy()
                vis_params = def_vis_params.copy()

                seg_params = load_params(process_stack.loc[i], seg_params)
                filter_params = load_params(process_stack.loc[i], filter_params)
                vis_params = load_params(process_stack.loc[i], vis_params)

                keep_ids = str(seg_params['keep_ids'])
                if len(keep_ids) > 0 and keep_ids != 'none':
                    seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
                else:
                    seg_params['keep_ids'] = []

                exclude_ids = str(seg_params['exclude_ids'])
                if len(exclude_ids) > 0 and exclude_ids != 'none':
                    seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
                else:
                    seg_params['exclude_ids'] = []

                for key, val in seg_params.items():
                    print('{}: {}'.format(key, val))

                for key, val in filter_params.items():
                    print('{}: {}'.format(key, val))

                for key, val in vis_params.items():
                    print('{}: {}'.format(key, val))

                ###### load WSI and segement and masking #############
                print('Initializing WSI object')
                if lvl in ['macro','heatmap']:
                    wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
                elif lvl == 'micro':
                    wsi_object = initialize_wsi(slide_path, seg_params=seg_params, filter_params=filter_params)
                print('Done!')

                # TODO: best downsample rate

                wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]  #downsample factor

                # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
                vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

                # visual level: best level
                # 
                if vis_params['vis_level'] < 0:
                    # level:  1, 2, 3, which means downsample rates
                    best_level = wsi_object.wsi.get_best_level_for_downsample(32)
                    # best_level = wsi_object.wsi.get_best_level_for_downsample(32)
                    vis_params['vis_level'] = best_level

                # TODO: turn-off
                if lvl in ['macro', 'heatmap']:
                    vis_params['vis_level'] = custom_best_level
                    mask = wsi_object.visWSI(**vis_params, number_contours=True)
                    mask.save(mask_path)    
                


                """ load data and coordinates
                """
                ########## load segmentation and features ############
                # if os.path.isfile(block_map_save_path): 
                if not os.path.isfile(block_map_save_path): 
                    features = torch.load(features_path)
                    process_stack.loc[i, 'bag_size'] = len(features)
                    print('the number of patches is {}'.format(len(features)))
                    ######## downstream model to get score ######################
                    # TODO: load Y_label
                    # Y_label = process_stack.iloc[i, args.start_ind:] # contain non-mutation 
                    
                    Y_hat, A = infer_single_slide_for_ml(model, features)
                    if len(A) == 3:
                        Y_attn = A[1]
                        Y_emb = A[2]
                        A = A[0]
                        Y_emb_p = None
                        one_emb_p = None
                    elif len(A) == 6:
                        Y_init = A[5]
                        Y_emb_p = A[3]
                        one_emb_p = A[4]
                        Y_attn = A[1]
                        Y_emb = A[2]
                        A = A[0]

                    print('the number of patches score is {}'.format(np.shape(A)[-1]))
                    if args.agg_method == "cor_mil":
                        pad_num = np.shape(A)[-1] - len(features)
                        A = A[:,:-pad_num]
                        print('the modified number of patches score is {}'.format(np.shape(A)[-1]))

                    Y_label = np.squeeze(Y_label.values.tolist()) # [K,]
                    Y_hat = np.squeeze(Y_hat) # [K,]

                    del features

                    # coords also save in the feature.h5 files
                    
                    feature_h5 = h5py.File(features_h5_path, 'r')
                    features = feature_h5['features'][:] 
                    feat_coords = feature_h5['coords'][:]
                    feature_h5.close()
                    print('the number of patches coordinates is {}'.format(len(feat_coords)))



                    # block_map: [score, coord]
                    file = h5py.File(h5_path, "r") 
                    coords = file['coords'][:] 
                    file.close()
                    if sum(sum(feat_coords-coords)) == 0:
                        print('coordinates of features and patchs are respectively consistent')
                    asset_dict = {'attention_scores': A, 'coords': coords, 
                                'Y_label': Y_label, 'Y_hat': Y_hat, 
                                'Y_attn':Y_attn,'Y_emb':Y_emb,
                                'one_emb_p':one_emb_p,
                                'Y_emb_p': Y_emb_p,
                                'Y_init': Y_init,
                                 }
                    block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

                ####### block map load ###############
                file = h5py.File(block_map_save_path, 'r')
                dset = file['attention_scores']
                coord_dset = file['coords']
                scores = dset[:]  # TODO: need to modify for multi-label classification
                coords = coord_dset[:]
                Y_label = file['Y_hat'][:]
                Y_hat = file['Y_hat'][:]
                #TODO:
                Y_attn = file['Y_attn'][:]
                Y_emb = file['Y_emb'][:]
                Y_emb_p = file['Y_emb_p'][:]
                one_emb_p = file['one_emb_p'][:]
                Y_init = file['Y_init'][:]
                file.close()
                

                #### save high score region based on region######

                # loop for gene
                sample = sample_args.samples[0]
                sample['k'] = 5
                """  samples:
                        - name: "topk_high_attention"
                        sample: true
                        seed: 1
                        k: 15 # save top-k patches
                        mode: topk
                """
                attn_save_dir = os.path.join(slide_save_dir,'attnmap')
                if lvl == 'macro':
                    os.makedirs(attn_save_dir, exist_ok=True)


                heatmap_save_dir = os.path.join(slide_save_dir,'heatmap')
                if lvl == 'heatmap':
                    os.makedirs(heatmap_save_dir, exist_ok=True)

                    # plot heatmap for 


                    # plot tSNE for label
                    import numpy as np
                    from sklearn.manifold import TSNE
                    from matplotlib import pyplot as plt
                    import seaborn as sns
                    import colorcet as cc

                    # one -hot 
                    num_emb = np.shape(Y_emb)[-1]
                    K = np.shape(Y_emb)[0]
                    m =  torch.normal(mean=0., std=1.0,size=(K, num_emb))
                    nn.init.xavier_normal_(m)
                    one_emb = torch.matmul(torch.eye(K),m).numpy()

                    one_hot_embedded = TSNE(n_components=2, 
                                            init='pca', perplexity=3).fit_transform(one_emb[:-1,:])
                    # init  
                    init_embedded = TSNE(n_components=2, 
                                            init='pca', perplexity=3).fit_transform(Y_init[:-1,:])
                    figure, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,
                                               figsize = (8, 4))
                    # plt.subplots_adjust(hspace=0.3, wspace=0.3)

                    palette = sns.color_palette(cc.glasbey, n_colors=len(gene_list))

                    plot_1 = sns.scatterplot(one_hot_embedded[:,0], one_hot_embedded[:,1], hue=gene_list, 
                                    legend=None, 
                                    palette=palette,
                                    ax= ax1, 
                                    )
                    plot_1.set_title('tSNE of One-hot Label Embedding')
                    plot_ = sns.scatterplot(init_embedded[:,0], init_embedded[:,1], hue=gene_list, 
                                    palette=palette,
                                    ax= ax2, 
                                    )
                    plot_.set_title('tSNE of linguistic Label Embedding')

                    locs = ['upper right', 'lower left', 'center left', 'lower center', 'center',
                            'right']
                    ax2.legend().remove()
                    lgnd = figure.legend(ncol=10, bbox_to_anchor=(1,0), loc=locs[0],fontsize=7)
                    #change the marker size manually for both lines
                    for i in range(len(lgnd.legendHandles)):
                        lgnd.legendHandles[i]._sizes = [32]
                    figure.tight_layout()              
                    figure.savefig(os.path.join(heatmap_save_dir, 
                                'tSNE_init.png'), 
                            dpi=800,bbox_inches='tight')
                    figure.clear

                    # Before


                    Y_embedded = TSNE(n_components=2, 
                                    # learning_rate='auto',
                                    init='pca', perplexity=3).fit_transform(Y_emb[:-1,:])
                    # plot and save
                    # palette = sns.color_palette("pastel", n_colors=len(gene_list))  
                    # palette = sns.color_palette("bright", n_colors=len(gene_list))  

                    figure, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,
                                               figsize = (8, 4))
                    # plt.subplots_adjust(hspace=0.3, wspace=0.3)

                    palette = sns.color_palette(cc.glasbey, n_colors=len(gene_list))

                    plot_1 = sns.scatterplot(one_hot_embedded[:,0], one_hot_embedded[:,1], hue=gene_list, 
                                    legend=None, 
                                    palette=palette,
                                    ax= ax1, 
                                    )
                    plot_1.set_title('tSNE of One-hot Label Embedding')
                    

                    
                    plot_2 = sns.scatterplot(Y_embedded[:,0], Y_embedded[:,1], hue=gene_list, 
                                    # legend=None, 
                                    palette=palette,
                                    ax= ax2, 
                                    )                                    
                    plot_2.set_title('tSNE of Linguistic Label Embedding')


                    # plot_2.legend(ncol=10, bbox_to_anchor=(1,-0.1), loc=locs[0])

                    # 左边，右边，顶边，底边
                    # X, Y, width height
                    # ref: https://blog.csdn.net/sinat_41299610/article/details/106494549
                    locs = ['upper right', 'lower left', 'center left', 'lower center', 'center',
                            'right']
                    ax2.legend().remove()
                    lgnd = figure.legend(ncol=10, bbox_to_anchor=(1,0), loc=locs[0],fontsize=7)
                    #change the marker size manually for both lines
                    for i in range(len(lgnd.legendHandles)):
                        lgnd.legendHandles[i]._sizes = [32]
                    # figure.legend(ncol=8, bbox_to_anchor=(1,-0.1), loc=locs[0])
                    figure.tight_layout()
                    # plt.legend(ncol=5, bbox_to_anchor=(1, 1))
                     
                    figure.savefig(os.path.join(heatmap_save_dir, 
                                'tSNE_before.png'), 
                            dpi=800,bbox_inches='tight')
                    figure.clear


                    # After
                    one_hot_embedded = TSNE(n_components=2, 
                                            init='pca', perplexity=3).fit_transform(one_emb_p[:-1,:])

                    Y_embedded = TSNE(n_components=2, 
                                    # learning_rate='auto',
                                    init='pca', perplexity=3).fit_transform(Y_emb_p[:-1,:])
                    figure, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,
                                               figsize = (8, 4))
                    # plot and save
                    # palette = sns.color_palette("pastel", n_colors=len(gene_list))  
                    # palette = sns.color_palette("bright", n_colors=len(gene_list))  

                    plot_1 = sns.scatterplot(one_hot_embedded[:,0], one_hot_embedded[:,1], hue=gene_list, 
                                    legend=None, 
                                    palette=palette,
                                    ax= ax1, 
                                    )
                    plot_1.set_title('tSNE of One-hot Label Embedding')
                    

                    
                    plot_2 = sns.scatterplot(Y_embedded[:,0], Y_embedded[:,1], hue=gene_list, 
                                    # legend=None, 
                                    palette=palette,
                                    ax= ax2, 
                                    )                                    
                    plot_2.set_title('tSNE of Linguistic Label Embedding')


                    # plot_2.legend(ncol=10, bbox_to_anchor=(1,-0.1), loc=locs[0])

                    # 左边，右边，顶边，底边
                    # X, Y, width height
                    # ref: https://blog.csdn.net/sinat_41299610/article/details/106494549
                    locs = ['upper right', 'lower left', 'center left', 'lower center', 'center',
                            'right']
                    ax2.legend().remove()
                    lgnd = figure.legend(ncol=10, bbox_to_anchor=(1,0), loc=locs[0],fontsize=7)
                    #change the marker size manually for both lines
                    for i in range(len(lgnd.legendHandles)):
                        lgnd.legendHandles[i]._sizes = [32]
                    # figure.legend(ncol=8, bbox_to_anchor=(1,-0.1), loc=locs[0])
                    figure.tight_layout()
                    # plt.legend(ncol=5, bbox_to_anchor=(1, 1))
                     
                    figure.savefig(os.path.join(heatmap_save_dir, 
                                'tSNE_after.png'), 
                            dpi=800,bbox_inches='tight')
                    figure.clear

                    # initial
                    # from utils.plot_util import heatmap, annotate_heatmap
                    # data = Y_attn[:-1,:-1]
                    # def normalization(data):
                    #     _range = np.max(data) - np.min(data)
                    #     return (data-np.min(data)) / _range
                    # def normalization(data):
                    #     _range = np.max(data) - np.min(data)
                    #     return (data-np.min(data)) / _range

                    # norm_data = normalization(data)
                    # fig, ax = plt.subplots()
                    # im, cbar = heatmap(norm_data, gene_list, gene_list, ax=ax,
                    #             cmap="YlGn", cbarlabel="Label Attetion Score")
                    # # texts = annotate_heatmap(im, valfmt="{x:.1f} t")

                    # fig.tight_layout()
                    # # plt.show()
                    # fig.savefig(os.path.join(heatmap_save_dir, 
                    #             'label_attn.png'), 
                    #         dpi=800,bbox_inches='tight')
                    # fig.clear

                """ Plot heatmap from label
                """
                # for count, gene in enumerate([gene_list[0]]):
                for count, gene in enumerate(gene_list):
                    # TODO: modify 
                    cur_scores = scores[count,:]
                    tag = 'label_{}_pred_{}'.format(Y_label[count], Y_hat[count])

                    if lvl == 'macro':
                        sample_save_dir = os.path.join(slide_save_dir, 'gene_{}'.format(gene))
                        os.makedirs(sample_save_dir, exist_ok=True)
                        print('sampling {} score'.format(gene))

                        # selection range of socre is (score_start,score_end]
                        sample_results = sample_rois(cur_scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                                score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))

                        # samples
                        # for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                        #         print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                        #         patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                                
                        #         # TODO: resize
                        #         newsize = custom_roi_size
                        #         patch = patch.resize(newsize)
                        #         patch.save(os.path.join(sample_save_dir, '{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, s_coord[0], s_coord[1], s_score)))


                    
                        # # save heat map
                        # wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
                        # 'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

                        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
                        if os.path.isfile(os.path.join(sample_save_dir, heatmap_save_name)):
                            pass
                        else:
                            heatmap = drawHeatmap(cur_scores, coords, slide_path, wsi_object=wsi_object, 
                                            cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, 
                                            use_holes=True, binarize=False, vis_level=-3, blank_canvas=False,
                                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
                            # heatmap = drawHeatmap(cur_scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                                            # thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
                        
                            heatmap.save(os.path.join(attn_save_dir, '{}_{}_blockmap.png'.format(gene,tag)))
                            del heatmap

                    if lvl == 'micro':
                        """ file structure
                                -gene
                                    -label1
                                        - pred 0
                                            -slide_id
                                        - pred 1
                                    -label0
                        """

                        # first lvl path
                        mic_path = os.path.join(vis_path, gene)
                        os.makedirs(mic_path, exist_ok=True)
                        mic_path = os.path.join(mic_path, 'label_{}'.format(Y_label[count]))
                        os.makedirs(mic_path, exist_ok=True)
                        mic_path = os.path.join(mic_path, 'pred_{}'.format(Y_hat[count]))
                        os.makedirs(mic_path, exist_ok=True)
                        sample_save_dir = os.path.join(mic_path, slide_id)
                        os.makedirs(sample_save_dir, exist_ok=True)

                        sample['k'] = 5

                        print('sampling {} score'.format(gene))
                        # selection range of socre is (score_start,score_end]
                        sample_results = sample_rois(cur_scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                                score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))

                        for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                                print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                                patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                                
                                # TODO: resize
                                newsize = custom_roi_size
                                patch = patch.resize(newsize)
                                patch.save(os.path.join(sample_save_dir, '{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, s_coord[0], s_coord[1], s_score)))


                """
                    process: whether to calculate heatmap with specified overlap 
                """
                # save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

                # if heatmap_args.use_ref_scores:
                #     ref_scores = scores
                # else:
                #     ref_scores = None

                # if heatmap_args.calc_heatmap:
                #     compute_from_patches_for_ML(args=args,wsi_object=wsi_object,
                #                                         model=model,
                #                                         feature_path=features_path,
                #                                             **wsi_kwargs,
                #                                             attn_save_path=save_path,  
                #                                             ref_scores=ref_scores,
                #                                             label_pred=count,)
                # if not os.path.isfile(save_path):
                #     print('heatmap {} not found'.format(save_path))
                #     if heatmap_args.use_roi:
                #         save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                #         print('found heatmap for whole slide')
                #         save_path = save_path_full
                #     else:
                #         continue
                    
                # file = h5py.File(save_path, 'r')
                # dset = file['attention_scores']
                # coord_dset = file['coords']
                # scores = dset[:]
                # coords = coord_dset[:]
                # file.close()

                # for count, gene in enumerate(gene_list):
                #     # visulize mapping
                #     heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
                #     if heatmap_args.use_ref_scores:
                #         heatmap_vis_args['convert_to_percentiles'] = False

                #     heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(gene, float(patch_args.overlap), int(heatmap_args.use_roi),
                #                                                                                 int(heatmap_args.blur), 
                #                                                                                 int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                #                                                                                 float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                #                                                                                 int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)

                #     if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                #         pass
                    
                #     else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                #         heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
                #                                 cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                #                                 binarize=heatmap_args.binarize, 
                #                                 blank_canvas=heatmap_args.blank_canvas,
                #                                 thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                #                                 overlap=patch_args.overlap, 
                #                                 top_left=top_left, bot_right = bot_right)
                        
                #         if heatmap_args.save_ext == 'jpg':
                #             heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                #         else:
                #             heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

                #     # save
                #     if heatmap_args.save_orig:
                #         if heatmap_args.vis_level >= 0:
                #             vis_level = heatmap_args.vis_level
                #         else:
                #             vis_level = vis_params['vis_level']
                #         heatmap_save_name = '{}_orig_{}.{}'.format(gene,int(vis_level), heatmap_args.save_ext)
                #         if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                #             pass
                #         else:
                #             heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                #             if heatmap_args.save_ext == 'jpg':
                #                 heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                #             else:
                #                 heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))            