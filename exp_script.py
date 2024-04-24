""" 
    implement the script according to different visual extractors

    variables:
        --model
        --mix_list
        --graph_module
    usages:
        terminal -> nohup python para_search_exp.py --gpu_index 5 &
"""
from cmath import log
import os
import subprocess
import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["OMP_NUM_THREADS"] = "6"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "6"  # 设置MKL-DNN CPU加速库的线程数。
import torch
torch.set_num_threads(6)

from torch import log
from utils.general_utils import Params,mkdir
from utils.file_utils import param_selection

import pandas as pd
import re

parser = argparse.ArgumentParser()

parser.add_argument('--if_ddp', 
                    default=True,
                    # default=False, 
                    help='if use the DDP, it will overide the gpu_index in main.py')

parser.add_argument('--gpu_index',default=7,type=int, help='determine the mode of the experiment')
parser.add_argument('--gpu_workers',default=4,type=int, help='the total number of gpu workers')

parser.add_argument('--mix_list', type=list, default=None, nargs='+', help='1234 -> [1,2,3,4]')
# parser.add_argument('--downstream_model', type=str, default='ml_decoder', choice=['mlp', 'ml_decoder'],help='the downstream model')

parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train', 
                        help='the mode of implementing models')

parser.add_argument('--label_category', type=str, 
                    default='oncokb',
                    choices=['cancer_driven','oncokb','subtype'])

parser.add_argument('--backbone', type=str, choices=['deephis', 'geneh', 'sa_mil','hipt','trans_mil','kat'], 
                    default='trans_mil',
                    help='exp model')

parser.add_argument('--graph_module', type=str, choices=['gcn', 'transformer','graphormer', 'geneformer_v2','mlp','mcat'], 
                    default='geneformer', 
                    help='the approach of using graph convolutional network')    

parser.add_argument('--word_embedd_model', default='biobert', choices=['biobert','clinical_bert','pubmed_bert','scibert','none'], help='Choose Bert ')

def caLL_script_with_args(path, params,i):
    mkdir(path)

    # make dictory
    save_path = os.path.join(path,'exp_{}'.format(i))
    os.makedirs(save_path, exist_ok=True)   

    save_path = os.path.join(save_path,'params.json').replace('\\','/')
    if params.mode == 'train':
        log_path = os.path.join(path, '{}_logs'.format(str(i))).replace('\\','/')
    elif params.mode == 'infer':
        log_path = os.path.join(path, '{}_logs_infer'.format(str(i))).replace('\\','/')

    off_set = len(params.graph_module) * 2  if params.graph_module!='gcn' else 60

    skip_flag = False
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f: 
            lines = f.readlines()  # read all lines
            if not lines :
                print('no content in logs')
                
            else:
                last_line = lines[-1]  # read last line
                if last_line == 'end script\n' or (last_line == '*****************************************\n' and len(lines)>5):
                    skip_flag = True

    if skip_flag == False:
        params.save(save_path)
        
        if not params.if_ddp:
            subprocess_param = ["python","[3]_main.py","--json_dir",path,
                                "--mag",str(params.mag),
                                "--patch_size",str(params.patch_size),
                                "--exp_count", str(i),
                                "--gpu", str(params.gpu),
                                ]
        else:             
            subprocess_param = ["python", "-m", "torch.distributed.launch", 
                                "--nproc_per_node=4", 
                                "--master_port", str(27519+off_set), 
                                "[3]_main.py", "--json_dir",path,
                                "--mag",str(params.mag),
                                "--patch_size",str(params.patch_size),
                                "--exp_count", str(i),
                                ]



        # call the main
        with open(log_path,'w') as f:
            completed = subprocess.run(subprocess_param, stdout=f,shell=False)

    # save 
    with open(log_path, 'r', encoding='utf-8') as f: 
        lines = f.readlines()  # read all lines
        last_line = lines[-1]  # read last line
        # TODO: this part should modified via your log
        if last_line == 'end script\n' or lines[-1]== '*****************************************\n':
            if lines[-1] == '*****************************************\n':
               shift_num = 3 
            else:
                shift_num = 0
            #TODO: modify at 5/8
            if params.model_type == 'att_ml':
                train_line = lines[-(8+shift_num)]
                val_line = lines[-(6+shift_num)]
                test_line = lines[-(4+shift_num)]
                train_auc = re.split('\[|\]',train_line)[1]
                val_auc = re.split('\[|\]',val_line)[1]
                test_auc = re.split('\[|\]',test_line)[1]

                train_acc = round(1 -float( re.split(':|\,| ',train_line) [3]), 4)  # ['Val', 'error', '', '0.3793', '', 'ROC', 'AUC', '', '0.6550']
                val_acc = round(1 -float( re.split(':|\,| ',val_line) [3]), 4)  # ['Val', 'error', '', '0.3793', '', 'ROC', 'AUC', '', '0.6550']
                test_acc = round(1 -float( re.split(':|\,| ',test_line) [3]), 4)
                data_number = []
                for i in range(3):            
                    ratio = '/'
                    data_number.extend([ratio])
            else:
                train_line = lines[-8]
                val_line = lines[-7]
                test_line = lines[-6]
                train_auc = round(float( re.split(':|\,| ',train_line) [-1]), 4)
                val_auc = round(float( re.split(':|\,| ',val_line) [-1]), 4)
                test_auc = round(float( re.split(':|\,| ',test_line) [-1]), 4)

                train_acc = round(1 -float( re.split(':|\,| ',train_line) [3]), 4)  # ['Val', 'error', '', '0.3793', '', 'ROC', 'AUC', '', '0.6550']
                val_acc = round(1 -float( re.split(':|\,| ',val_line) [3]), 4)  # ['Val', 'error', '', '0.3793', '', 'ROC', 'AUC', '', '0.6550']
                test_acc = round(1 -float( re.split(':|\,| ',test_line) [3]), 4)
                data_number = []
                for i in range(3):
                    data_line = lines[22+i] #Training set is: [{}/{}]
                    ratio = re.split('\[|\]',data_line)[1]
                    data_number.extend([ratio])
            
            result = [train_acc, val_acc, test_acc, train_auc, val_auc, test_auc]

            result.extend(data_number)
    return result

# load based params
json_path = os.path.join('./experiments/base_model','params.json').replace('\\', '/')# based params
params = Params(json_path)

# we can modify default settings
params.dict['data'] = 'tcga'
if inter_region == True:
    params.dict['specific_region'] = False
    params.dict['mix_list'] = '0123456789'
    params.dict['mix_list'] = '012345678'
    params.dict['if_select'] = False
else:
    params.dict['specific_region'] = True
    params.dict['mix_list'] = None
    params.dict['region_ind'] = 0 #brca
    params.dict['if_select'] = False
    # params.dict['region_ind'] = 2 #luad


params.dict['max_epochs'] = 200

# magnification and patch
params.dict['mag'] = 20
params.dict['patch_size'] = 256

params.dict['reduce_patch'] = False
params.dict['lr'] = 1e-4 

params.dict['label_category'] = 'oncokb' # subtype or gene
params.dict['downstream_model'] = 'exp_pipe'

params.dict['decoder_layer'] = 1
params.dict['encoder_layer'] = 2


params.dict['region_cls_model'] = 'wsl' # wsl, corr, and none
params.dict['region_cls'] = False
params.dict['weight_adj'] = True

params.dict['num_graph_layer'] = 2
params.dict['cls_token'] = False        

params.dict['word_embedd_model'] = 'biobert'



import copy
default_params = copy.deepcopy(params)


""" Script with Argparser to implement scrpits with different modes
"""
if __name__ =='__main__':
    print(os.getpid())
    args = parser.parse_args()

    if args.traverse == True:
        args.if_ddp = False
    params.if_ddp = args.if_ddp
    params.gpu = args.gpu_index
    params.label_category = args.label_category
    params.model = args.backbone
    params.mode = args.mode

    args.label_ind = 0 #

    for j, k in enumerate(params.__dict__):
        v = params.__dict__[k]
        args.__dict__[k] = v         


    # TODO: give setting values
    gene_list, args = param_selection(args)
    params.dict['region'] = args.region
    params.dict['label_ind'] = args.label_ind



    default_params = copy.deepcopy(params)
    params_dict = {'graph_module': [args.graph_module]}
    if args.traverse == False:
        count = 0
        for i, (param, setting_list) in enumerate(params_dict.items()):
            for val in setting_list:
                #reset
                params = copy.deepcopy(default_params)
                
                params.dict[param] = val

                for j, k in enumerate(params.__dict__):
                    v = params.__dict__[k]
                    args.__dict__[k] = v
                gene_list, args = param_selection(args)

                # create a pandas
                summary_path =  os.path.join(args.results_dir, 'summary.csv'.format(args.model_type, args.data, args.region)) 
                start_train_flag = False

                if not os.path.exists(summary_path):
                    df = pd.DataFrame(columns=['data','label','cancer','gene space', 'backbone', 
                                    'hyper-parameters',
                                    'train_PR', 'val_PR', 'test_PR','train_f1', 'val_f1', 'test_f1o', 
                                    'train','val','test',
                                    ])
                else:
                    df = pd.read_csv(summary_path, index_col=0)
                start_point = len(df)

                count = start_point
                cur_ind = count
                # cur_ind = start_point-1 + count
                col = df.columns.values.tolist()
                new_row = {key:0 for key in col}
                df = df.append(new_row, ignore_index=True)

                df.iloc[cur_ind, 0:5] = [args.data, args.s+'_'+args.atr, args.region, len(gene_list), args.model_type]
                df.iloc[cur_ind, 5] = param + ': ' + str(val)        

                path = args.results_dir
                result = caLL_script_with_args(path, params, count)
 
                df.iloc[cur_ind, 6:] = result
                df.to_csv(summary_path)