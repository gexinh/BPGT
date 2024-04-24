import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_args_parser():
    # parse parameters
    parser = argparse.ArgumentParser(description='Configurations for WSI Training', add_help=False)

    # TODO: script related settings
    parser.add_argument('--exp_name', type=str, help='experiment code for saving results',default= 'gene_')
    parser.add_argument('--task', type=str, choices=['cptac',  'tcga'], default='cptac') # control the split dir
    # parser.add_argument('--gpu', nargs='+', type=int,default=[1,2,3], 
    #                     help='data directory')
    parser.add_argument('--gpu', type=int,default=4, 
                        help='data directory')
    parser.add_argument('--label_ind', type=int, default=20,
                        help='the index of gene to binary classification, cptac: maximun/minimun = 288/112, 285:118')
    parser.add_argument('--label_category', type=str, choices=['gene', 'subtype','oncokb'], default='gene',
                        help='the selection of label space, including gene mutation, subtyping, and so on.')
    parser.add_argument('--if_select', type=int, default=0,
                        help='if specifically choose gene labels')
    parser.add_argument('--exp_count',type=int,default=0,
                        help='the count number of the current experiment'
                        )
    # TODO: data selection
    parser.add_argument('--data', default='tcga', choices=['tcga', 'cptac', 'no'], help='choose the dataset')
    parser.add_argument('--s', default='pancan', choices=['pancan', 'persample'], help='choose the source of tcga dataset label')
    parser.add_argument('--atr', default='driven', choices=['freq', 'driven'], help='choose the attribution of tcga dataset')

    parser.add_argument('--region_ind', type=int, default=None, help='choose the specific region to construct label space')
    parser.add_argument('--specific_region', type = int, default=True,
					help='if process the specific region for tcga/cptac dataset')
    parser.add_argument('--mix_list', type=list, default=None, nargs='+', help='1234 -> [1,2,3,4]')

    # magnification and patch size
    parser.add_argument('--mag', type = int, default=-1, choices=[-1, 40, 20, 10, 5], 
                help='choose the magnification of slides')
    parser.add_argument('--patch_size', type=int, default=256, help='the original patch size of WSI')

    #TODO: data path related
    parser.add_argument('--data_feature_dir', type=str, default='/', 
                        help='data feature directory')
    parser.add_argument('--data_root_dir', type=str, default='./data/cptac/', 
                        help='data directory')
    parser.add_argument('--results_dir', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, 
                        help='manually specify the set of splits to use, ' 
                        +'instead of infering from the task and label_frac argument (default: None)')
    parser.add_argument('--json_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
    # TODO: model selection
    parser.add_argument('--learnable_embedd', default=0,type=int,help='if is learnable')
    parser.add_argument('--word_embedd_model', default='biobert', choices=['biobert','clinical_bert','pubmed_bert','scibert'], help='Choose Bert ')
    parser.add_argument('--model', type=str, choices=['deephis-ve', 'genehe-ve', 'att_mil', 'sa_mil','hipt','trans_mil','kat'], default='trans_mil', 
                        help='exp model')

    parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train', 
                        help='the mode of implementing models')

    parser.add_argument('--pretrained_model', choices = ['dino', 'resnet', 'incep','hipt'], default='dino', help='choose pretrained model')
    parser.add_argument('--downstream_model', choices = ['attention', 'wsl', 'mlp', 'ml_decoder','geneh','deephis'], default='wsl', 
                        help='choose the downstream model, including attention or weakly supervised learnig for binary classification and MLP or multi-label decoder for multi-label classification')
    parser.add_argument('--agg', choices = ['att', 'mean', 'max'], default='att', help='choose the method of patch aggregation method for binary classification')

    parser.add_argument('--model_size', type=str, choices=['1024_small', '1024_big','1024_min', 'dino', 'dino_b','dino_s','dino_ss', 'hipt'], default='dino_b', help='size of model, does not affect mil')
    parser.add_argument('--model_type', type=str, choices=['multi_label', 'binary_cls', 'clam', 'att-gate','att_ml'], default='att_ml', 
                        help='type of model (default: binary classification, multi-label classification, or multi-instance learning)')
    parser.add_argument('--loss', type=str, choices=['focal', 'ce', 'cb'], default='ce', help='model loss')
        #TODO: for multi-label task
    parser.add_argument('--graph_module', type=str, choices=['gcn','graph_transformer', 'bpgt','mlp','mcat'], default='gcn', help='the approach of using graph convolutional network')    
    parser.add_argument('--agg_method', choices = ['att', 'trans_mil','kat', 'hipt','sa_mil'], default='att', help='choose the method of patch aggregation method for ML decoder')
    parser.add_argument('--decoder_layer', type=int, default=1, help='number of multi-label decoder`s layer (default: 1)')
    parser.add_argument('--encoder_layer', type=int, default=1, help='number of transformer encoder`s layer (default: 1)')

    parser.add_argument('--region_cls', type=int, default=True, help='if add additional task: multi-class classfication for WSI`s region')
    parser.add_argument('--region_cls_model', choices = ['wsl', 'corr', 'none'], default='wsl', help='the module of region classification')
    parser.add_argument('--cls_token', type=int, default=True, help='if use class token for label embedding to classify')

    # TODO: training settings
    parser.add_argument('--reduce_patch', action='store_true', default=True, help='enabel reduce_patch')
    parser.add_argument('--resample', action='store_true', default=True, help='enabel dataset resample')
    parser.add_argument('--gamma', type=float, default=2.0, help='hyper-parameter weight of CB loss')
    parser.add_argument('--k_sample', type=int, default=8, help='number of sample to calculate cluster')
    parser.add_argument('--weight', type=float, default=0.5, help='loss weight bewteen Cluster and CE loss')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='L2 weight decay (default: 1e-5)')
    parser.add_argument('--dropout_rate', type=float, default=0.25,
                        help='L2 weight decay (default: 1e-5)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--l1_norm', default=True,
                        help='the index whether use the L1 regularization for weights')                    
    parser.add_argument('--clsw', default=True,
                        help='the index whether use the class weight for binary classification')
                    
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='maximum number of epochs to train (default: 200)')


    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')

    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes (default: 2)')

    parser.add_argument('--k', type=int, default=1, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')

    parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
    parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
    parser.add_argument('--drop_out', action='store_true', default=True, help='enabel dropout (p=0.25)')


    # TODO: distributed data parallel (DDP) related
    parser.add_argument('--if_ddp', default=False, help='if use the DDP')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", choices=["env://", "tcp://localhost:23456" ],type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser
parser = argparse.ArgumentParser('main', parents=[get_args_parser()])
args = parser.parse_args()

if len(np.shape(args.mix_list)) > 1:
    args.mix_list = list(np.squeeze(args.mix_list))


""" Obtain hyper-parameters
"""
from utils.general_utils import Params,mkdir
json_path = os.path.join(args.json_dir, 'exp_{}'.format(args.exp_count),'params.json').replace('\\', '/') 
assert os.path.isfile(
    json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

""" Obtain args
"""
for j, k in enumerate(params.__dict__):
    v = params.__dict__[k]
    args.__dict__[k] = v     

args.mix_list = list(params.mix_list) if params.mix_list is not None else params.mix_list

if not args.if_ddp:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    os.environ["OMP_NUM_THREADS"] = "6"  
    os.environ["MKL_NUM_THREADS"] = "6"  
    import torch
    torch.set_num_threads(6)

    from utils.utils_dino import init_distributed_mode
    init_distributed_mode(args)


import numpy as np
import torch
from data.data_loader import Generic_ML_Dataset, Generic_ML_Table, Generic_Split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from utils.core_utils import train
import logging
from utils.file_utils import save_pkl, load_pkl

# Train part
import torch.optim as optim

def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start

    end = args.k_end


    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(split_bool_path='{}/splits_{}.csv'.format(args.split_dir, i))  
        datasets = (train_dataset, val_dataset, test_dataset)

        # train
        test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)

        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        # save_pkl(filename, results)


    final_df = pd.DataFrame({'folds': folds, 'test_f1': all_test_auc, 
        'val_f1 ': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=10):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)



from utils.file_utils import param_selection
_, args = param_selection(args)

""" load gene list
"""
# gene list
graph_dir = './data/graph'
if args.data == 'tcga':
    graph_atr = args.region+'_'+args.s+'_'+args.atr if args.label_category =='react' else args.region + '_oncokb'
else:
    graph_atr = args.region

select_gene = pd.read_csv(graph_dir + graph_atr +'_gene_list.csv',index_col=0).to_numpy().squeeze().tolist()


""" preprocess for knowledge graph matrixs
"""
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
from utils.utils import normalize
import scipy.sparse as sp
# add self loop for the zero entry on the diagnal line
adj_mx = normalize(adj_mx + np.diag(np.max(adj_mx,axis=0).tolist()[0]) + sp.eye(adj_mx.shape[0]))

# to tensor
adj_mx = torch.tensor(adj_mx,dtype=torch.float32)
# save in args
args.process_gene_list = select_gene
args.adj = adj_mx
        

""" load datasets
"""
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_name,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            "use_drop_out": args.drop_out,
            'opt': args.opt}

print('\nLoad Dataset')
if args.model_type !='att_ml':
    dataset = Generic_ML_Dataset(csv_path = args.csv_path,
                                dataset = args.data,
                                data_dir= os.path.join(args.data_feature_dir),
                                shuffle = False, 
                                seed = args.seed,
                                region_list=args.region_list,
                                )
else:
    dataset = Generic_ML_Dataset(csv_path = modify_path,
                                dataset = args.data,
                                data_dir= os.path.join(args.data_feature_dir),
                                shuffle = False, 
                                seed = args.seed,
                                region_list=args.region_list,
                                )

""" processed data list
"""
args.list_dir = './data/processed data'


""" load word embedding  
"""
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

""" load gene index dict, which is used for gene-wise metric evaluation.
"""
import json
with open(args.list_dir+'/list/gene_dict_.json', 'r') as fp:
   args.gene_dict = json.load(fp)


""" data path 
"""
# create the folder
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)
# result dir: experiments/experiment_names/exp_i/seed_i

args.results_dir = os.path.join(args.results_dir, 'exp_{}'.format(args.exp_count))
args.results_dir = os.path.join(args.results_dir, 'seed_{}'.format(args.seed))

os.makedirs(args.results_dir, exist_ok=True)


# split dir
args.split_dir = os.path.join(args.data_root_dir + args.add_path+'/splits', args.task+'_{}'.format(int(args.label_frac*100)))
print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})


# save setting 
with open(args.results_dir + '/experiment.txt', 'w') as f:
    print(settings, file=f)
f.close()

# print setting
print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

# call main() function
if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")