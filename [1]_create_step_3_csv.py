"""Load SNV label and inter_gene_list to construct gene list
    Convert them as Step 3.csv, gene_dict

    Inputs:
        args.save_path: save path of Step_3.csv
        data_path: path of dataset processed by 
        args.label_path: path of SNV label

        args.r: the specific args.region_list for dataset generation, if 'None', process the whole args.region_list list 

    Outputs:
        Step_3.csv
        Step_3_.csv
        gene_dict_.json
"""
import os
import csv
import json
import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import argparse
from utils.file_utils import get_args_parser_for_dataset, param_selection_for_dataset
parser = argparse.ArgumentParser('main', parents=[get_args_parser_for_dataset()])
args = parser.parse_args()


if len(np.shape(args.mix_list)) > 1:
    args.mix_list = list(np.squeeze(args.mix_list))

args.count = 0

# args.label_space = 'react'
# args.label_space = 'oncokb'

args.if_step_3 = True
args.if_gene_dict = False

args.add_region_column = True
args.if_str_to_ind = True

def list_txt(path, list=None):
    '''
    '''
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

""" part 1: change the columns of gene label
        case_id slide_id args.region_list gene1 gene2 ... --> 
        case_id slide_id args.region_list total_gene_list
"""
def add_gene_column(total_df=None,total_gene_list=None):
    if args.specific_region == False:

        """ choose intersection gene
        """        
        inter_gene_ind_set = []

        if args.mix_list is not None:
            cur_region_list = []
            for j in [int(i) for i in args.mix_list]:
                cur_region_list.append(args.region_list[j])
        else:
            cur_region_list = args.region_list

        for n, ind in enumerate(cur_region_list):
            total_df_copy = total_df.copy()
            cur_df = pd.read_csv(args.label_path + cur_region_list[n] + '.csv',index_col=0) # current dataframe
            pd_gene_label = cur_df.iloc[:,args.start_ind:]        
            cur_gene_list = pd_gene_label.columns.values.tolist()  #current gene list

            cur_gene_list_ = list(set(cur_gene_list).intersection(total_gene_list))   # select current intersection gene list
            
            inter_gene_ind = [total_gene_list.index(i) for i in cur_gene_list_]        


            inter_gene_ind_set.append(inter_gene_ind)
            for i, item in enumerate(inter_gene_ind):
                idx = cur_gene_list.index(cur_gene_list_[i]) 
                total_df_copy.iloc[:,item] = pd_gene_label.iloc[:,idx]
            # concatenate
            df = pd.concat([cur_df.iloc[:,0:args.start_ind], total_df_copy],axis=1)

            if n == 0 :
                df_copy = df.copy()
            else:
                df_copy = pd.concat([df_copy, df],axis=0)        

        df = df_copy
    
    else:
        cur_df = pd.read_csv(args.label_path+args.region_list[args.r]+'.csv',index_col=0) # current dataframe
        pd_gene_label = cur_df.iloc[:,args.start_ind:] # fetch corresponding gene labels


        if args.inter == True:
            # if construct intersection(mutual) label space for each dataset
            # intersection list
            cur_gene_list = pd_gene_label.columns.values.tolist() #current gene list
            cur_gene_list_ = list(set(cur_gene_list).intersection(total_gene_list)) # select current intersection gene list
            inter_gene_ind = [total_gene_list.index(i) for i in cur_gene_list_]

            # TODO: modify the code since 
            # giving the value
            for i, item in enumerate(inter_gene_ind):
                idx = cur_gene_list.index(cur_gene_list_[i]) 
                total_df_copy.iloc[:,item] = pd_gene_label.iloc[:,idx]
            # concatenate
            df = pd.concat([cur_df.iloc[:,0:args.start_ind], total_df],axis=1)

        else:
            df = cur_df
    
    return df



def change_bar_code(df, patch_path):
    slide_id_set = os.listdir(patch_path+'/patches/')
    for i in range(len(df)):
        for idx, slide_id in enumerate(slide_id_set):
            if slide_id[:12]==df.iloc[i,0]:
                slide_name = slide_id[:-3]
                slide_id_set.pop(idx)
                # [slide_id for slide_id in slide_id_set if slide_id[:12]==df.iloc[i,0]]
                df.iloc[i,0] = slide_name  # drop out .h5
    return df

def pipe_line(args):
    num = []
    if args.data == 'tcga':
        total_gene_list = list_txt(path=args.list_path + '/{}_{}_inter_gene_list.txt'.format(args.data, args.label_space)) # load txt
    else:
        total_gene_list = list_txt(path=args.list_path + '/{}_inter_gene_list.txt'.format(args.data))

    total_df = pd.DataFrame(columns=total_gene_list) # creat null dataframe


    if args.add_region_column:
        """ insert args.region_list column for SNV label
        """
        for i in range(len(args.region_list)):
            df = pd.read_csv(args.label_path+args.region_list[i]+'.csv',index_col=0)
            # df = pd.DataFrame(df['slide_id'].unique())
            if 'TCGA_barcode' in df.columns.values.tolist():
                df.rename(columns={'TCGA_barcode':'slide_id'}, inplace = True)

            df.drop_duplicates(subset=['slide_id'], keep='first', inplace=True)
            # df = df.iloc[:,1:-1]
            col_name = df.columns.tolist()
            if 'region' in col_name:
               pass 
            else:
                col_name.insert(col_name.index('slide_id')+1,'region')
                df.loc[:,'region'] = args.region_list[i]
                df = df.reindex(columns=col_name)
                df.to_csv(args.label_path+args.region_list[i]+'.csv',index=True)
            num_ = len(df['slide_id'].values)
            num.append(num_)
            print('The number of gene-containing-WSIs in region '+ args.region_list[i] +' is ', num_)
    print('The total number of gene-containing-WSIs is ', np.sum(num))


    """ part 2: choose mutual slides 
    """

    
    if args.if_step_3 == True:
        df = add_gene_column(total_df=total_df,total_gene_list=total_gene_list)
        
        #TODO: select the WSI which has patches
        ids1 = [i for i in df.slide_id]
        ids1 = list(set(ids1))
        
        # TODO: form different data_root_dir
        if args.mix_list is not None:
            patch_dir = []
            ids2 = []
            start = 0 if args.data=='cptac' else 5
            for k in args.mix_list:
                cur_patch_dir = os.path.join(args.data_root, args.region_list[int(k)][start:] + args.add_path)
                patch_dir.append(cur_patch_dir)
                if args.data == 'cptac':
                    ids2.extend([i[:-3] for i in os.listdir(cur_patch_dir + '/patches/')])   #[:-3] 指的是去除 .h5 后缀
                elif args.data == 'tcga':
                    ids2.extend([i[:12] for i in os.listdir(cur_patch_dir + '/patches/')]) 

        else:
            if args.data == 'cptac':
                ids2 = [i[:-3] for i in os.listdir(args.data_root_dir+'/patches/')]  #[:-3] 指的是去除 .h5 后缀
            elif args.data == 'tcga':
                ids2 = [i[:12] for i in os.listdir(args.data_root_dir+'/patches/')]
 
        ids2 = list(set(ids2).intersection(set(ids1)))       
        ids = df['slide_id'].isin(ids2)  # choose mutual data


        df = df.loc[ids] #

        if args.data == 'tcga':
            # TODO: 4/24 exchange barcode to slide id 
            if args.mix_list is not None :
                for k in range(len(args.mix_list)):
                    df = change_bar_code(df, patch_dir[k])
            else:
                df = change_bar_code(df, args.data_root_dir)


        if args.mix_list is not None :
            # label space which only exists intetsection.
            intersect_df = df.dropna(axis=1, how='any')
            inter_list = intersect_df.columns.values.tolist()[args.start_ind:]
            print('the number of intersected label space is {}'.format(len(inter_list)))
            print('the gene is: {}'.format(inter_list))
            # mutual label space
            df = df.dropna(axis=1, how='all') # drop columns which contains all Nan values     
            mutual_list = df.columns.values.tolist()[args.start_ind:]
            print('the number of intersected label space is {}'.format(len(mutual_list)))
            print('the gene is: {}'.format(mutual_list))
        #
        df = df.fillna(0) # fill nan value to zero
        len1 = len(df.columns.values.tolist())
        # drop all zero columns
        df = df.loc[:, (df != 0).any(axis=0)]
        len2 = len(df.columns.values.tolist())
        if len1 != len2:
            print('exsit non-mutation gene')

        # df = pd.DataFrame(df['slide_id'].unique())
        df.drop_duplicates(subset=['slide_id'], keep='first', inplace=True)
        print('the number of sample is ', str(len(df)))
        print('-----------------------')

        df.to_csv(args.csv_1_path,index=False)


    """final form:
        case_id slide_id region gene1 gene2 gene3 ...
        xxx    xxx       BR      0      0     0
        xxx    xxx       GBM     ..    ...     ...
    """

    if args.if_str_to_ind and args.if_step_3:
        """ change args.region_list dictionary to index numbers 
                form:  case_id slide_id region gene1 gene2 gene3 ...
                        xxx     xxxx      0      0     0    0    ...
                        xxx     xxxx      1      0     0    0    ... 
        """
        pd_csv = pd.read_csv(args.csv_1_path)

        for i, name in enumerate(args.region_list):
            ind = pd_csv.index[pd_csv['region'] == name].tolist()
            # ind = pd_csv.columns['region']   
            pd_csv.loc[ind,'region'] = i

        pd_csv.to_csv(args.csv_2_path,index=False)   


    if args.if_str_to_ind and args.if_gene_dict:
        """ Further step: form the gene dict index for different args.region_list index's label
                form: {0}: ['gene1','gene2',...]
                      {1}: ['gene1','gene2',...]
                    ...
        """
        gene_dict = {}
        # total_gene_list = list_txt(path=args.list_path + '/inter_gene_list.txt') # load txt
        if args.data == 'tcga':
            total_gene_list = list_txt(path=args.list_path + '/{}_{}_inter_gene_list.txt'.format(args.data, args.label_space)) # load txt
            gene_dict_path = '/{}_{}_gene_dict_.json'.format(args.data, args.label_space)
        else:
            total_gene_list = list_txt(path=args.list_path + '/{}_inter_gene_list.txt'.format(args.data))
            gene_dict_path = '/{}_{}_gene_dict_.json'.format(args.data)
        for i, name in enumerate(args.region_list):     
            cur_df = pd.read_csv(args.label_path + name + '.csv',index_col=0) # current dataframe
            pd_gene_label = cur_df.iloc[:,3:] # fetch corresponding gene labels
            cur_gene_list = pd_gene_label.columns.values.tolist() #current gene list
            cur_gene_list_ = list(set(cur_gene_list).intersection(total_gene_list)) # select current intersection gene list
            inter_gene_ind = [total_gene_list.index(i) for i in cur_gene_list_]
            gene_dict[i] = inter_gene_ind
            print('The number of gene-containing-WSIs label in args.region_list '+ str(i) +' is ', len(gene_dict[i]))
        print('The total number of gene-containing-WSIs label is ', len(total_gene_list))
        import json
        with open(args.list_path + gene_dict_path, 'w') as fp:
            json.dump(gene_dict, fp, sort_keys=True, indent=4)


itera = args.iter
if itera == True:
    args.data = 'tcga'
    # for i in ['pancan', 'persample']:
    for i in ['pancan']:
        # for j in ['freq', 'driven']: 
        for j in ['driven']:               
            label_source = i
            label_tag = j
            param_selection_for_dataset(args.data, label_source, label_tag, args)
            pipe_line(args)
else:
    label_source = args.s
    label_tag = args.atr
    param_selection_for_dataset(args.data, label_source, label_tag, args)
    pipe_line(args)