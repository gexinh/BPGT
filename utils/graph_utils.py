import numpy as np
""" create mutual information for gene encoder
"""


def node_info(args, df):
    node_list = df.iloc[:,args.start_ind:].columns.values.tolist()
    organs = df.iloc[:,args.start_ind-1].tolist()
    unique, counts = np.unique(organs, return_counts=True)
    organ_num = len(unique)

    # get unique regions regarding to different nodes 
    node_unique_dict = {}
    node_unique_arr = []

    for i, node in enumerate(node_list):    
        row = np.zeros(shape=(organ_num))
        idx = df[node]==1
        process_df = df.loc[idx]
        region = process_df.iloc[:,args.start_ind-1].tolist()
        unique, counts = np.unique(region, return_counts=True)
    
        node_unique_dict[node] = unique
    
        row[unique] = 1
        node_unique_arr.append(row)
    node_unique_arr = np.array(node_unique_arr)

    return node_list, node_unique_dict, node_unique_arr

def mutual_info_encoding(args, df, if_cls_token=False):
    node_list, node_unique_dict, node_unique_arr = node_info(args, df)
    organ_num = np.shape(node_unique_arr)[-1]
    node_num =  np.shape(node_unique_arr)[0] + if_cls_token

    mutual_mask = np.zeros(shape=(node_num, node_num))
    # calculate mutual infor
    for i in range(0, node_num):
        for j in range(i+1, node_num):
            if i == 0 and if_cls_token:
                v_ij = organ_num
            else: 
                node_i = node_list[i-if_cls_token]
                node_j = node_list[j-if_cls_token]
                v_ij = len(set.intersection(set(node_unique_dict[node_i]), set(node_unique_dict[node_j])))
            mutual_mask[i, j] = v_ij
    mutual_mask_tran = np.transpose(mutual_mask)
    mutual_mask = mutual_mask + mutual_mask_tran 
    # norm
    mutual_mask = mutual_mask/organ_num

    # if args.indent:
    mutual_mask = mutual_mask + np.eye(node_num)  # provide diagonal elements with max values


    return mutual_mask

""" create phenotype encoding as potional encoding
"""
def organ_positional_encoding(args, df):
    node_list, node_unique_dict, node_unique_arr = node_info(args, df)
    organ_num = np.shape(node_unique_arr)[-1]

    P = np.eye(organ_num)
    M = node_unique_arr
    PE = np.matmul(M, P)

    return PE 