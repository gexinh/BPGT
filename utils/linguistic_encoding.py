""" Use gene wikipiedia to construct word embedding 

    before that, use the utils.spider to generate wikipedia
"""
import imp
import numpy as np
import pandas as pd
import torch
import os
import h5py
import argparse

# add sys path, ref: https://zhuanlan.zhihu.com/p/371299627
import sys
sys.path.append(os.getcwd())

from utils.file_utils import save_hdf5

# parameters
no_skip = True
# no_skip = Falsec
parser = argparse.ArgumentParser()
parser.add_argument('--model', 
                    # default='biobert',
                    # default='clinical_bert',
                    # default='dnabert',
                    # default='scibert', 
                    default='pubmed_bert',
                    choices=['biobert','clinical_bert','dnabert','scibert','pubmed_bert'], help='Choose Bert ')
parser.add_argument('--data', default='tcga', choices=['tcga','cptac'], help='Choose data ')
parser.add_argument('--label_space', default='oncokb', choices=['oncokb','react','expand'], help='Choose label space ')

args = parser.parse_args()

args.data = 'tcga'
args.label_space = 'react'
args.label_space = 'oncokb'

# model_name = 'biobert'
# model_name = 'bioner'
# model_name = 'clinical_bert'
model_name = args.model
# model 1: bio bert
if model_name == 'biobert': #  https://huggingface.co/dmis-lab/biobert-v1.1
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# model 2: bio-NER
elif model_name == 'bioner': #https://huggingface.co/alvaroalon2/biobert_genetic_ner
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("Ilyabarigou/Genesis-harrybotter")
    model = AutoModelForCausalLM.from_pretrained("Ilyabarigou/Genesis-harrybotter")

elif model_name == 'clinical_bert': # https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

elif model_name == 'dnabert': # https://huggingface.co/armheb/DNA_bert_6
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
    model = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_6")
elif model_name == 'scibert': #https://aclanthology.org/D19-1371/
    from transformers import AutoTokenizer,AutoModel
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

elif model_name == 'pubmed_bert':
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

""" load wikipedia from files
"""

list_path = './data/processed data/list'
save_path = os.path.join(list_path, '{}_{}_model_{}_encoded_gene_card_embed.h5'.format(args.data, args.label_space, model_name))

gene_card_pd = pd.read_csv(os.path.join(list_path,'{}_{}_gene_card_embed.csv'.format(args.data,args.label_space)),index_col=False)
# word_list = gene_card_pd['index'].values.tolist()
word_list = gene_card_pd.iloc[:,0].values.tolist()


def csv2dic(df):
    columns=['Entrez Gene Summary', 'CIViC Summary', 'GeneCards Summary', 'UniProtKB/Swiss-Prot Summary']
    # concate 0 + 1 + 2 columns

    dic = {}
    # dic_keys = df['index'].values.tolist()
    dic_keys = df.iloc[:,0].values.tolist()
    for i in range(len(df)):
        dic[dic_keys[i]] = df.iloc[i,1] + df.iloc[i,3] + df.iloc[i,4]

    return dic

sequence_set = csv2dic(gene_card_pd)  # {"word": sequence}

if no_skip == True:
    """ inference
    """
    word_embedding = {}
    for count, token_name in enumerate(word_list):
        sequence = sequence_set[token_name] # string

        # pre-process
        tokens = tokenizer.tokenize(sequence)
        # print(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_ids
        # set maximun length
        encoded_input = tokenizer(sequence, return_tensors="pt",  truncation=True, max_length=512)
        decoded_sequence = tokenizer.decode(encoded_input["input_ids"].squeeze()) # [cls] text.. [seq]

        # inference
        last_hidden_state = model(encoded_input["input_ids"]) # last hidden state: [0] embedding; [1] pooling embedding
        output = last_hidden_state[0]
        cls_token_embed = output[:,0,:] # (batch, tokens+2, embedding) -> (1, embedding)

        if count == 0:
            print(cls_token_embed.size())
        ## numpy save
        # if count == 0:
        #     word_embedding = np.empty((0,cls_token_embed.squeeze().size()[0]), float)
        # word_embedding = np.append(word_embedding, cls_token_embed.detach().numpy(), axis=0)

        # h5 save
        word_embedding[token_name] = cls_token_embed.detach().numpy() # (token_num, embedding_num) 

    """ save embedding
    """
    # output = torch.from_numpy(word_embedding).type(torch.float32)
    outputs = word_embedding
    save_hdf5(output_path=save_path, asset_dict=outputs)

""" load embedding
"""
file = h5py.File(save_path, 'r')

for idx, token_name in enumerate(word_list):
    dset = file[token_name]
    cur_embed = dset[:]
    if idx == 0:
        word_embedding = cur_embed
        # np.empty((0, np.shape(cur_embed)[-1]), float)
    else:
        word_embedding = np.append(word_embedding, cur_embed, axis=0)
file.close()


def load_embedd(save_path, word_list, covert_pt=True):
    import h5py
    file = h5py.File(save_path, 'r')

    for idx, token_name in enumerate(word_list):
        dset = file[token_name]
        cur_embed = dset[:]
        if idx == 0:
            word_embedding = cur_embed
        else:
            word_embedding = np.append(word_embedding, cur_embed, axis=0)
    file.close()

    # convert to tensor
    if covert_pt == True:
        word_embedding = torch.from_numpy(word_embedding).type(torch.float32)

    return word_embedding