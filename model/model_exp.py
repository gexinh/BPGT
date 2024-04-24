"""
Experiment for model
Backbone models for the visual extractor
includes:
    1. DeepHis
    2. GeneHe
    3. HIPT
    4. Trans-MIL
    5. KAT
    6. AB-MIL
"""
from tkinter import E
from typing import Sequence
from matplotlib.lines import Line2D
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.model_selection import model_selection
from utils.utils import initialize_weights
import math
from torch.nn.parameter import Parameter

from model.base_model import size_selection
from model.base_model import transformer_FC, TransMIL, Attn_Net_Gated, AttnClassifier, SA_MIL, KAT
from model.MCAT import SNN
from model.base_model import GCN, Graph_Tran, MLDecoder, MCAT_Decoder
from model.Geneformer import GeneformerLayer, Geneformer


class GatedAttention(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)
    
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1, dropout_rate=0.5):
        super(GatedAttention, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout_rate))
            self.attention_b.append(nn.Dropout(dropout_rate))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b) # inner product
        A = self.attention_c(A)  # N x 1
        return A, x

class ABMIL(nn.Module):
    """ attention based MIL for binary classification/ multi-label classification

    """
    def __init__(self, size_arg = "dino", dropout = False, dropout_rate=0.5):
        super(ABMIL, self).__init__()
        size = size_selection(size_arg)
        # self.size_dict = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
        #                 "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64]}
        # size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc) 

        initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        A, h = self.attention_net(h)  # nx1 and nxd
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        
        # matrix production
        M = torch.mm(A, h)   # 1 x d
        # 

        return M

class model_markets(nn.Module):
    """ model markets
                O = softmax(Q*K^T./sqrt(d)) * V 
        Inputs:
            In1: (, d) embedding to generate Q and K
            In2: (d, K) embdding to generate V
        Outputs:
            (, K) embedding
    """
    def __init__(self, num_classes, num_of_groups=-1, 
                    model_size='1024_small',zsl=0, adj=None, dropout=True, 
                    dropout_rate=0.5, k_sample=8, num_layers_decoder=1, num_layers_encoder=2, 
                    graph_module='geneformer', graph_connect='forward', agg_method='att', cls_token=True,
                    region_cls=True, num_region=1, instance_loss_fn=None, 
                    spatial_mask=None, pe=None,num_graph_layer=2, word_embedd=None):
            super(model_markets, self).__init__() 

            size = size_selection(model_size)

            initial_d_features = size[1]   # x: 384 -> 256 
            d_decoder_embed = size[1]    #  256 

            # patch-leve extractor
            elif agg_method == 'dense':
                # Deephis
                self.att = AttnClassifier(num_classes=num_classes, aggregation='mean', size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate, k_sample=-1)
                region_cls = False
                initial_d_features = size[2]
                initial_d_features = size[2]
                d_decoder_embed = initial_d_features

            elif agg_method == 'self_att':
                # attention for GeneHe
                self.att = AttnClassifier(num_classes=num_classes, aggregation='att', size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate, k_sample=-1)                
                region_cls = False
                initial_d_features = size[2]
                d_decoder_embed = initial_d_features


            elif agg_method == 'abmil':
                # attention based MIL
                self.att = ABMIL(size_arg = model_size, dropout = dropout, dropout_rate=dropout_rate)
                region_cls = False
                initial_d_features = size[1]
                d_decoder_embed = initial_d_features

            elif agg_method == 'samil':
                # Self-attention 
                self.att = SA_MIL(model_size)

            elif agg_method == 'hipt':
                self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)

            elif agg_method == 'kat':
                self.att = KAT(model_size=model_size, depth=num_layers_encoder, heads=8,dropout = dropout_rate)

            elif agg_method == 'trans_mil':
                self.att = TransMIL(model_size, n_classes=num_region)

            self.agg_method = agg_method


            # parameters
            d_label_embed = d_decoder_embed # 256

            # region classification
            self.region_cls = region_cls
            if self.region_cls == True and num_region != 1 and agg_method != 'trans_mil':
                model_dict = {"model":'wsl',  
                'n_classes': num_region, "size_arg": "hipt", 
                'k_sample': k_sample, 
                "dropout": dropout,
                'dropout_rate':dropout_rate}

                self.region_cls_module = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
                self.alpha = torch.nn.Parameter(data=torch.ones(1, dtype=torch.float32) ,requires_grad=True)

            # label embedding
            """ d_decoder_embed: the dimension of decoder's input embedding: 128
                num_label_embed: the length of label embedding
            """
            num_label_embed = 100 if num_of_groups < 0 else num_of_groups
            if num_label_embed > num_classes:
                num_label_embed = num_classes

            # create non-learnable label queries
            self.if_word_embedd = False
            d_label_embed_m = d_label_embed if graph_module != 'gcn' else d_label_embed * 3
            if not zsl:            
                if word_embedd is None:            
                    query_embed = nn.Embedding(num_label_embed, d_label_embed_m)
                    # embeddin的weight已经是对每个label产生不同的索引
                    query_embed.requires_grad_(False)
                else:
                    query_embed = word_embedd
                    self.word_proj = nn.Linear(in_features= word_embedd.size()[-1], out_features=d_label_embed_m, bias=True)
                    # TODO: modify
                    self.word_proj.requires_grad_(True)
                    self.if_word_embedd = True 
                # matrix factorization??
                # if word_embedded is not None:
                # query_embedd = factorization(word_embedd, d_label_embedd)  # (labels, 768) -> (labels, label embedd) * (label embedd, 768)
            else:
                query_embed = None
            

            self.query_embed = query_embed
            self.zsl = zsl

            if str(type(adj)) == "<class 'numpy.ndarray'>":
                self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
            else:
                self.adj = adj
            
            self.mlp = False
            if graph_module == 'geneformer':
                graphlayer = GeneformerLayer(d_label_embed, nhead=8, dim_feedforward=d_label_embed, dropout=0.1, activation="relu",
                        layer_norm_eps=1e-5)      
                self.gcn = Geneformer(graphlayer,d_label_embed, num_layer=num_graph_layer, adj=adj, spatial_mask=spatial_mask, pe=pe, if_cls_token=cls_token)        
            
            elif graph_module == 'gcn':
                self.gcn = GCN(nclass=d_decoder_embed,model_size=model_size,dropout=0.1)
            
            elif graph_module == 'graph_transformer':
                self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=2, adj=adj, if_cls_token=cls_token)
            
            elif graph_module == 'mlp':

                self.mlp_classifier = nn.Linear(d_decoder_embed, num_classes)
                self.mlp = True
                cls_token = False

            elif graph_module == 'mcat':
                from model.MCAT import SNN
                self.gcn = SNN(input_dim=768, model_size_omic='small')


            self.graph_module = graph_module

            self.cls_token = cls_token
            if self.cls_token:
                self.fc = nn.Linear(d_label_embed, num_classes)

            self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
            
            if dropout:
                self.forward_net.append(nn.Dropout(dropout_rate)) 
            self.forward_net = nn.Sequential(*self.forward_net)
            
            self.gamma = nn.Parameter(torch.ones(1))
            self.graph_connect = graph_connect

            # multi-label classification
            if graph_module == 'mcat':
                self.ml_decoder = MCAT_Decoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                    initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)
            else:        
                self.ml_decoder = MLDecoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                    initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)
            
                    # decoder embedding is the dimension of cross attention 
                    # initial_num_features is the embeeding dimension from x 
            initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, h_coor=None, label=None, instance_eval=True):
        results_dict = {}
        Y_region_logits = []
        Y_region_prob = []
        A = []
        a_score = []
        
        # patch aggregatio            
        
        if self.agg_method =='dense':
            # _, x, _= self.att(h.unsqueeze(0)) # d, features
            # if self.rand_sample:
            total_patch = h.size()[0]
            row_sequence = np.arange(total_patch)
            np.random.shuffle(row_sequence)
            # k = int(np.random.choice(self.k_sample))
            k = row_sequence[0:500]
            h = h[k, :]
            _, x, _ = self.att(h.unsqueeze(0)) # 1, features

        elif self.agg_method =='self_att':
            # if self.rand_sample:
            total_patch = h.size()[0]
            row_sequence = np.arange(total_patch)
            np.random.shuffle(row_sequence)
            # k = int(np.random.choice(self.k_sample))
            k = row_sequence[0:500]
            h = h[k, :]                
            _, x, _= self.att(h.unsqueeze(0)) # d, features

        else:
            if self.agg_method =='kat':
                result = self.att(h, krd=h_coor)
            else:
                result = self.att(h)
            Y_region_logits = result['logits']
            Y_region_prob = result['Y_prob']
            Y_region_hat = result['Y_hat']
            h_cls = result['h_cls']
            x = result['h_feat']


        adj = self.adj

        if self.mlp != True:
            # label embedding 
            if self.zsl:
                query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
            else:
                if self.if_word_embedd: 
                    query_embed = self.word_proj(self.query_embed) if self.graph_module!='gcn' else self.query_embed
                else:
                    query_embed = self.query_embed.weight

            # graph
            if self.graph_connect == 'forward':            
                y = self.gcn(query_embed, adj=adj) if self.graph_module=='gcn' else self.gcn(query_embed) 
                
            elif self.graph_connect == 'bypass':
                y = self.forward_net(query_embed)
                b = self.gcn(query_embed, adj)
                y = y + self.gamma * b
            elif self.graph_connect == 'abandon':
                y = self.forward_net(query_embed)

            logits, virtual_node, a_score = self.ml_decoder(x, y)                  

            if self.cls_token:
                logits = self.fc(virtual_node)

        else:
            # MLP classifier
            if len(x.size())==2:
                x = x.mean(dim=0) #patch, feature - > 1, feature

            logits = self.mlp_classifier(x)

        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob

        return logits, probs, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.mlp:
            self.mlp_classifier = self.mlp_classifier.to(device)
        else:
            self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)
