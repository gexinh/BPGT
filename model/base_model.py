from tkinter import E
from typing import Sequence
from matplotlib.lines import Line2D
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import math
from torch.nn.parameter import Parameter

def size_selection(size_arg):
    size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                             "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                             "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                             }

    size = size_setting[size_arg]
    return size

""" CLAM
"""
class Attn_Net(nn.Module):
    
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

class Attn_Net_Gated(nn.Module):
    """Attention Network with Sigmoid Gating (3 fc layers)
    
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1, dropout_rate = 0.5):
        super(Attn_Net_Gated, self).__init__()
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
        initialize_weights(self)
        
    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b) # inner product
        A = self.attention_c(A)  # N x 1

        return A, x

class CLAM_SB(nn.Module):
    """ 
        model: 'wsl'- weakly supervised learning
    """
    def __init__(self, model='wsl', gate = True, size_arg = "small", dropout = False, k_sample=8, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dropout_rate=0.5):
        super(CLAM_SB, self).__init__()
        size = size_selection(size_arg)
        
        self.model = model

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1, dropout_rate=dropout_rate)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

        

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.attention_net = self.attention_net.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        self.classifiers = self.classifiers.to(device)


    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]   # topk -> (values, inds)
        # print('topk ind is {}'.format(top_p_ids))
        # print(A.size())
        # print(top_p_ids)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        # print(logits.data.size(), all_targets.data.size())
        instance_loss = self.instance_loss_fn(logits, all_targets)
        
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device

        # print('before att', h, h.size())
        A, h = self.attention_net(h)  # NxK
        # print('after att', h, h.size())

        A = torch.transpose(A, 1, 0)  # KxN            
        A = F.softmax(A, dim=1)  # softmax over N
        A_raw = A

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            # print(inst_labels)

            for i in range(len(self.instance_classifiers)):
                # print(i)
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
                    
        M = torch.mm(A, h) 
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]  
        Y_prob = F.softmax(logits, dim = 1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})


        return logits, Y_prob, Y_hat, A_raw, results_dict


"""
Attention Based MIL
"""
class Attn_Based_MIL(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Based_MIL, self).__init__()
        self.query = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.keys = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.query.append(nn.Dropout(0.25))
            self.keys.append(nn.Dropout(0.25))

        self.query = nn.Sequential(*self.query)
        self.keys = nn.Sequential(*self.keys)
        
        self.additional_transform = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.query(x)
        b = self.keys(x)
        A = a.mul(b) # inner product
        A = self.additional_transform(A)  # N x n_classes

        return A, x

""" Basic model:
        GeneHe:        
        MLP + self-attention + average pooling  -> binary classification
        DeepHist:
        MLP + average pooling  -> binary classification
""" 

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( 1 X C X N)
            returns :
                out : (1 X C X N) self attention value + input feature
                attention:  1 X N X N  (N is number of patches)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)  # 1 X N X C
        proj_key = self.key_conv(x)  # 1 X C x N
        energy = torch.bmm(proj_query, proj_key)  # 1 X N X N
        attention = self.softmax(energy)  # 1 X N X N

        out = torch.bmm(x, attention.permute(0, 2, 1))  # 1 X C X N
        out = self.gamma * out + x

        return out

class AttnClassifier(nn.Module):
    def __init__(self, aggregation='att',size_arg = "small", num_classes=2, k_sample=-1, dropout=True, dropout_rate=0.1):
        super(AttnClassifier, self).__init__()
        size = size_selection(size_arg)

        self.agg = aggregation
        self.fc1 = [nn.Linear(size[0], size[1]), nn.ReLU()]
        self.fc2 = [nn.Linear(size[1], size[2]), nn.ReLU()]

        self.attn = Self_Attn(size[2])

        self.fc3 = nn.Linear(size[2], num_classes)
        self.num_classes= num_classes
        self.k_sample = k_sample

        if dropout:
            self.fc1.append(nn.Dropout(dropout_rate))
            self.fc2.append(nn.Dropout(dropout_rate))

        self.fc1 = nn.Sequential(*self.fc1)
        self.fc2 = nn.Sequential(*self.fc2)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        """
        Inputs:
            x: (1, patch, features)
        Outputs: 
            x: (1, features)
        """
        # ramdomly select
        if self.k_sample != -1:
            total_patch = x.size()[1]
            row_sequence = np.arange(total_patch)
            np.random.shuffle(row_sequence)
            # k = int(np.random.choice(self.k_sample))
            k = row_sequence[0:self.k_sample]
            x = x[:, k, :]

        # forward
        x = self.fc1(x)
        x = self.fc2(x)

        if self.agg == 'att':
            x = self.attn(x.permute(0, 2, 1))  # X: 1, features, patches
            x2 = x.mean(dim=-1) # 1, features
            x3 = x.permute(0, 2, 1)

        elif self.agg == 'mean':
            x3 = x
            x2 = x.permute(0, 2, 1).mean(dim=-1)

        elif self.agg == 'max':
            x3 = x
            x2 = x.permute(0, 2, 1).max(dim=-1)

        x = self.fc3(x2) 
        
        # x3 is the tensor without pooling
        if self.num_classes == 2:
            return x
        else:
            # x: (1, k), x2: (1,d), x3: (1, p, d)
            return x, x2, x3

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: modify this part
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.attn = self.attn.to(device)
        self.fc3 = self.fc3.to(device)


from model.KAT import KATBlocks, PreNorm, FeedForward, KernelAttention
class KAT(nn.Module):
    def __init__(self, model_size, depth, heads=8, num_kernal=16, dim_head = 64, dropout = 0.5, emb_dropout = 0.):
        super().__init__()

        # TODO: dismiss the patch-level pool
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        size = size_selection(model_size)
        num_classes = 9 # cancer types

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, dim))
        self.nk = num_kernal

        self.dropout = nn.Dropout(emb_dropout)
        num_pk=0.8
        mlp_dim=1024
        self.kt = KATBlocks(num_pk, dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes) 
        )

    def forward(self, node_features, krd, mask=None, kmask=None):
        x = self.to_patch_embedding(node_features)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b = b, k = self.nk)

        x = self.dropout(x)
        k_reps, clst = self.kt(x, kernel_tokens, krd, cls_tokens, mask, kmask)

        h_feature = k_reps
        h_cls = clst
        logits = self.mlp_head(h_cls)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'h_feat':h_feat, 'h_cls':h_cls}

        return results_dict

from torch.nn.modules.transformer import TransformerEncoderLayer

"""# Hierarchical Transformer for HIPT
"""
class transformer_FC(nn.Module):
    """ Self attention Layer
    """
    def __init__(self, num_layer, size_arg, dropout_rate):
        super(transformer_FC, self).__init__()
        self.size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                        "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                        "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                        }
        dmodel = self.size_setting[size_arg]
        in_dim = dmodel[0]    
        att = []
        for _ in range(num_layer):
            att.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim*4,dropout=dropout_rate,activation="relu"))
        self.att = nn.Sequential(*att)

    def forward(self, x):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        out = self.att(x)
        return out

"""# the modified Trans MIL
"""
from .TransMIL import TransLayer, PPEG

class TransMIL(nn.Module):
    def __init__(self, model_size, n_classes):
        super(TransMIL, self).__init__()
        size = size_selection(model_size)

        self.pos_layer = PPEG(dim=size[1])
        self._fc1 = nn.Sequential(nn.Linear(size[0], size[1]), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, size[1]))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=size[1])
        self.layer2 = TransLayer(dim=size[1])
        self.norm = nn.LayerNorm(size[1])
        self._fc2 = nn.Linear(size[1], self.n_classes)

        # PE
        d_word_vec = size[1]
        self.position_enc = torch.nn.Parameter(torch.Tensor(1, d_word_vec))
        # self.position_enc = nn.Embedding(
                    # n_position, d_word_vec, padding_idx=Constants.PAD)

        # self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

    def forward(self, h):
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)

        # TODO: old version
        h_feat, h_cls = h[:,1:], h[:,0]
        # remove padding patch
        # h_feat, h_cls = h[:,1:-add_length], h[:,0]

        #---->predict
        logits = self._fc2(h_cls) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'h_feat':h_feat, 'h_cls':h_cls}

        return results_dict


""" graph related part: MCAT, GCN, Graph Tranformer
"""
class MCAT_Decoder(nn.Module):
    """ the MCAT decoder: concate the label embedding (y) with cross-attentioned embedding (h) before label-wise projection 

            x:(patch, initial_num_features) -> (patch, d_decoder_embed)
            y:(num_classes, num_label_embed)
        Args: 
            num_label_embed
            d_decoder_embed
            initial_num_features: 
        Input: K, Q, V

        Output: logits
    """
    def __init__(self, num_classes, num_label_embed, d_decoder_embed=768,
                 initial_num_features=2048, zsl=0, num_layers_decoder=1, cls_token=False):
        super(MCAT_Decoder, self).__init__()

        self.cls_token = cls_token
        # switching to 768 initial embeddings
        d_decoder_embed = 768 if d_decoder_embed < 0 else d_decoder_embed
        
        if initial_num_features != d_decoder_embed:
            embed_standard = nn.Linear(initial_num_features, d_decoder_embed)
            self.standard = True
        else:
            embed_standard = []
            self.standard = False

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = num_layers_decoder

        #TODO: usually the 'dim_feedforward' should be higher than the 'd_decoder_embed'
        # dim_feedforward = d_decoder_embed * 2  # the dimension of feed forward layer in the transformer decoder
        dim_feedforward = d_decoder_embed
        #TODO: use customized cross-attention layer
        # import copy
        # c = copy.deepcopy
        # attn = MultiHeadedAttention(nhead=8, d_model=d_decoder_embed)
        # ff = PositionwiseFeedForward(d_model=d_decoder_embed, d_ff=dim_feedforward,dropout=0.1)
        # self.decoder = Decoder(DecoderLayer(
        #                         d_model=d_decoder_embed, src_attn=c(attn), feed_forward=c(ff), dropout=0.1), 
        #                         N=num_layers_decoder)

        ######## old version ######
        layer_decode = TransformerDecoderLayerOptimal(d_model=d_decoder_embed,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)

        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        ###########################

        self.decoder.embed_standart = embed_standard # initial feature -> d_decoder_embed

        self.zsl = zsl

        if self.zsl:
            if d_decoder_embed != 300:
                self.wordvec_proj = nn.Linear(300, d_decoder_embed)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(d_decoder_embed, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / num_label_embed + 0.999)
            """ due to the concatenation
            """
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(num_label_embed, d_decoder_embed*2, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        
        # label-wise projection
        self.decoder.group_fc = GroupFC(num_label_embed)
        
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x, y, attention_only=False):
        data_shape = len(x.shape)
        # x feature
        embedding_spatial = x
        if data_shape == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
            bs = embedding_spatial[0]

        elif data_shape == 3:
            # [batch, patch, feature]            
            bs = embedding_spatial.shape[0]

        elif data_shape ==2:  # [patch, embedding]
            bs = 1

        if self.standard == True:
            embedding_spatial_standard = self.decoder.embed_standart(embedding_spatial)
            embedding_spatial_standard = torch.nn.functional.relu(embedding_spatial_standard, inplace=True)
        else:
            embedding_spatial_standard = embedding_spatial

        # y feature
        query_embed = y # [label, embedding]

        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        # h = self.decoder(tgt, embedding_spatial_standard.transpose(0, 1))  # [num_label_embed, batch, 768]
        # h = h.transpose(0, 1)

        """TODO: old torch version < 1.10 : 
            data need the dimension of batch size for MHA 
            input data shape -> (seq, batch, d_feature)
        """
        if data_shape == 3 or data_shape == 4:
            x_memory = embedding_spatial_standard.transpose(0, 1)
        else:
            # [patch, feature] -> [patch, batch, feature], which batch = 1
            x_memory = embedding_spatial_standard.unsqueeze(1)  # expand dim on axis 1

        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1) # [clsNum, feature] -> [clsNum, batch, feature]

        # h = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        # get attention 
        h, a_score = self.decoder(tgt, x_memory)  # [num_label_embed, batch, feature]
        h = h.transpose(0, 1) # [clsNum, batch, feature] -> [batch, clsNum, feature]

        """ key variant: concatenate y after h
        """
        h = torch.cat([h, tgt.transpose(0, 1)],dim = -1) # [] + [] -> [batch, clsNum, 2 * feature]

        if self.cls_token == True:
            cls_token = h[:,0,:]
            h = h[:,1:,:]
        else:
            cls_token = []

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        # out_extrap: [batch, clsNum, 1]
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)        
        # duplicate_pooling: [clsNum, feature, group ratio]

        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        return logits, cls_token, a_score


# GCN 
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nclass, model_size='small', dropout=0.1):
        """ 
            model_size
        """
        super(GCN, self).__init__()

        dmodel =  [nclass*3, nclass*2, nclass]

        self.gc1 = GraphConvolution(dmodel[0], dmodel[1])
        self.gc2 = GraphConvolution(dmodel[1], nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.relu(x)
        return x
        #TODO: delete the classifier
        # return F.log_softmax(x, dim=1)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gc1 = self.gc1.to(device)
        self.gc2 = self.gc2.to(device)


# Graph transformer
class Graph_Tran(nn.Module):
    """
        Graph Transformer

        Input:
            [N, d]

        Output:
            [N+1, d]  if virtual node
            [N, d] else
    """
    def __init__(self, in_dim, dropout_rate, num_layer, adj, if_cls_token=True):
        super(Graph_Tran, self).__init__()    
        # positional embedding 

        # multi-head self-attention
        mha = []
        for i in range(num_layer):
            #TODO: decrease the forward model
            mha.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim,dropout=dropout_rate,activation="relu"))
            # mha.append(TransformerEncoderLayer(d_model=in_dim,nhead=8,dim_feedforward=in_dim*4,dropout=dropout_rate,activation="relu"))
        self.mha = nn.Sequential(*mha) 

        self.cls_token = nn.Parameter(torch.randn(1, in_dim))
        pos_enc_dim = round(adj.size()[0]/2)
        self.pe = self.positional_encoding(adj=adj, pos_enc_dim=pos_enc_dim, if_cls_token=if_cls_token)
        self.if_cls_token = if_cls_token

        self.pe_proj = nn.Linear(pos_enc_dim, in_dim, bias=True)

    def positional_encoding(self, adj, pos_enc_dim, if_cls_token=True):
        """
            Graph positional encoding v/ Laplacian eigenvectors
        """
        from scipy import sparse as sp
        # Laplacian
        # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        # N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        # L = sp.eye(g.number_of_nodes()) - N * A * N
        if if_cls_token == True:
           # add virtual node 
           row = torch.ones(adj.size()[1]).unsqueeze(0)
           adj = torch.cat([row, adj], dim=0) 
           col = torch.ones(adj.size()[0]).unsqueeze(1)
           adj = torch.cat([col, adj], dim=1) 

        number_of_nodes = adj.size()[0]
        A = adj
        D, A = A.sum(axis=0) - torch.ones(number_of_nodes), A.numpy()
        N = sp.diags(D.numpy().clip(1) ** -0.5, dtype=float)
        L = sp.eye(number_of_nodes) - N * A * N
        

        # Eigenvectors with numpy
        # EigVal, EigVec = np.linalg.eig(L.toarray()) # np.linalg.eig: calculate eigen vector
        EigVal, EigVec = np.linalg.eig(L)
        idx = EigVal.argsort() # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        Pos_En = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

        # g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

        # # Eigenvectors with scipy
        # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
        # EigVec = EigVec[:, EigVal.argsort()] # increasing order
        # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
        
        return Pos_En

    def forward(self, h):
        """ (num_token, emb) -> (num_token, emb')
        """
        device = h.device
        self.pe = self.pe.to(device)
        pe = self.pe_proj(self.pe)

        if self.if_cls_token:           
            h = torch.cat((self.cls_token, h), dim=0) # [N, d] -> [N+1, d]

        h = h + pe
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        out = self.mha(h)
        out = out.squeeze(0)
        return out

class transformer_decoder(nn.Module):
    """ x_emb ----------------------------> 
                                            } --> transformer decoder/ dot product --> multi-label prediction
        label_emb --> graph transformer --> 
    """
    def __init__(self, num_classes, num_of_groups=-1, 
                model_size='1024_small',zsl=0, adj=None, dropout=True, 
                dropout_rate=0.5, k_sample=8, num_layers_decoder=1, gcn='forward', agg_method='att',num_layers_encoder=2, 
                region_cls=True, num_region=1,instance_loss_fn=None,cls_token=True):
        super(transformer_decoder, self).__init__() 
        self.size_setting = {"1024_small": [1024, 512, 256], "1024_big": [1024, 512, 384], "1024_min": [1024, 256,128], 
                             "dino": [384, 256, 128], "dino_b": [384, 300, 256], "dino_ss": [384, 64, 32], "dino_s": [384, 128, 64],
                             "hipt": [192, 128, 64], "hipt_att": [192,192,192]
                             }

        size = self.size_setting[model_size]

        initial_d_features = size[1]   # x: 384 -> 256 
        d_decoder_embed = size[1]    #  256 
        d_label_embed = d_decoder_embed # 256

        # patch aggregation
        if agg_method == 'att':
            self.att = Att_ML(size_arg=model_size, dropout=dropout, dropout_rate=dropout_rate)
        elif agg_method == 'tran':
            self.att = transformer_FC(num_layer=num_layers_encoder, size_arg=model_size, dropout_rate=dropout_rate)
        elif agg_method == 'cor_mil':
            self.att = TransMIL(model_size, n_classes=num_region)
        self.agg_method = agg_method

        # region classification
        self.region_cls = region_cls
        if self.region_cls == True and num_region != 1 and agg_method != 'cor_mil':
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
        if not zsl:            
            query_embed = nn.Embedding(num_label_embed, d_label_embed)  # embeddin的weight已经是对每个label产生不同的索引
            query_embed.requires_grad_(False)
        else:
            query_embed = None
        

        self.query_embed = query_embed
        self.zsl = zsl

        if str(type(adj)) == "<class 'numpy.ndarray'>":
            self.adj = adj.astype(np.float32).to_tensor(dtype=torch.float32)
        else:
            self.adj = adj
        
        # TODO: add transformer-based graph network 
        self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=2, adj=adj, if_cls_token=cls_token)        
        # self.gcn = Graph_Tran(in_dim=d_label_embed, dropout_rate=0.1, num_layer=3, adj=adj, if_cls_token=cls_token)        
        self.cls_token =cls_token
        if self.cls_token:
            self.fc = nn.Linear(d_label_embed, num_classes)

        self.forward_net = [nn.Linear(d_label_embed, d_decoder_embed), nn.ReLU()]
        if dropout:
            self.forward_net.append(nn.Dropout(dropout_rate)) 
        self.forward_net = nn.Sequential(*self.forward_net)
        
        self.gamma = nn.Parameter(torch.ones(1))
        self.gcn_connect = gcn

        # multi-label classification
        self.ml_decoder = MLDecoder(num_classes=num_classes, num_label_embed=num_label_embed, d_decoder_embed=d_decoder_embed,
                 initial_num_features=initial_d_features, zsl=zsl,num_layers_decoder=num_layers_decoder, cls_token=cls_token)
                 # decoder embedding is the dimension of cross attention 
                 # initial_num_features is the embeeding dimension from x 

        initialize_weights(self)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def forward(self, h, label=None, instance_eval=True):
        # patch aggregation
        if self.agg_method == 'cor_mil':
            result = self.att(h)
            Y_region_logits = result['logits']
            Y_region_prob = result['Y_prob']
            Y_region_hat = result['Y_hat']
            x = result['h_feat']
            results_dict = {}
        
        else:
            x = self.att(h)

            # region classification
            if self.region_cls == True:
                Y_region_logits, Y_region_prob, Y_region_hat, A, results_dict = self.region_cls_module(x.squeeze(), label, instance_eval)
                #x = A.transpose(0,1) * x.squeeze()            
                x = x.squeeze() + self.alpha * A.transpose(0,1) * x.squeeze()  # (P, 1) .* (P, D)

            else:
                Y_region_logits = []
                Y_region_prob = []
                A = []
                results_dict ={}

        adj = self.adj
        # label embedding 
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.query_embed))
        else:
            query_embed = self.query_embed.weight
        
        # graph
        if self.gcn_connect == 'forward':
            y = self.gcn(query_embed)
            
        # elif self.gcn_connect == 'bypass':
        #     y = self.forward_net(query_embed)
        #     b = self.gcn(query_embed, adj)
        #     y = y + self.gamma * b
        # elif self.gcn_connect == 'abandon':
        #     y = self.forward_net(query_embed)

        # decoder: TODO:without sigmoid, should be added in the loss
        logits, virtual_node, a_score = self.ml_decoder(x, y)
        if self.cls_token:
            logits = self.fc(virtual_node)

        probs = torch.sigmoid(logits)
        lasso_loss = self.compute_l1_loss(probs)

        results_dict['a_score'] = a_score
        results_dict['lasso_loss'] = lasso_loss
        results_dict['Y_region_logits'] = Y_region_logits
        results_dict['Y_region_prob'] = Y_region_prob
        return logits, probs, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gcn = self.gcn.to(device)
        self.att = self.att.to(device)
        self.query_embed = self.query_embed.to(device)
        self.ml_decoder = self.ml_decoder.to(device)

