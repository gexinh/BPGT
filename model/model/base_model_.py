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