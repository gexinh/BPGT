"""
    Simple implementation of BPGT
"""

import copy
import math
from matplotlib.pyplot import axis
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    """ input: (tokens, batch, head, dmodel)

        reshape-> (batch, head, tokens, dmodel) for the batched matrix multiply

        output: x \in (tokens, batch, head, dmodel), score \in (head, batch, tokens, tokens)
    """
    d_k = query.size(-1)
    # reshape
    query, key, value = query.permute(1,2,0,3), key.permute(1,2,0,3), value.permute(1,2,0,3)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        if len(mask) != 0:
            # normal version
            # mask = mask.permute(1,2,0,3)
            # scores = scores.masked_fill(mask == 0, -1e9)
            nhead = scores.size()[1]
            # BPGT
            for m in mask:
                # scores: (batch, head, tokens, tokens)
                # mask: (batch, tokens, tokens) -> broadcast-> (batch, head, tokens, tokens)
                scores = scores + m.unsqueeze(1).repeat(1,nhead, 1, 1)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).permute(2,0,1,3), p_attn.transpose(0,1)


class MultiHeadedAttention(nn.Module):
    """
        Inputs:
            query, key, value, and mask  \in R^(tokens, batch, dmodel)

        Outputs:
            x \in R^(tokens, batch, dmodel)
            attention \in R^(batch, tokens, tokens) 
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.h = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        
        nbatches = query.size(1)
        query, key, value = \
            [l(x).view(-1, nbatches, self.h, self.d_k)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.contiguous().view(-1, nbatches, self.h * self.d_k)
        self.attn = self.attn.mean(axis=0)
        # heat_map = self.attn.mean(0).squeeze()

        # x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn



from torch.nn.modules.transformer import _get_activation_fn
class BPGTLayer(nn.Module):
    """ MHD -> FFN -> Residual Connection
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(BPGTLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # need weight == True -> return [tgt, attention score]
        self.multihead_attn = MultiHeadedAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(BPGTLayer, self).__setstate__(state)

    def forward(self, tgt, tgt_mask = None,):

        # tgt = tgt + self.dropout1(tgt)
        # tgt = self.norm1(tgt)
        
        # BPGT attention
        tgt2, attn = self.multihead_attn(tgt, tgt, tgt, mask=tgt_mask)
        
        # Resiudal 1
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        
        # Resiudal 2
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


class BPGT(nn.Module):
    """ 
        Args:
            decoder layer: BPGT layer
            in_dim: dmodel
            num_layer: the number of BPGT layer
            spatial mask: mutual information \in R^(K, K)
            pe: organ positional encoding \in R^(K, N)
            adj
            
        Outputs:
            target \in R^(K, dmodel)
            att \in R^(K, K)

    """
    def __init__(self, decoder_layer, in_dim, num_layer, adj=None, spatial_mask=None, pe=None, if_cls_token=True):
        super(BPGT, self).__init__()    

        self.layers = clones(decoder_layer, num_layer) # multi-head self-attention


        self.if_cls_token = if_cls_token
        self.cls_token = nn.Parameter(torch.randn(1, in_dim))

        if pe is not None:
            if isinstance(pe, np.ndarray):
                pe = torch.from_numpy(pe).float()
            pos_enc_dim = pe.size()[-1]
            self.pe_proj = nn.Linear(pos_enc_dim, in_dim, bias=True) # positional embedding
            # modify 11/6 
            nn.init.xavier_normal_(self.pe_proj.weight)
            self.pe_proj.bias.data.zero_()

        if spatial_mask is not None: 
            if isinstance(spatial_mask, np.ndarray):
                spatial_mask = torch.from_numpy(spatial_mask).float()
                if spatial_mask.sum() > spatial_mask.size()[0]* spatial_mask.size()[1]:
                    spatial_mask = spatial_mask/spatial_mask.size()[0] # normalize
            K = spatial_mask.size()[-1] 
            self.mask_scalar_proj = nn.Parameter(torch.randn(K, K))
            # TODO: modify 11/6
            nn.init.uniform_(self.mask_scalar_proj)
            # nn.init.xavier_normal_(self.mask_scalar_proj)
        
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj).float()       


        # graph related 
        self.mask = spatial_mask
        self.pe, self.adj = self.positional_encoding(pe=pe, adj=adj, if_cls_token=if_cls_token)

        if self.adj is not None:
            #TODO: modify if we use the abs 
            self.adj = torch.abs(self.adj)
            K = self.adj.size()[-1]  
            self.adj_scalar_proj = nn.Parameter(torch.randn(K, K))  
            # modify 11/6          
            nn.init.uniform_(self.adj_scalar_proj)
            # nn.init.xavier_normal_(self.mask_scalar_proj)

        


    def positional_encoding(self, pe, adj, if_cls_token=True):
        """ input: pe \in (K,N)

            output: pe \in (K+1, N)
        """
        if pe is not None:        
            N = pe.size()[-1]
            
            if if_cls_token:
                # add virtual node
                row = torch.ones(N).unsqueeze(0)
                pe = torch.cat([row, pe], dim=0) 
        
        if adj is not None:
            # all-one matrix
            if sum(sum(adj)) == adj.size()[0]*adj.size()[1]:
                flag = 0
            else:
                flag = 1

            if if_cls_token == True:
                # add virtual node 
                row = torch.ones(adj.size()[1]).unsqueeze(0)
                adj = torch.cat([row, adj], dim=0) 
                col = torch.ones(adj.size()[0]).unsqueeze(1)
                adj = torch.cat([col, adj], dim=1) 
            
            if flag:             
                from scipy import sparse as sp
                number_of_nodes = adj.size()[0]
                A = adj
                D, A = A.sum(axis=0) - torch.ones(number_of_nodes), A.numpy()
                N = sp.diags(D.numpy().clip(1) ** -0.5, dtype=float)
                L = sp.eye(number_of_nodes) - N * A * N
                L = torch.from_numpy(L).float()
                adj = L

        return pe, adj

    def forward(self, h, atten_score=False):
        device = h.device
        if self.if_cls_token:           
            h = torch.cat((self.cls_token, h), dim=0)

        if len(h.size()) == 3:
            bs = h.size()[1]           
        else:
            bs = 1
        h = h.unsqueeze(1).repeat(1, bs, 1)
        
        if self.pe is not None:
            self.pe = self.pe.to(device)
            output = h + self.pe_proj(self.pe).unsqueeze(1).repeat(1, bs, 1)
        else:
            output = h
        
        tgt_mask = []
        if self.adj is not None:
            self.adj = self.adj.to(device)
            tgt_mask.append((self.adj_scalar_proj * self.adj).unsqueeze(0).repeat(bs, 1, 1)) #(batch, tokens, tokens)

        if self.mask is not None:
            self.mask = self.mask.to(device)
            tgt_mask.append((self.mask_scalar_proj * self.mask).unsqueeze(0).repeat(bs, 1, 1))
            
        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask)

            if len(output) == 2:
               attn = output[1]
               output = output[0] 
        
        if len(output.size()) == 3:
            output = output.transpose(0, 1) # -> (batch, tokens, tokens)
            output = output.squeeze() #-> (tokens, tokens)
        
        if atten_score:
            return output, attn
        else:
            return output