import copy
import math
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
        mask = mask.permute(1,2,0,3)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value).permute(2,0,1,3), p_attn.transpose(0,1)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states):
        for layer in self.layers:
            x, y = layer(x, hidden_states)
        return self.norm(x), y


class DecoderLayer(nn.Module):
    def __init__(self, d_model, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        # self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, hidden_states):
        m = hidden_states
        # x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        # return self.sublayer[2](x, self.feed_forward), self.self_attn.attn, self.src_attn.attn

        
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward), self.src_attn.attn


class MultiHeadedAttention(nn.Module):
    """
        Inputs:
            query, key, value, and mask  \in R^(tokens, batch, dmodel)

        Outputs:
            x, attention
    """
    def __init__(self, nhead, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.h = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        
        if mask is not None:
            mask = mask.unsqueeze(0)
        nbatches = query.size(1)
        query, key, value = \
            [l(x).view(-1, nbatches, self.h, self.d_k)
             for l, x in zip(self.linears, (query, key, value))]

        # view is resize
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.contiguous().view(-1, nbatches, self.h * self.d_k)
        
        # heat_map = self.attn.mean(0).squeeze()

        # x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


""" Auxiluary Task
"""
class PositionwiseFeedForward(nn.Module):
    """
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)