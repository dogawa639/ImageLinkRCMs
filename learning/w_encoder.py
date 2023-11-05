import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.gnn import GT
from models.general import FF

import numpy as np

class CNNWEnc(nn.Module):
    # input: (sum(links), total_feature, 3, 3)
    # output: (sum(links), 3, 3)
    def __init__(self, total_feature, w_dim):
        super().__init__()

        self.total_feature = total_feature
        self.w_dim = w_dim

        self.lin = FF(total_feature, w_dim, total_feature*2, bias=True)

    def forward(self, input, g_output):
        # g_output: prob of each link
        g_output = g_output.unsqueeze(1)
        feature = (input * g_output).sum((0, 2, 3)) / g_output.sum()  # (total_feature)
        w = self.lin(feature)
        return w
    
class GNNWEnc(nn.Module):
    # input: (bs, link_num, feature_num)
    # output: (bs, link_num, link_num)
    def __init__(self, feature_num, emb_dim, w_dim, adj_matrix):
        super().__init__()

        kwargs = {"enc_dim": int(emb_dim / 2), "in_emb_dim": None, "num_head": 1, "dropout": 0.0, "depth": 3, "pre_norm": False, "sn": False, "sln": False, "w_dim": None, "output_atten": False}
        self.gnn = GT(feature_num, emb_dim, adj_matrix, **kwargs)
        self.lin = FF(emb_dim*2, w_dim, emb_dim*4, bias=True)

    def forward(self, input, g_output):
        # g_output: prob of link transition
        features = self.gnn(input)
        features = torch.cat((features.unsqueeze(1), features.unsqueeze(2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        features = (features * g_output.unsqueeze(-1)).sum((0, 1, 2)) / g_output.sum()  # (emb_dim*2)
        w = self.lin(features)
        return w




