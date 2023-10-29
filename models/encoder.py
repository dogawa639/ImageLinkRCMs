import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from deeplabv3 import resnet50, ResNet50_Weights
from transformer import Transformer
from gnn import GAT
from general import FF, SLN

import numpy as np

# forward input(patch(c, h, width), w) -> (bs, emb_dim)
# w: None -> bs=1
# w: tensor(bs, w_dim) -> bs=bs

class CNNEnc(nn.Module):
    def __init__(self, patch_size, emb_dim, adj_mat, num_source=1, sln=True, w_dim=10):
        super().__init__()

        self.patch_size = patch_size  #(c, h, w)
        self.emb_dim = emb_dim
        self.adj_mat = adj_mat  # tensor
        self.sln = sln
        self.w_dim = w_dim
        self.mid_dim = int(patch_size[1]/32*patch_size[2]/32*2048)

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, True, True])
        self.flatten = nn.Flatten()
        self.lin = [FF(self.mid_dim, emb_dim, bias=True) for _ in range(num_source)]
        self.norm0 = nn.LayerNorm(emb_dim)
        if sln:
            self.norm1 = SLN(w_dim, self.mid_dim)
            self.norm2 = SLN(w_dim, emb_dim)  # layer norm at the last output
        else:
            self.norm1 = nn.LayerNorm(self.mid_dim)
            self.norm2 = nn.LayerNorm(emb_dim)

        self.seq = nn.Sequential(
            self.norm0,
            self.resnet50,
            self.flatten,
            self.norm1
        )

        # graph encoding
        self.gnn = GAT(emb_dim, emb_dim, k=1, depth=4, dropout=0.1)

    def forward(self, patch, source_i=0, w=None):
        # no batch
        # patch: (c, h, w)
        # w: (bs, w_dim)
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        x = self.seq(patch)
        x = self.lin[source_i](x)
        if self.sln:
            x = self.norm2(x, w)
        else:
            x = self.norm2(x)
        return x
    
    def encode(self, features):
        # features: (bs, line_num, emb_dim)
        x = torch.zeros(features.shape, dtype=torch.float32, device=features.device)
        for i in range(features.shape[0]):
            x[i, :, :], atten = self.gnn(features[i, :, :], self.adj_mat)
        return x
