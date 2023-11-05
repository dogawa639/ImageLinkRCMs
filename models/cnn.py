from models.general import *

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import itertools

__all__ = ["CNN3x3", "CNN1x1"]


class CNN3x3(nn.Module):
    def __init__(self, patch_size, channels, act_fn=lambda x : x, residual=True, sn=False, sln=False, w_dim=None):
        # patch_size: (H, W)
        # channels: list
        # forward: (B, C, H, W)->(B, C', H, W)
        # sn: spectral normalization, sln: self-modulated layer normalization
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.patch_size = patch_size
        self.channels = channels
        self.act_fn = act_fn
        self.residual = residual
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim

        # convolution
        if not sn:
            self.covs = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], 3, padding=1, bias=False) for i in range(len(channels)-1)])  # (bs, channels[i], H, W)->(bs, channels[i+1], H, W)
            self.fc1 = nn.Conv2d(channels[-1], channels[-1]*2, 1, padding=0, bias=False)  # (bs, channels[-1], H, W)->(bs, channels[-1]*2, H, W)
        else:
            self.covs = nn.ModuleList([spectral_norm(nn.Conv2d(channels[i], channels[i+1], 3, padding=1, bias=False)) for i in range(len(channels)-1)])  # (bs, channels[i], H, W)->(bs, channels[i+1], H, W)
            self.fc1 = spectral_norm(nn.Conv2d(channels[-1], channels[-1]*2, 1, padding=0, bias=False))  # (bs, channels[-1], H, W)->(bs, channels[-1]*2, H, W)
        self.fc2 = nn.Conv2d(channels[-1]*2 + channels[0] * residual, channels[-1], 1, padding=0, bias=False)  #(bs, channels[-1]*2+channels[0], H, W)->(bs, channels[-1], H, W)

        if not sln:
            self.layer_norms = nn.ModuleList([nn.LayerNorm((channels[i], *self.patch_size)) for i in range(len(channels))])
        else:
            self.layer_norms = nn.ModuleList([SLN(w_dim, (channels[i], *self.patch_size)) for i in range(len(channels))])

        # layer_norm->conv->softplus->...
        self.sequence1 = nn.Sequential(*itertools.chain(*zip(self.layer_norms[:-1], self.covs, [Softplus()] * (len(channels)-1))))
        self.sequence2 = nn.Sequential(self.layer_norms[-1], self.fc1, Softplus())

    def forward(self, x, w=None):
        # x: (bs, channels[0], H, W)
        # w: (bs, w_dim)
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        
        if self.sln:
            for i in range(len(self.channels)):
                self.layer_norms[i].set_w(w)

        y = self.sequence2(self.sequence1(x))  # (bs, channels[-1]*2, H, W)
        if self.residual:
            return self.act_fn(self.fc2(torch.cat([y, x], dim=1)))
        else:
            return self.act_fn(self.fc2(y))

class CNN1x1(nn.Module):
    def __init__(self, patch_size, channels, act_fn=lambda x : x, sn=False, sln=False, w_dim=None):
        # patch_size: (H, W)
        # channels: list
        # forward: (B, C, H, W)->(B, C', H, W)
        # sn: spectral normalization, sln: self-modulated layer normalization
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.patch_size = patch_size
        self.channels = channels
        self.act_fn = act_fn
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim

        # convolution
        if not sn:
            self.covs = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], 1, padding=0, bias=False) for i in range(len(channels)-1)])  # (bs, channels[i], H, W)->(bs, channels[i+1], H, W)
        else:
            self.covs = nn.ModuleList([spectral_norm(nn.Conv2d(channels[i], channels[i+1], 1, padding=0, bias=False)) for i in range(len(channels)-1)])  # (bs, channels[i], H, W)->(bs, channels[i+1], H, W)

        if not sln:
            self.layer_norms = nn.ModuleList([nn.LayerNorm((channels[i], *self.patch_size)) for i in range(len(channels)-1)])
        else:
            self.layer_norms = nn.ModuleList([SLN(w_dim, (channels[i], *self.patch_size)) for i in range(len(channels)-1)])

        # layer_norm->conv->softplus->...
        self.sequence1 = nn.Sequential(*itertools.chain(*zip(self.layer_norms[:-1], self.covs[:-1], [Softplus()] * (len(channels)-2))))
        self.sequence2 = nn.Sequential(self.layer_norms[-1], self.covs[-1])

    def forward(self, x, w=None):
        # x: (bs, channels[0], H, W)
        # w: (bs, w_dim)
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        
        if self.sln:
            for i in range(len(self.channels)-1):
                self.layer_norms[i].set_w(w)

        return self.act_fn(self.sequence2(self.sequence1(x)))  # (bs, channels[-1], H, W)


# test
if __name__ == "__main__":
    b, c, h, w = 2, 3, 4, 5
    w_dim = 6
    device = "mps"
    inputs = torch.randn(b, c, h, w).to(device)
    ws = torch.randn(b, 6).to(device)

    cnn3 = CNN3x3((h, w), [c, c*2, c*4], act_fn=lambda x : -softplus(x), residual=True, sn=True, sln=True, w_dim=w_dim).to(device)
    out3 = cnn3(inputs, ws)

    cnn1 = CNN1x1((h, w), [c, c*2, c*4], act_fn=lambda x : -softplus(x), sn=False, sln=True, w_dim=w_dim).to(device)
    out1 = cnn1(inputs, ws)

    print(out3.shape, out1.shape)


