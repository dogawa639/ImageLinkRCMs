from models.general import *

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import itertools

__all__ = ["CNN3x3", "CNN1x1", "ResNet"]


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
        self.sequence1 = nn.Sequential(*itertools.chain(*zip(self.layer_norms[:-1], self.covs, [Softplus()] * (len(channels) - 1))))
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


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, sn=False, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        if sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act_fn1 = nn.ReLU()
        self.act_fn2 = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

        self.sequence = nn.Sequential(
                self.conv1,
                self.bn1,
                self.act_fn1,
                self.dropout,
                self.conv2,
                self.bn2,
                self.act_fn2
        )

    def forward(self, x):
        return self.sequence(x)


class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes, depth=4, sn=False, dropout=0.0, act_fn=lambda x : x):
        # pool_type: none, max or avg
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.act_fn = act_fn
        self.depth = depth

        self.conv0 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.blocks = nn.ModuleList([])
        self.identitys = nn.ModuleList([])
        for i in range(depth):
            self.blocks.append(BaseConv(64 * 2 ** i, 128 * 2 ** i, sn=sn, dropout=dropout))  # (128 * 2^i, H/2^(i+1), W/2^(i+1))
            self.identitys.append(nn.Conv2d(64 * 2 ** i, 128 * 2 ** i, 1, bias=False))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_fn = nn.Linear(128 * 2 ** (depth-1), num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.act_fn(x)
        x = self.maxpool(x)
        for i in range(self.depth):
            x = self.blocks[i](x) + self.identitys[i](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.out_fn(x)
        return x


