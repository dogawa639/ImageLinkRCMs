from models.general import *

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["CNN2L", "CNN2LDepth", "CNN2LPositive", "CNN2LNegative"]


class CNN2L(nn.Module):
    def __init__(self, input_channel, output_channel, residual=True, sn=False, sln=False, w_dim=None, device="cpu"):
        # forward: (B, C, 3, 3)->(B, C', 3, 3)
        # sn: spectral normalization, sln: self-modulated layer normalization
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.residual = residual
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim
        self.device = device

        # convolution
        if not sn:
            self.cov1 = nn.Conv2d(self.input_channel, self.input_channel*8, 3, padding=1, bias=False) #(bs, in_channel, 3, 3)->(bs, in_channel*8, 3, 3)
            self.cov2 = nn.Conv2d(self.input_channel*8, self.input_channel*16, 3, padding=1, bias=False) #(bs, in_channel*8, 3, 3)->(bs, in_channel*32, 3, 3)
            # fully connected(1x1 convolution)
            self.fc1 = nn.Conv2d(self.input_channel*16, self.input_channel, 1, padding=0, bias=False) #(bs, in_channel*16, 3, 3)->(bs, in_channel, 3, 3)
        else:
            self.cov1 = spectral_norm(nn.Conv2d(self.input_channel, self.input_channel*8, 3, padding=1, bias=False)) #(bs, in_channel, 3, 3)->(bs, in_channel*8, 3, 3)
            self.cov2 = spectral_norm(nn.Conv2d(self.input_channel*8, self.input_channel*16, 3, padding=1, bias=False)) #(bs, in_channel*8, 3, 3)->(bs, in_channel*32, 3, 3)
            # fully connected(1x1 convolution)
            self.fc1 = spectral_norm(nn.Conv2d(self.input_channel*16, self.input_channel, 1, padding=0, bias=False)) #(bs, in_channel*16, 3, 3)->(bs, in_channel, 3, 3)
        self.fc2 = nn.Conv2d(self.input_channel + self.input_channel * residual, self.output_channel, 1, padding=0, bias=False)  #(bs, in_channel*2, 3, 3)->(bs, oc, 3, 3)

        if not sln:
            self.layer_norm1 = nn.LayerNorm((self.input_channel, 3, 3))
            self.layer_norm2 = nn.LayerNorm((self.input_channel*16, 3, 3))
        else:
            self.layer_norm1 = SLN(self.w_dim, self.input_channel)
            self.layer_norm2 = SLN(self.w_dim, self.input_channel*16)

        self.sequence1 = nn.Sequential(
            self.cov1,
            Softplus(),
            nn.AvgPool2d(3, stride=1, padding=1),
            self.cov2,
            Softplus(),
            nn.AvgPool2d(3, stride=1, padding=1)
        )
        self.sequence2 = nn.Sequential(
            self.fc1,
            Softplus()
        )
        self.sequence3 = nn.Sequential(
            self.fc2
        )

        self.to(self.device)

    def forward(self, x, w=None):
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        x_reshape = x.reshape(-1, self.input_channel, 9).transpose(1, 2)  # (bs, 9, c)
        if self.sln:
            # w: (bs, w_dim)
            w_reshape = w.unsqueeze(1).repeat(1, 9, 1)
        if self.sln:
            x_norm = self.layer_norm1(x_reshape, w_reshape).transpose(1, 2)
        else:
            x_norm = self.layer_norm1(x).transpose(1, 2)
        y = self.sequence1(x_norm.reshape(-1, self.input_channel, 3, 3))

        y_reshape = y.reshape(-1, self.input_channel*16, 9).transpose(1, 2)  # (bs, 9, c)
        if self.sln:
            y_norm = self.layer_norm2(y_reshape, w_reshape)
        else:
            y_norm = self.layer_norm2(y_reshape)
        z = self.sequence2(y_norm.transpose(1, 2).reshape(-1, self.input_channel*16, 3, 3))

        if self.residual:
            return self.sequence3(torch.cat([z, x], dim=1))
        else:
            return self.sequence3(y)


class CNN2LDepth(nn.Module):
    def __init__(self, input_channel, output_channel, sn=False, sln=False, w_dim=None):
        # forward: (B, C, 3, 3)->(B, C', 3, 3)
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.sn = sn
        self.sln = sln
        self.w_dim = w_dim

        if not sln:
            self.layer_norm = nn.LayerNorm((self.input_channel, 3, 3))
        else:
            self.layer_norm = SLN(self.w_dim, self.input_channel)

        if not sn:
            self.cov1 = nn.Conv2d(self.input_channel, self.input_channel*16, 1, padding=0, bias=False)
            self.cov2 = nn.Conv2d(self.input_channel*16, self.output_channel, 1, padding=0, bias=False)
        else:
            self.cov1 = spectral_norm(nn.Conv2d(self.input_channel, self.input_channel*16, 1, padding=0, bias=False))
            self.cov2 = spectral_norm(nn.Conv2d(self.input_channel*16, self.output_channel, 1, padding=0, bias=False))

        self.sequence = nn.Sequential(
            self.cov1,
            self.cov2
        )

    def forward(self, x, w=None):
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        if self.sln:
            x_norm = self.layer_norm(x, w)
        else:
            x_norm = self.layer_norm(x)
        return self.sequence(x_norm)


class CNN2LPositive(CNN2L):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.sequence3 = nn.Sequential(
            self.fc2,
            Softplus()
        )

    def forward(self, x, w=None):
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        if self.sln:
            x_norm = self.layer_norm1(x, w)
        else:
            x_norm = self.layer_norm1(x)
        y = self.sequence1(x_norm)

        if self.sln:
            y_norm = self.layer_norm2(y, w)
        else:
            y_norm = self.layer_norm2(y)
        y = self.sequence2(y_norm)

        if self.residual:
            return self.sequence3(torch.cat([y, x], dim=1))
        else:
            return self.sequence3(y)
    

class CNN2LNegative(CNN2L):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.sequence3 = nn.Sequential(
            self.fc2,
            Softplus()
        )

    def forward(self, x, w=None):
        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        if self.sln:
            x_norm = self.layer_norm1(x, w)
        else:
            x_norm = self.layer_norm1(x)
        y = self.sequence1(x_norm)

        if self.sln:
            y_norm = self.layer_norm2(y, w)
        else:
            y_norm = self.layer_norm2(y)
        y = self.sequence2(y_norm)

        if self.residual:
            return -self.sequence3(torch.cat([y, x], dim=1))
        else:
            return -self.sequence3(y)

