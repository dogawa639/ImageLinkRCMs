from models.general import *

import torch
from torch import tensor, nn
import torch.nn.functional as F

__all__ = ["CNN2L", "CNN2LDepth", "CNN2LPositive", "CNN2LNegative"]


class CNN2L(nn.Module):
    def __init__(self, input_channel, output_channel, residual=True, sln=False, w_dim=None):
        # forward: (B, C, 3, 3)->(B, C', 3, 3)
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.residual = residual
        self.sln = sln
        self.w_dim = w_dim

        # convolution
        self.cov1 = nn.Conv2d(self.input_channel, self.input_channel*8, 3, padding=1, bias=False)#(bs, in_channel, 3, 3)->(bs, in_channel*8, 3, 3)
        self.cov2 = nn.Conv2d(self.input_channel*8, self.input_channel*16, 3, padding=1, bias=False)#(bs, in_channel*8, 3, 3)->(bs, in_channel*32, 3, 3)
        # fully connected(1x1 convolution)
        self.fc1 = nn.Conv2d(self.input_channel*16, self.input_channel, 1, padding=0, bias=False)  #(bs, in_channel*16, 3, 3)->(bs, in_channel, 3, 3)
        self.fc2 = nn.Conv2d(self.input_channel + self.input_channel * residual, self.output_channel, 1, padding=0, bias=False)  #(bs, in_channel*2, 3, 3)->(bs, oc, 3, 3)

        if not sln:
            self.layer_norm1 = nn.LayerNorm(self.in_channel)
            self.layer_norm2 = nn.LayerNorm(self.in_channel)
        else:
            self.layer_norm1 = SLN(self.w_dim, self.in_channel)
            self.layer_norm2 = SLN(self.w_dim, self.in_channel)

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


class CNN2LDepth(nn.Module):
    def __init__(self, input_channel, output_channel, sln=False, w_dim=None):
        # forward: (B, C, 3, 3)->(B, C', 3, 3)
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.sln = sln
        self.w_dim = w_dim

        if not sln:
            self.layer_norm = nn.LayerNorm(self.in_channel)
        else:
            self.layer_norm = SLN(self.w_dim, self.in_channel)

        self.cov1 = nn.Conv2d(self.input_channel, self.input_channel*16, 1, padding=0, bias=False)
        self.cov2 = nn.Conv2d(self.input_channel*16, self.output_channel, 1, padding=0, bias=False)

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

