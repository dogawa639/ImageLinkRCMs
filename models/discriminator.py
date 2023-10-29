import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from cnn import CNN2LNegative, CNN2LDepth
from transformer import Transformer
from general import FF, SLN

import numpy as np


class CNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, gamma=0.9, max_num=40, device="cpu"):
        self.nw_data = nw_data
        self.output_channel = output_channel
        self.gamma = gamma
        self.max_num = max_num
        self.device = device

        self.input_feature = self.nw_data.link_feature_num + self.nw_data.context_feature_num
        self.util = CNN2LNegative(self.input_feature, self.output_channel, sn=True)
        self.ext = CNN2LNegative(self.input_feature, self.output_channel, sn=True)
        self.val = CNN2LDepth(self.input_feature, self.output_channel, sn=True)

    def forward(self, input, i):
        # input: (sum(links), input_feature, 3, 3)
        # output: (sum(links), 3, 3)
        # model output: (sum(links), oc, 3, 3)
        return (self.util(input) + self.ext(input) + self.gamma * self.val(input)
                - self.val(input[:, :, 1, 1].view(input.shape[0], input.shape[1], 1, 1)))[:, i, :, :]

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnndis.pth"))


class GNNDis(nn.Module):
    def __init__(self, in_channel, output_channel, enc_dim, gamma=0.9, device="cpu", h_dim=1, sln=False, w_dim=None):
        super().__init__()
        self.in_channel = in_channel
        self.output_channel = output_channel
        self.enc_dim = enc_dim
        self.gamma = gamma
        self.device = device
        self.h_dim = h_dim
        self.sln = sln
        self.w_dim = w_dim

        if self.sln:
            self.ff0 = FF(h_dim, w_dim, bias=True)
        self.util = Transformer(in_channel, output_channel, k=3, dropout=0.1, depth=3, residual=True, sn=True, sln=sln,
                                       w_dim=w_dim)
        self.ext = Transformer(in_channel, output_channel, k=3, dropout=0.1, depth=3, residual=True, sn=True, sln=sln,
                                        w_dim=w_dim)
        self.val = Transformer(in_channel, 1, k=3, dropout=0.1, depth=3, residual=True, sn=True, sln=sln,
                                        w_dim=w_dim)

    def forward(self, x, num, i, enc, w):
        # x: (link_num, in_channel)
        # enc: (trip_num, enc_dim)
        # output: (trip_num, link_num, link_num)
        x_rep = x.expand(num, x.shape[0], x.shape[1])
        if self.sln:
            return self.transformer(x_rep, enc, w)[:, i, :, :]
        else:
            return self.transformer(x, enc)[:, i, :, :]


