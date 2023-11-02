import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from cnn import CNN2LNegative, CNN2LDepth
from transformer import Transformer
from general import FF, SLN

import numpy as np


class CNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, gamma=0.9, max_num=40):
        super().__init__()
        self.nw_data = nw_data
        self.output_channel = output_channel
        self.gamma = gamma
        self.max_num = max_num

        self.input_feature = self.nw_data.feature_num + self.nw_data.context_feature_num
        self.util = CNN2LNegative(self.input_feature, self.output_channel, sn=True)
        self.ext = CNN2LNegative(self.input_feature, self.output_channel, sn=True)
        self.val = CNN2LDepth(self.input_feature, self.output_channel, sn=True)

    def forward(self, input, i):
        # input: (sum(links), input_feature, 3, 3)
        # output: (sum(links), 3, 3)
        # model output: (sum(links), oc, 3, 3)
        return (self.util(input) + self.ext(input) + self.gamma * self.val(input)
                - self.val(input)[:, :, 1, 1].view(-1, self.output_channel, 1, 1))[:, i, :, :]

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnndis.pth"))


class GNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, enc_dim, gamma=0.9, h_dim=1, sln=False, w_dim=None):
        super().__init__()
        self.feature_num = nw_data.feature_num
        self.link_num = nw_data.link_num
        self.output_channel = output_channel
        self.enc_dim = enc_dim
        self.gamma = gamma
        self.h_dim = h_dim
        self.sln = sln
        self.w_dim = w_dim

        if self.sln:
            self.ff0 = FF(h_dim, w_dim, h_dim*2, bias=True)
        kargs = {"enc_dim": enc_dim, "k": 3, "dropout": 0.1, "depth": 3, "residual": True, "sn": True, "sln": sln, "w_dim": w_dim}
        self.util = nn.ModuleList([Transformer(self.feature_num, self.link_num, **kargs) for _ in range(output_channel)])
        self.ext = nn.ModuleList([Transformer(self.feature_num, self.link_num, **kargs) for _ in range(output_channel)])
        self.val = Transformer(self.feature_num, output_channel, **kargs)

    def forward(self, x, num, i, enc=None, w=None):
        # x: (link_num, in_channel)
        # enc: (trip_num, link_num, enc_dim) positional encoding
        # output: (trip_num, link_num, link_num)
        if enc is not None:
            num = enc.shape[0]
        x_rep = x.expand(num, x.shape[0], x.shape[1])

        v_val = self.val(x_rep, enc, w)[:, :, [i]]
        f_val = self.util[i](x_rep, enc, w) + self.ext(x_rep, enc, w)[i] + self.gamma * v_val.transpose(1, 2) - v_val
        return f_val

# test
if __name__ == "__main__":
    from preprocessing.network_processing import *

    device = "mps"
    node_path = '/Users/dogawa/Desktop/bus/estimation/data/node.csv'
    link_path = '/Users/dogawa/Desktop/bus/estimation/data/link.csv'
    link_prop_path = '/Users/dogawa/Desktop/bus/estimation/data/link_attr_min.csv'
    model_dir = "/Users/dogawa/PycharmProjects/GANs/trained_models"
    input_channel = 5
    output_channel = 2
    w_dim = 5
    enc_dim = 3
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    f = nw_data.feature_num
    c = nw_data.context_feature_num

    dis = CNNDis(nw_data, output_channel).to(device)

    inputs = torch.randn(10, f+c, 3, 3).to(device)
    out = dis(inputs, 0)
    out2 = dis(inputs, 1)
    print(out.shape, out2.shape)

    dis = GNNDis(nw_data, output_channel, enc_dim, sln=True, w_dim=w_dim).to(device)
    inputs = torch.randn(10, nw_data.feature_num).to(device)
    w = torch.randn(10, w_dim).to(device)
    enc = torch.randn(10, enc_dim).to(device)
    out = dis(inputs, 10, 0, enc, w=w)
    print(out.shape)



