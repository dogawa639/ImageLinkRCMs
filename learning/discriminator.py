import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.cnn import CNN3x3, CNN1x1
from models.transformer import TransformerEncoder
from models.gnn import GT, GAT
from models.general import FF, SLN, Softplus

import numpy as np


class CNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, 
                 gamma=0.9, max_num=40, sln=True, w_dim=10, ext_coeff=1.0):
        super().__init__()
        self.nw_data = nw_data
        self.output_channel = output_channel
        self.gamma = gamma
        self.max_num = max_num
        self.ext_coeff = ext_coeff

        self.feature_num = self.nw_data.feature_num
        self.context_num = self.nw_data.context_feature_num
        self.total_feature = self.feature_num + self.context_num

        self.util = CNN3x3((3, 3), (self.total_feature, self.total_feature*2, output_channel), act_fn=lambda x : -Softplus(x), residual=True, sn=True, sln=sln, w_dim=w_dim)

        self.ext = CNN3x3((3, 3), (self.feature_num+output_channel, (self.feature_num+output_channel)*2, output_channel), act_fn=lambda x : -Softplus(x), residual=True, sn=True, sln=sln, w_dim=w_dim)

        self.val = CNN1x1((3, 3), (self.total_feature, self.total_feature*2, output_channel), act_fn=lambda x : -Softplus(x), residual=True, sn=True, sln=sln, w_dim=w_dim)

    def forward(self, input, pi, i=None):
        # input: (sum(links), total_feature, 3, 3)
        # pi: (sum(links), oc, 3, 3)
        # output: (sum(links), 3, 3)
        # model output: (sum(links), oc, 3, 3)
        ext_input = torch.cat((input[:, :self.feature_num, :, :], pi), dim=1)
        util = self.util(input)
        ext = self.ext(ext_input)
        val = self.val(input)

        f_val = util + self.ext_coeff * ext + self.gamma * val - val[:, :, 1, 1].view(-1, self.output_channel, 1, 1)
        if i is None:
            return f_val
        return f_val[:, i, :, :]

    def get_vals(self, input, pi, i):
        ext_input = torch.cat((input[:, :self.feature_num, :, :], pi), dim=1)
        util = self.util(input)
        ext = self.ext(ext_input)
        val = self.val(input)

        f_val = util + self.ext_coeff * ext + self.gamma * val - val[:, :, 1, 1].view(-1, self.output_channel, 1, 1)
        return f_val[:, i, :, :], util[:, i, :, :], ext[:, i, :, :], val[:, i, :, :]

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnndis.pth"))


class GNNDis(nn.Module):
    def __init__(self, nw_data, emb_dim, output_channel, enc_dim, 
                 gamma=0.9, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None, ext_coeff=1.0):
        super().__init__()
        self.nw_data = nw_data
        self.emb_dim = emb_dim
        self.feature_num = nw_data.feature_num
        self.output_channel = output_channel
        self.adj_matrix = tensor(nw_data.edge_graph).to_sparse()
        self.gamma = gamma
        self.sln = sln
        self.w_dim = w_dim

        kargs_local = {
            "enc_dim": enc_dim,
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": 1,  # depth=1
            "pre_norm": pre_norm, 
            "output_atten": True, 
            "sn": sn, 
            "sln": sln, 
            "w_dim": w_dim
            }
        kwargs_grobal = {
            "enc_dim": enc_dim, 
            "in_emb_dim": in_emb_dim,
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": depth,  # depth=depth
            "pre_norm": pre_norm, 
            "output_atten": True, 
            "sn": sn, 
            "sln": sln, 
            "w_dim": w_dim
            } 

        self.gnn_local = nn.ModuleList([GAT(self.feature_num, self.emb_dim, self.adj_matrix, **kargs_local) for _ in range(output_channel)])
        self.gnn_grobal = nn.ModuleList([GAT(self.feature_num, self.emb_dim, self.adj_matrix, **kwargs_grobal) for _ in range(output_channel)])

        self.util = FF(self.emb_dim*2, 1, self.emb_dim*2, act_fn=lambda x : -Softplus(x), sn=sn, sln=sln, w_dim=w_dim)  # grobal

        self.ext = FF(self.emb_dim*2+output_channel, 1, (self.emb_dim+output_channel)*2, act_fn=lambda x : -Softplus(x), sn=sn, sln=sln, w_dim=w_dim)  # local + pi

        self.val = FF(self.emb_dim, 1, self.emb_dim*2, act_fn=lambda x : -Softplus(x), sn=sn, sln=sln, w_dim=w_dim)  # grobal

    def forward(self, x, bs, pi, i=None, enc=None, w=None):
        # x: (trip_num, link_num, feature_num) or (link_num, feature_num)
        # pi: (trip_num, link_num, link_num, oc)
        # enc: (trip_num, link_num, enc_dim) positional encoding
        # output: (trip_num, link_num, link_num)
        if enc is not None:
            if enc.dim() == 2:
                bs = 1
                enc = enc.unsqueeze(0)
            elif enc.dim() == 3:
                bs = enc.shape[0]
            else:
                raise Exception("enc should be 2 or 3 dim")
        x_rep = x.expand(bs, x.shape[-2], x.shape[-1])
        if i is None:
            f_val = None
            for j in range(self.output_channel):
                f_val_j = self.get_vals(x_rep, pi, j, enc, w)[0].unsqueeze(1)
                if f_val is None:
                    f_val = f_val_j
                else:
                    f_val = torch.cat((f_val, f_val_j), dim=1)
            return f_val[:, i, :, :]
        return self.get_vals(x_rep, pi, i, enc, w)[0]
    
    def get_vals(self, x_rep, pi, i, enc, w):
        # (f_val, util, ext, val)
        # f_val, util, ext: (bs, link_num, link_num)
        # val: (bs, link_num)
        y = self.gnn_grobal[i](x_rep, enc, w)[0]  # (bs, link_num, emb_dim)

        val = self.val(y).squeese(-1)  # (bs, link_num)
        
        y = y.unsqueeze(-2)  # (bs, link_num, 1, emb_dim)
        z = torch.cat((y, y.transpose(-3, -2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        util = self.util(z).squeese(-1) * self.adj_matrix.view(1, *self.adj_matrix.shape)  # (bs, link_num, link_num)

        y = self.gnn_local[i](x_rep, enc, w)[0].unsqueeze(-2)  # (bs, link_num, 1, emb_dim)
        z = torch.cat((y, y.transpose(-3, -2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        ext = self.ext(torch.cat((z, pi), dim=-1)).squeese(-1) * self.adj_matrix.view(1, *self.adj_matrix.shape)  # (bs, link_num, link_num)

        f_val = util + self.ext_coeff * ext + self.gamma * val.unsqueeze(-2) - val.unsqueeze(-1)
        return f_val, util, ext, val


# test
if __name__ == "__main__":
    from preprocessing.network_processing import *

    device = "mps"
    node_path = '/Users/dogawa/Desktop/bus/estimation/data/node.csv'
    link_path = '/Users/dogawa/Desktop/bus/estimation/data/link.csv'
    link_prop_path = '/Users/dogawa/Desktop/bus/estimation/data/link_attr_min.csv'
    model_dir = "/Users/dogawa/PycharmProjects/GANs/trained_models"
    bs = 3
    input_channel = 5
    output_channel = 2
    w_dim = 5
    enc_dim = 3
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    f = nw_data.feature_num
    c = nw_data.context_feature_num

    dis = CNNDis(nw_data, output_channel).to(device)

    inputs = torch.randn(bs, f+c, 3, 3).to(device)
    out = dis(inputs, 0)
    out2 = dis(inputs, 1)
    print(out.shape, out2.shape)

    dis = GNNDis(nw_data, output_channel, enc_dim, sln=True, w_dim=w_dim).to(device)
    inputs = torch.randn(nw_data.link_num, nw_data.feature_num).to(device)
    w = torch.randn(bs, w_dim).to(device)
    enc = torch.randn(bs, nw_data.link_num, enc_dim).to(device)
    out = dis(inputs, bs, 0, enc, w=w)
    print(out.shape)



