import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.cnn import CNN3x3, CNN1x1
from models.transformer import TransformerEncoder
from models.gnn import GT, GAT
from models.general import FF, SLN, Softplus, softplus

import numpy as np


# CNN
# input: (sum(links), total_feature, 3, 3), pi: (sum(links), oc, 3, 3)
# output: (sum(links), 3, 3) or (sum(links), oc, 3, 3)
# GNN
# input: (bs, link_num, feature_num) or (link_num, feature_num), pi: (trip_num, link_num, link_num, oc)
# output: (bs, link_num, link_num) or (trip_num, oc, link_num, link_num)
class CNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, 
                 image_feature_num=0, gamma=0.9, max_num=40, sn=True, sln=True, w_dim=10, ext_coeff=1.0):
        super().__init__()
        self.nw_data = nw_data
        self.output_channel = output_channel
        self.image_feature_num = image_feature_num
        self.gamma = gamma
        self.max_num = max_num
        self.ext_coeff = ext_coeff

        self.feature_num = self.nw_data.feature_num
        self.context_num = self.nw_data.context_feature_num
        self.total_feature = self.feature_num + self.context_num + image_feature_num

        self.util = CNN3x3((3, 3), (self.total_feature, self.total_feature*2, output_channel), act_fn=lambda x : -softplus(x), residual=True, sn=sn, sln=sln, w_dim=w_dim)

        self.ext = CNN3x3((3, 3), (self.feature_num+self.image_feature_num+output_channel, (self.feature_num+self.image_feature_num+output_channel)*2, output_channel), act_fn=lambda x : -softplus(x), residual=True, sn=sn, sln=sln, w_dim=w_dim)

        self.val = CNN1x1((3, 3), (self.total_feature, self.total_feature*2, output_channel), act_fn=lambda x : -softplus(x), sn=sn, sln=sln, w_dim=w_dim)

        self.w = None

    def forward(self, input, pi, w=None, i=None):
        # input: (sum(links), total_feature, 3, 3)
        # pi: (sum(links), oc, 3, 3)
        # output: (sum(links), 3, 3) or (sum(links), oc, 3, 3)
        if w is not None:
            self.w = w
        ext_input = torch.cat((input[:, :self.feature_num, :, :], input[:, self.feature_num+self.context_num:, :, :], pi), dim=1)
        util = self.util(input, w=self.w)
        ext = self.ext(ext_input, w=self.w)
        val = self.val(input, w=self.w)

        f_val = util + self.ext_coeff * ext + self.gamma * val - val[:, :, 1, 1].view(-1, self.output_channel, 1, 1)
        if i is None:
            return f_val
        return f_val[:, i, :, :]

    def get_vals(self, input, pi, i=None, w=None):
        if w is not None:
            self.w = w
        ext_input = torch.cat((input[:, :self.feature_num, :, :], input[:, self.feature_num+self.context_num:, :, :], pi), dim=1)
        util = self.util(input, w=self.w)
        ext = self.ext(ext_input, w=self.w)
        val = self.val(input, w=self.w)

        f_val = util + self.ext_coeff * ext + self.gamma * val - val[:, :, 1, 1].view(-1, self.output_channel, 1, 1)
        if i is None:
            return f_val, util, ext, val
        return f_val[:, i, :, :], util[:, i, :, :], ext[:, i, :, :], val[:, i, :, :]

    def set_w(self, w):
        # w : (w_dim)
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnndis.pth"))


class GNNDis(nn.Module):
    def __init__(self, nw_data, emb_dim, output_channel,
                 image_feature_num=0, gamma=0.9, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, sn=False, sln=False, w_dim=None, ext_coeff=1.0):
        super().__init__()
        self.nw_data = nw_data
        self.emb_dim = emb_dim
        self.feature_num = nw_data.feature_num
        self.output_channel = output_channel
        self.adj_matrix = nn.parameter.Parameter(tensor(nw_data.edge_graph).to(torch.float32).to_dense(), requires_grad=False)
        self.image_feature_num = image_feature_num
        self.gamma = gamma
        self.sln = sln
        self.w_dim = w_dim
        self.ext_coeff = ext_coeff

        kwargs_local = {
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": 1,  # depth=1
            "output_atten": True, 
            "sn": sn, 
            "sln": sln, 
            "w_dim": w_dim
            }
        kwargs_global = {
            "in_emb_dim": in_emb_dim,
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": depth,  # depth=depth
            "output_atten": True, 
            "sn": sn, 
            "sln": sln, 
            "w_dim": w_dim
            } 

        self.gnn_local = nn.ModuleList([GAT(self.feature_num + self.image_feature_num, self.emb_dim, self.adj_matrix, **kwargs_local) for _ in range(output_channel)])
        self.gnn_global = nn.ModuleList([GAT(self.feature_num + self.image_feature_num, self.emb_dim, self.adj_matrix, **kwargs_global) for _ in range(output_channel)])

        self.util = FF(self.emb_dim*2, 1, self.emb_dim*2, act_fn=lambda x : -softplus(x), sn=sn)  # global

        self.ext = FF(self.emb_dim*2+output_channel, 1, (self.emb_dim+output_channel)*2, act_fn=lambda x : -softplus(x), sn=sn)  # local + pi

        self.val = FF(self.emb_dim, 1, self.emb_dim*2, act_fn=lambda x : -softplus(x), sn=sn)  # global

        self.w = None

    def forward(self, x, pi, w=None, i=None):
        # x: (trip_num, link_num, feature_num) or (link_num, feature_num)
        # pi: (trip_num, oc, link_num, link_num)
        # output: f_val (trip_num, link_num, link_num)
        if w is not None:
            self.w = w
        bs = pi.shape[0]
        if x.dim() == 2:
            x_rep = x.expand(bs, x.shape[-2], x.shape[-1])
        else:
            x_rep = x
        if i is None:
            f_val = None
            for j in range(self.output_channel):
                f_val_j = self.get_vals(x_rep, pi, j, w=w)[0].unsqueeze(1)
                if f_val is None:
                    f_val = f_val_j
                else:
                    f_val = torch.cat((f_val, f_val_j), dim=1)
            return f_val
        return self.get_vals(x_rep, pi, i, w)[0]
    
    def get_vals(self, x_rep, pi, i, w=None):
        # (f_val, util, ext, val)
        # f_val, util, ext: (bs, link_num, link_num)
        # val: (bs, link_num, 1)
        if w is not None:
            self.w = w
        y = self.gnn_global[i](x_rep, w=self.w)[0]  # (bs, link_num, emb_dim)

        val = self.val(y).squeeze(-1)  # (bs, link_num)

        n = y.shape[1]
        y2= y.unsqueeze(-2).expand(-1, n, n, -1)  # (bs, link_num, link_num, emb_dim)
        z = torch.cat((y2, y2.transpose(-3, -2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        util = self.util(z).squeeze(-1) * self.adj_matrix.view(1, *self.adj_matrix.shape)  # (bs, link_num, link_num)

        y3 = self.gnn_local[i](x_rep, w=self.w)[0].unsqueeze(-2).expand(-1, n, n, -1)  # (bs, link_num, link_num, emb_dim)
        z2 = torch.cat((y3, y3.transpose(-3, -2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        ext = self.ext(torch.cat((z2, pi.transpose(1, -1)), dim=-1)).squeeze(-1) * self.adj_matrix.view(1, *self.adj_matrix.shape)  # (bs, link_num, link_num)

        f_val = util + self.ext_coeff * ext + self.gamma * val.unsqueeze(-2) - val.unsqueeze(-1)
        return f_val, util, ext, val.unsqueeze(-1)

    def set_w(self, w):
        # w : (trip_num, w_dim)
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/gnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/gnndis.pth"))




