import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.cnn import CNN3x3, CNN1x1
from models.transformer import TransformerEncoder
from models.gnn import GT, GAT
from models.unet import UNet
from models.general import FF, SLN, softplus

import numpy as np
__all__ = ["CNNDis", "GNNDis", "UNetDis", "UNetDisStatic"]


# CNN
# input: (sum(links), total_feature, 3, 3), pi: (sum(links), oc, 3, 3)
# output: (sum(links), 3, 3) or (sum(links), oc, 3, 3)
# GNN
# input: (bs, link_num, feature_num) or (link_num, feature_num), pi: (trip_num, link_num, link_num, oc)
# output: (bs, link_num, link_num) or (trip_num, oc, link_num, link_num)
class CNNDis(nn.Module):
    def __init__(self, nw_data, output_channel, 
                 image_feature_num=0, gamma=0.9, sn=True, sln=True, w_dim=10, ext_coeff=1.0):
        super().__init__()
        self.nw_data = nw_data
        self.output_channel = output_channel
        self.image_feature_num = image_feature_num
        self.gamma = gamma
        self.ext_coeff = ext_coeff

        self.feature_num = self.nw_data.feature_num
        self.context_num = self.nw_data.context_feature_num
        self.total_feature = self.feature_num + self.context_num + image_feature_num

        self.util = CNN3x3((3, 3), (self.total_feature, self.total_feature*2, output_channel), act_fn=lambda x : -softplus(x), residual=True, sn=sn, sln=sln, w_dim=w_dim)

        self.ext = CNN3x3((3, 3), (self.feature_num+self.image_feature_num+output_channel, (self.feature_num+self.image_feature_num+output_channel)*2, output_channel), act_fn=lambda x : -softplus(x), residual=True, sn=True, sln=sln, w_dim=w_dim)

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

    def get_f_val(self, util, ext, val):
        if util.dim() == 3:
            return util + self.ext_coeff * ext + self.gamma * val - val[:, 1, 1].view(-1, 1, 1)
        return util + self.ext_coeff * ext + self.gamma * val - val[:, :, 1, 1].view(-1, self.output_channel, 1, 1)

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
        self.context_num = nw_data.context_feature_num
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

        self.gnn_local = nn.ModuleList([GAT(self.feature_num + self.context_num + self.image_feature_num, self.emb_dim, self.adj_matrix, **kwargs_local) for _ in range(output_channel)])
        self.gnn_global = nn.ModuleList([GAT(self.feature_num + self.context_num + self.image_feature_num, self.emb_dim, self.adj_matrix, **kwargs_global) for _ in range(output_channel)])

        self.util = FF(self.emb_dim*2, 1, self.emb_dim*2, act_fn=lambda x : -softplus(x), sn=sn)  # global

        self.ext = FF(self.emb_dim*2+output_channel, 1, (self.emb_dim+output_channel)*2, act_fn=lambda x : -softplus(x), sn=True)  # local + pi

        self.val = FF(self.emb_dim, 1, self.emb_dim*2, act_fn=lambda x : -softplus(x), sn=sn)  # global

        self.w = None

    def forward(self, x, pi, w=None, i=None):
        # x: (trip_num, link_num, feature_num + context_num + image_feature_num) or (link_num, feature_num + context_num + image_feature_num)
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

    def get_f_val(self, util, ext, val):
        return util + self.ext_coeff * ext + self.gamma * val.unsqueeze(-2) - val.unsqueeze(-1)

    def set_w(self, w):
        # w : (trip_num, w_dim)
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/gnndis.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/gnndis.pth"))


class UNetDis(nn.Module):
    # one discriminator for one transportation
    def __init__(self, feature_num, context_num, output_channel,
                 gamma=0.9, sn=True, dropout=0.0, ext_coeff=1.0, depth=2):
        super().__init__()
        # state : (bs, feature_num, 2d+1, 2d+1)
        # context : (bs, context_num, 2d+1, 2d+1)
        self.feature_num = feature_num  # contains output_channels (transportation features)
        self.context_num = context_num
        self.total_feature = feature_num + context_num
        self.output_channel = output_channel  # number of transportations
        self.gamma = gamma
        self.ext_coeff = ext_coeff

        self.util = UNet(self.total_feature, 1, sn=sn, dropout=dropout, depth=depth)
        self.ext = UNet(self.total_feature + output_channel * 2, 1, sn=True, dropout=dropout, depth=depth)  # state + other agent's (position + pi)
        self.val = UNet(self.total_feature, 1, sn=sn, dropout=dropout, depth=depth)

    def forward(self, input, positions, pis):
        # input: (bs, total_feature, 2d+1, 2d+1)
        # positions: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # pis: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        f_val, _, _, _ = self.get_vals(input, positions, pis)
        return f_val

    def get_vals(self, input, positions, pis):
        # input: (bs, total_feature, 2d+1, 2d+1)
        # positions: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # pis: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        if input.shape[-1] % 2 != 1:
            raise Exception("input.shape[-1] should be odd.")
        d = int((input.shape[-1] - 1) / 2)
        util = self.util(input).squeeze(1)
        val = self.val(input).squeeze(1)
        active_agent = positions.view(*positions.shape[:-3], -1).sum(dim=-1) > 0.0  # (bs, num_agents)
        ext_input = torch.cat((input.unsqueeze(1).expand(-1, positions.shape[1], -1, -1, -1), positions, pis), dim=2)  # (bs, num_agents, total_feature+output_channel*2, 2d+1, 2d+1)
        ext = self.ext(ext_input.view(-1, *ext_input.shape[2:])).view(ext_input.shape[0], ext_input.shape[1], ext_input.shape[3], ext_input.shape[4])  # (bs, num_agents, 2d+1, 2d+1)
        ext = ext * active_agent.unsqueeze(-1).unsqueeze(-1)

        f_val = util + self.ext_coeff * ext.sum(dim=1, keepdims=False) + self.gamma * val - val[:, d, d].view(-1, 1, 1)
        return f_val, util, ext, val

    def get_f_val(self, util, ext, val):
        d = int((util.shape[-1] - 1) / 2)
        return util + self.ext_coeff * ext.sum(dim=1, keepdims=False) + self.gamma * val - val[:, d, d].view(-1, 1, 1)

    def save(self, model_dir, i=None):
        if i is None:
            torch.save(self.state_dict(), model_dir + "/unetdis.pth")
        else:
            torch.save(self.state_dict(), model_dir + "/unetdis_{}.pth".format(i))

    def load(self, model_dir, i=None):
        if i is None:
            self.load_state_dict(torch.load(model_dir + "/unetdis.pth"))
        else:
            self.load_state_dict(torch.load(model_dir + "/unetdis_{}.pth".format(i)))


class UNetDisStatic(nn.Module):
    # one discriminator for one transportation
    def __init__(self, feature_num, context_num, output_channel,
                 gamma=0.9, sn=True, dropout=0.0, ext_coeff=1.0, depth=1):
        super().__init__()
        # state : (bs, feature_num, 2d+1, 2d+1)
        # context : (bs, context_num, 2d+1, 2d+1)
        self.feature_num = feature_num  # contains output_channels (transportation features)
        self.context_num = context_num
        self.total_feature = feature_num + context_num
        self.output_channel = output_channel  # number of transportations
        self.gamma = gamma
        self.ext_coeff = ext_coeff

        pool_type = "none"
        self.util = UNet(self.total_feature, 1, sn=sn, pool_type=pool_type, dropout=dropout, depth=depth, act_fn=lambda x : -softplus(x))
        self.ext = UNet(self.total_feature + output_channel - 1, 1, sn=True, pool_type=pool_type, dropout=dropout, depth=depth, act_fn=lambda x : -softplus(x))  # state + other agent's pi
        self.val = UNet(self.total_feature, 1, sn=sn, dropout=dropout, pool_type=pool_type, depth=depth, act_fn=lambda x : -softplus(x))

    def forward(self, input, pi_other):
        # input: (bs, total_feature, 2d+1, 2d+1)
        # pi_other: (bs, output_channel-1, 2d+1, 2d+1)
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        f_val, _, _, _ = self.get_vals(input, pi_other)
        return f_val

    def get_vals(self, input, pi_other):
        # input: (bs, total_feature, 2d+1, 2d+1)
        # pi_other: (bs, output_channel-1, 2d+1, 2d+1)
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        if input.shape[-1] % 2 != 1:
            raise Exception("input.shape[-1] should be odd.")
        d = int((input.shape[-1] - 1) / 2)
        util = self.util(input).squeeze(1)
        val = self.val(input).squeeze(1)
        ext_input = torch.cat((input, pi_other), dim=1)  # (bs, total_feature+output_channel-1, 2d+1, 2d+1)
        ext = self.ext(ext_input).squeeze(1)  # (bs, 2d+1, 2d+1)

        f_val = util + self.ext_coeff * ext + self.gamma * val - val[:, d, d].view(-1, 1, 1)
        return f_val, util, ext, val  # (bs, 2d+1, 2d+1)

    def get_f_val(self, util, ext, val):
        d = int((util.shape[-1] - 1) / 2)
        return util + self.ext_coeff * ext + self.gamma * val - val[:, d, d].view(-1, 1, 1)

    def save(self, model_dir, i=None):
        if i is None:
            torch.save(self.state_dict(), model_dir + "/unetdisstat.pth")
        else:
            torch.save(self.state_dict(), model_dir + "/unetdisstat_{}.pth".format(i))

    def load(self, model_dir, i=None):
        if i is None:
            self.load_state_dict(torch.load(model_dir + "/unetdisstat.pth"))
        else:
            self.load_state_dict(torch.load(model_dir + "/unetdisstat_{}.pth".format(i)))





