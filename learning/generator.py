import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.cnn import CNN3x3
from models.gnn import GT
from models.unet import UNet
from models.general import FF, SLN, softplus

import numpy as np
__all__ = ["CNNGen", "GNNGen", "UNetGen"]


# CNN
# input: (sum(links), total_feature, 3, 3)
# output: (sum(links), 3, 3) or (sum(links), oc, 3, 3)
# GNN
# input: (bs, link_num, feature_num)
# output: (bs, link_num, link_num) or (trip_num, oc, link_num, link_num)
class CNNGen(nn.Module):
    def __init__(self, nw_data, output_channel, image_feature_num=0, max_num=40, sln=False, w_dim=10):
        super().__init__()

        self.nw_data = nw_data
        self.output_channel = output_channel
        self.image_feature_num = image_feature_num
        self.max_num = max_num

        self.total_feature = self.nw_data.feature_num + self.nw_data.context_feature_num + self.image_feature_num
        self.cnn = CNN3x3((3, 3), (self.total_feature, self.total_feature*4, self.output_channel), residual=True, sn=False, sln=sln, w_dim=w_dim)  # forward: (B, C, 3, 3)->(B, C', 3, 3)

        self.w = None

    def forward(self, inputs, w=None, i=None):
        # inputs: (sum(links), total_feature, 3, 3)
        # output: (sum(links), 3, 3) or (sum(links), oc, 3, 3)
        if w is not None:
            self.w = w
        if i is not None:
            return self.cnn(inputs, w=self.w)[:, i, :, :]
        return self.cnn(inputs, w=self.w)

    def get_pi(self, dataset, d_weight, w=None, i=None):
        # merginal probability of link transition
        # dataset: GridDataset
        # output: [link_num, link_num] or [oc, link_num, link_num]
        # w : (w_dim)
        if w is not None:
            self.w = w
        if i is not None:
            trans_mat = torch.zeros((self.nw_data.link_num, self.nw_data.link_num), dtype=torch.float32)
        else:
            trans_mat = torch.zeros((self.output_channel, self.nw_data.link_num, self.nw_data.link_num), dtype=torch.float32)
        for d_node_id in self.nw_data.nids:
            # inputs : [link_num, f+c, 3, 3]
            # masks : [link_num, 9]
            # link_idxs : [link_num, 9]
            if d_weight[d_node_id] <= 0:
                continue
            inputs, masks, link_idxs = dataset.get_all_input(d_node_id)
            out = self.cnn(inputs, w=self.w)  # (link_num, oc, 3, 3)
            if i is not None:
                out = out[:, i, :, :]
                out = torch.where(masks > 0, out.view(out.shape[0], -1), tensor(-9e15, dtype=torch.float32, device=out.device))  # (link_num, 9)
                pi = F.softmax(out, dim=1)  # (link_num, 9)
                for j in range(self.link_num):
                    tmp_link_idxs = link_idxs[j][link_idxs[j] >= 0]
                    trans_mat[j, tmp_link_idxs] = trans_mat[j, link_idxs] + pi[j][link_idxs[j] >= 0] * d_weight[d_node_id]
            else:
                out = torch.where(masks.unsqueeze(1) > 0, out.view(*out.shape[:2], -1), tensor(-9e15, dtype=torch.float32, device=out.device))
                pi = F.softmax(out, dim=1)  # (link_num, oc, 9)
                for j in range(self.link_num):
                    tmp_link_idxs = link_idxs[j][link_idxs[j] >= 0]
                    trans_mat[:, j, tmp_link_idxs] = trans_mat[:, j, link_idxs] + pi[j][link_idxs[j] >= 0] * d_weight[d_node_id]
        trans_mat = trans_mat / d_weight.sum()
        return trans_mat

    def generate(self, num, w=None):
        # num: int or list(self.output_channel elements)
        # fake_data: [{pid: {d_node: d_node, path: [link]}}]
        # w : (sum(num), w_dim)
        # d_node is specified in context feature
        if type(num) == int:
            num = [num for _ in range(self.output_channel)]
        if len(num) != self.output_channel:
            raise Exception("num should be int or list(self.output_channel elements)")

        device = next(self.cnn.covs[0].parameters()).device

        tmp_links = np.random.choice(self.nw_data.lids, sum(num))  # [link_id]
        d_nodes = np.random.choice(self.nw_data.nids, sum(num))  # [d_node_id]
        mode = np.array([m for m in range(self.output_channel) for i in range(num[m])])  # [mode]
        paths = [[l] for l in tmp_links]
        pids = np.array([i for i in range(sum(num))])  # i th element: pid of i th link

        for _ in range(self.max_num):
            # policy function (batch_size, feature_num+context_feature_num, 3, 3)->(batch_size, 9, output_channel)
            bs = len(pids)
            inputs = tensor(np.zeros((bs, self.total_feature, 3, 3), dtype=np.float32),
                           device=device)
            masks = tensor(np.zeros((bs, 1, 9), dtype=np.float32), device=device)
            action_edges = []
            for idx, pid in enumerate(pids):
                feature_mat, _ = self.nw_data.get_feature_matrix(paths[pid][-1])
                context_mat, action_edge = self.nw_data.get_context_matrix(paths[pid][-1], d_nodes[pid])
                mask = tensor((np.array([len(action_edge[j]) for j in range(9)]) > 0).astype(np.float32),
                              device=device)

                inputs[idx, :, :, :] = tensor(np.concatenate((feature_mat, context_mat), axis=0).astype(np.float32))
                masks[idx, 0, :] = mask
                action_edges.append(action_edge)

            masks = masks.expand(-1, self.output_channel, -1)
            pi = torch.where(masks > 0, self.cnn(inputs, w=w).view(masks.shape), tensor(-9e15, dtype=torch.float32, device=device))  # [bs , oc, 9]
            pi_mode = tensor(np.zeros((bs, 9), dtype=np.float32), device=device)
            for j in range(bs):
                pi_mode[j, :] = pi[j, mode[j], :]
            pi = pi_mode
            pi[:, 4] = tensor(-9e15, dtype=torch.float32, device=device)

            pi_sum = torch.sum(pi, dim=1, keepdim=True)
            pi = F.softmax(pi, dim=1)
            elem = torch.multinomial(pi, 1).squeeze(dim=1).cpu().detach().numpy()

            active = torch.squeeze(pi_sum > 0.0, dim=1).cpu().detach().numpy()
            for j in range(bs):
                if active[j]:
                    paths[pids[j]].append(
                        action_edges[j][elem[j]][int(np.random.rand() * len(action_edges[j][elem[j]]))])

            tmp_links = tmp_links[active]
            mode = mode[active]
            pids = pids[active]
            if len(pids) == 0:
                break

        fake_data = [dict() for i in range(self.output_channel)]
        for mode in range(self.output_channel):
            for n in range(num[mode]):
                pid = sum(num[0:mode]) + n
                if len(paths[pid]) > 0:
                    fake_data[mode][n] = {"d_node": d_nodes[pid], "path": paths[pid]}
        return fake_data


    def set_w(self, w):
        # w : (w_dim)
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnngen.pth"))


class GNNGen(nn.Module):
    def __init__(self, nw_data, emb_dim, output_channel, enc_dim, 
                 image_feature_num=0, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None):
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.nw_data = nw_data
        self.emb_dim = emb_dim
        self.feature_num = nw_data.feature_num
        self.context_num = nw_data.context_feature_num
        self.image_feature_num = image_feature_num
        self.output_channel = output_channel
        self.adj_matrix = nn.parameter.Parameter(tensor(nw_data.edge_graph).to(torch.float32).to_dense(), requires_grad=False)
        self.sln = sln
        self.w_dim = w_dim

        kwargs = {
            "enc_dim": enc_dim, 
            "in_emb_dim": in_emb_dim, 
            "num_head": num_head, 
            "dropout": dropout, 
            "depth": depth, 
            "pre_norm": pre_norm, 
            "output_atten": True, 
            "sn": sn, 
            "sln": sln, 
            "w_dim": w_dim
            }
        self.gnn = GT(self.feature_num + self.context_num + self.image_feature_num, emb_dim, self.adj_matrix, **kwargs)  # (bs, link_num, emb_dim)
        self.ff = FF(emb_dim*2, output_channel, emb_dim*4, sn=sn)

        self.w = None

    def forward(self, x, w=None, i=None):
        # x: (bs, link_num, feature_num + context_num + image_feature_num)
        # output: (bs, link_num, link_num) or (trip_num, oc, link_num, link_num)
        if w is not None:
            self.w = w
        n = x.shape[1]
        y = self.gnn(x, w=self.w)[0].unsqueeze(-2).expand(-1, n, n, -1)  # (bs, link_num, 1, emb_dim)
        z = torch.cat((y, y.transpose(-3, -2)), dim=-1)  # (bs, link_num, link_num, emb_dim*2)
        logits = self.ff(z).permute(0, 3, 1, 2)
        mask = self.adj_matrix.expand(logits.shape) > 0.0
        logits = torch.where(mask, logits, tensor(-9e15, dtype=torch.float32, device=logits.device))  # (bs, oc, link_num, link_num)

        if i == None:
            return logits
        else:
            return logits[:, i, :, :]

    def get_pi(self, dataset, d_weight, w=None, i=None):
        # merginal probability of link transition
        # dataset: PPEmbedDataset
        # output: [link_num, link_num] or [oc, link_num, link_num]
        # w : (w_dim)
        if w is not None:
            self.w = w
        if i is not None:
            trans_mat = torch.zeros((self.nw_data.link_num, self.nw_data.link_num), dtype=torch.float32)
        else:
            trans_mat = torch.zeros((self.output_channel, self.nw_data.link_num, self.nw_data.link_num), dtype=torch.float32)
        for d_node_id in self.nw_data.nids:
            # inputs : [link_num, f+ c]
            # a_matrix : [link_num, link_num]
            if d_weight[d_node_id] <= 0:
                continue
            inputs, a_mat = dataset.get_all_input(d_node_id)
            inputs = inputs.unsqueeze(0)  # (1, link_num, f+c)
            logits = self(inputs, w=self.w, i=i).squeeze(0)  # (link_num, link_num) or (oc, link_num, link_num)
            pi = F.softmax(logits, dim=1)
            trans_mat = trans_mat + pi * d_weight[d_node_id]
        trans_mat = trans_mat / d_weight.sum()
        return trans_mat

    def set_w(self, w):
        # w : (trip_num, w_dim)
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/gnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/gnngen.pth"))


class UNetGen(nn.Module):
    # one generator for one transportation
    # unet bottleneck: (*, 1024, H/16, W/16)
    def __init__(self, feature_num, context_num,
                 sn=True, dropout=0.0, depth=1):
        super().__init__()
        # state : (bs, feature_num, 2d+1, 2d+1)
        # context : (bs, context_num, 2d+1, 2d+1)
        self.feature_num = feature_num
        self.context_num = context_num
        self.total_feature = feature_num + context_num

        self.unet = UNet(self.total_feature, 1, sn=sn, dropout=dropout, depth=depth, act_fn=lambda x : -softplus(x))

    def forward(self, inputs):
        # inputs: (bs, total_feature, 2d+1, 2d+1)
        # output: (bs, 2d+1, 2d+1)
        return self.unet(inputs).squeeze(1)

    def save(self, model_dir, i=None):
        if i is None:
            torch.save(self.state_dict(), model_dir + "/unetgen.pth")
        else:
            torch.save(self.state_dict(), model_dir + "/unetgen_{}.pth".format(i))

    def load(self, model_dir, i=None):
        if i is None:
            self.load_state_dict(torch.load(model_dir + "/unetgen.pth"))
        else:
            self.load_state_dict(torch.load(model_dir + "/unetgen_{}.pth".format(i)))




