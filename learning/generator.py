import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.cnn import CNN3x3
from models.gnn import GT
from models.general import FF, SLN

import numpy as np


class CNNGen(nn.Module):
    def __init__(self, nw_data, output_channel, max_num=40, sln=True, w_dim=10):
        super().__init__()

        self.nw_data = nw_data
        self.output_channel = output_channel
        self.max_num = max_num

        self.total_feature = self.nw_data.feature_num + self.nw_data.context_feature_num
        self.cnn = CNN3x3((3, 3), (self.total_feature, self.total_feature*4, self.output_channel), residual=True, sn=False, sln=sln, w_dim=w_dim)  # forward: (B, C, 3, 3)->(B, C', 3, 3)

        self.w = None

    def forward(self, inputs, w=None, i=None):
        # inputs: (sum(links), total_feature, 3, 3)
        # output: (sum(links), 3, 3)
        # model output: (sum(links), oc, 3, 3)
        if w is not None:
            self.w = w
        if i is not None:
            return self.cnn(inputs, w=self.w)[:, i, :, :]
        return self.cnn(inputs, w=self.w)

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

        tmp_links = np.random.choice(self.nw_data.lids, sum(num))
        d_nodes = np.random.choice(self.nw_data.nids, sum(num))
        mode = np.array([m for m in range(self.output_channel) for i in range(num[m])])
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
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnngen.pth"))


class GNNGen(nn.Module):
    def __init__(self, nw_data, emb_dim, output_channel, enc_dim, 
                 in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None):
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.nw_data = nw_data
        self.emb_dim = emb_dim
        self.feature_num = nw_data.feature_num
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
        self.gnn = GT(self.feature_num, emb_dim, self.adj_matrix, **kwargs)  # (bs, link_num, emb_dim)
        self.ff = FF(emb_dim*2, output_channel, emb_dim*4, sn=sn)

        self.w = None

    def forward(self, x, w=None, i=None):
        # x: (bs, link_num, feature_num)
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

    def set_w(self, w):
        self.w = w

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/gnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/gnngen.pth"))



