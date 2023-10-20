import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from cnn import CNN2L
from transformer import Transformer
from general import FF, SLN

import numpy as np

class CNNGen(nn.Module):
    def __init__(self, nw_data, output_channel, max_num=40, device="cpu"):
        super().__init__()

        self.nw_data = nw_data
        self.output_channel = output_channel
        self.max_num = max_num
        self.device = device

        self.input_feature = self.nw_data.link_feature_num + self.nw_data.context_feature_num
        self.cnn = CNN2L(self.input_feature, self.output_channel, sln=sln, w_dim=w_dim)

    def forward(self, input, i):
        # input: (sum(links), input_feature, 3, 3)
        # output: (sum(links), 3, 3)
        # model output: (sum(links), oc, 3, 3)
        return self.cnn(input)[:, i, :, :]

    def generate(self, num):
        # num: int or list(self.output_channel elements)
        # generate num paths (the path is stopped after 1000 steps)
        # fake_data: [{pid: {d_node: d_node, path: [link]}}]
        if type(num) == int:
            num = [num for _ in range(self.output_channel)]
        if len(num) != self.output_channel:
            raise Exception("num should be int or list(self.output_channel elements)")

        tmp_links = np.random.choice(self.nw_data_lids, sum(num))
        d_nodes = np.random.choice(self.nw_data.nids, sum(num))
        mode = np.array([m for m in range(self.output_channel) for i in range(num[m])])
        paths = [[l] for l in tmp_links]
        pids = np.array([i for i in range(sum(num))])  # i th element: pid of i th link

        for _ in range(self.max_num):
            # policy function (batch_size, link_feature_num+context_feature_num, 3, 3)->(batch_size, 9, output_channel)
            bs = len(pids)
            input = tensor(np.zeros((bs, self.link_feature_num + self.context_feature_num, 3, 3), dtype=np.float32),
                           device=self.device)
            masks = tensor(np.zeros((bs, 9, 1), dtype=np.float32), device=self.device)
            action_edges = []
            for idx, pid in enumerate(pids):
                feature_mat, _ = self.nw_data.get_feature_matrix(paths[pid][-1])
                context_mat, action_edge = self.nw_data.get_context_matrix(paths[pid][-1], d_nodes[pid])
                mask = tensor((np.array([len(action_edge[j]) for j in range(9)]) > 0).astype(np.float32),
                              device=self.device)

                input[idx, :, :, :] = tensor(np.concatenate((feature_mat, context_mat), axis=0).astype(np.float32))
                masks[idx, :, 0] = mask
                action_edges.append(action_edge)

            pi = torch.where(masks, self.cnn(input), tensor(-9e15, dtype=torch.float32, device=self.device))  # [bs ,9, oc]
            pi_mode = tensor(np.zeros((bs, 9), dtype=np.float32), device=self.device)
            for j in range(bs):
                pi_mode[j, :] = pi[j, :, mode[j]]
            pi = pi_mode
            pi[:, 4] = tensor(-9e15, dtype=torch.float32, device=self.device)

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

        fake_data = [dict() for i in range(self.output_channel)]
        for mode in range(self.output_channel):
            for n in range(num[mode]):
                pid = sum(num[0:mode]) + n
                if len(paths[pid]) > 0:
                    fake_data[mode][n] = {"d_node": d_nodes[pid], "path": paths[pid]}
        return fake_data

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/cnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/cnngen.pth"))


class GNNGen(nn.Module):
    def __init__(self, in_channel, output_channel, device="cpu", h_dim=1, sln=False, w_dim=None):
        super().__init__()
        if sln and w_dim is None:
            raise Exception("w_dim should be specified when sln is True")

        self.in_channel = in_channel
        self.output_channel = output_channel
        self.device = device
        self.h_dim = h_dim
        self.sln = sln
        self.w_dim = w_dim

        self.transformer = Transformer(in_channel, output_channel, k=3, dropout=0.1, depth=3, residual=True, sln=sln, w_dim=w_dim)

    def forward(self, x, i, w=None):
        # x: (link_num, in_channel)
        # enc: (trip_num, enc_dim)
        # output: (trip_num, link_num, link_num)
        if self.sln:
            return self.transformer(x, None, w)[:, i, :, :]
        else:
            return self.transformer(x, None)[:, i, :, :]

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir + "/gnngen.pth")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir + "/gnngen.pth"))


