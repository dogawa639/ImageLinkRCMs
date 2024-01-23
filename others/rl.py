import torch
from torch import tensor

import random
import scipy
from scipy.sparse import lil_array, csr_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import os

from utility import *
from preprocessing.network import *

__all__ = ["RL"]

class RL:
    # link to link
    def __init__(self, pp_data, beta=0.9, device="cpu", max_sample=None):
        self.pp_data = pp_data
        self.nw_data = pp_data.nw_data
        self.lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        self.beta = beta

        self.device = device
        self.max_sample = max_sample

        self.up_link = {nid: [] for nid in self.nw_data.nids}
        for edge in self.nw_data.edges.values():
            self.up_link[edge.end.id].append(edge.id)

        self._initialize_props()

    def get_sample(self):
        # path_dict: {tid: {"path": [link_id], "d_node": node_id}}
        # return: {d_link_idx: [(k_link_idx, a_link_idx)]}
        sample = dict()
        for num, trip in enumerate(self.pp_data.path_dict.values()):
            path = trip["path"]
            d_link_idx = self.lid2idx[path[-1]]
            if d_link_idx not in sample:
                sample[d_link_idx] = []
            for i in range(len(path) - 1):
                sample[d_link_idx].append((self.lid2idx[path[i]], self.lid2idx[path[i + 1]]))
            if self.max_sample is not None and num > self.max_sample:
                break

        return sample

    def comp_m(self, p, mix_term=None, use_bias=False):
        # p: [param for prop_vec] + [param for prop_mat] (+ [param for bias]) + [param for mix_term]
        # A,x,p1,p2からMを計算する．Mに値を代入
        # mix_term: None or ndarray (n, n)
        param_len = self.prop_len + (1 * (mix_term is not None)) + (1 * use_bias)
        if len(p) != param_len:
            raise Exception(f"p should be {param_len} length, but {len(p)}")
        v = np.sum(self.prop_vec * p[:self.prop_vec.shape[1]], axis=1).astype(np.float32)  # (n)
        v = np.expand_dims(v, 0).repeat(self.n, 0)  # (n, n)
        v += self.prop_mat[0].toarray() * p[self.prop_vec.shape[1]]  # (n, n)
        if use_bias:
            v += p[self.prop_len]
        if not mix_term is None:
            v += p[-1] * mix_term

        v = np.clip(v, a_min=None, a_max=0)

        m = np.exp(v).astype(np.float32) * (self.nw_data.edge_graph > 0)
        # IM=np.linalg.inv(np.eye(a).astype(np.float32)-M.detach().cpu().numpy())
        return m  # 遷移効用行列

    def comp_z(self, m, sample, d_node=False):
        # M,dtからzを計算する．zに値を代入．
        # z=Mz^{beta}+dt
        # z : (n, len(sample)), M : (n, n), dt : (n, len(sample))
        # d_node: bool, if True z=1 for all links connected to d_node
        m = tensor(m, device=self.device)
        z = tensor(np.zeros((self.n, len(sample)), dtype=np.float32), device=self.device)
        dt = tensor(np.zeros((self.n, len(sample)), dtype=np.float32), device=self.device)
        for i, d_index in enumerate(sample.keys()):
            if d_node:
                d_nid = self.nw_data.edges[self.nw_data.lids[d_index]].end.id
                for lid in self.up_link[d_nid]:
                    tmp_d_idx = self.lid2idx[lid]
                    z[tmp_d_idx, i] = 1.0
                    dt[tmp_d_idx, i] = 1.0
            else:
                z[d_index, i] = 1.0
                dt[d_index, i] = 1.0

        z_dict = dict()  # 価値関数ベクトル
        for i in range(int(5 * (self.n + 1) ** 0.5)):
            z_beta = torch.pow(z, self.beta)
            new = torch.matmul(m, z_beta)
            new = new * (dt != 1.0) + dt

            dz = torch.sqrt(torch.sum(torch.pow(z - new, 2)))
            z = new
            if i > (self.n + 1) ** 0.5 and dz < 1e-8:
                break
        for i, d_index in enumerate(sample.keys()):
            z_dict[d_index] = z[:, i].detach().cpu().numpy()
        return z_dict

    def get_choice_prob(self, d_idx, k_idx, a_idx, m, z_dict=None, z_beta=None):
        # d_idx: int, k_idx: int, a_idx: int
        # m: ndarray(n, n), z_dict: {d_idx: ndarray(n)}
        # return: float
        if z_dict is None and z_beta is None:
            raise Exception("z_dict or z_beta should be set")
        u = m[k_idx, :]
        if z_dict is not None:
            q = u * z_dict[d_idx] ** self.beta
        elif z_beta is not None:
            q = u * z_beta
        return q[a_idx] / np.sum(q)

    def get_choice_prob_mat(self, m, z_dict=None, z_beta=None):
        # m: ndarray(n, n), z_dict: {d_idx: ndarray(n)}
        # return: {d_idx: ndarray(n, n)}
        if z_dict is None and z_beta is None:
            raise Exception("z_dict or z_beta should be set")
        if z_dict is not None:
            z_beta_dict = {d_idx: np.power(z_dict[d_idx], self.beta) for d_idx in z_dict.keys()}
            prob_mat = dict()
            for d_idx in z_beta_dict.keys():
                q = m * np.expand_dims(z_beta_dict[d_idx], 0)
                q_sum = np.sum(q, axis=1, keepdims=True)
                q_sum[q_sum == 0] = 1.0
                prob_mat[d_idx] = q / q_sum
        elif z_beta is not None:
            q = m * z_beta.reshape(1, -1)
            prob_mat = q / np.sum(q, axis=1, keepdims=True)
        return prob_mat

    def get_choice_prob_marginal(self, m, z_dict=None, z_beta=None):
        # m: ndarray(n, n), z_dict: {d_idx: ndarray(n)}
        # return: ndarray(n, n)
        if z_dict is None and z_beta is None:
            raise Exception("z_dict or z_beta should be set")
        if z_dict is not None:
            z_beta = {d_idx: np.power(z_dict[d_idx], self.beta) for d_idx in z_dict.keys()}
        prob_mat = dict()
        for d_idx in z_beta.keys():
            q = m * np.expand_dims(z_beta[d_idx], 0)
            q_sum = np.sum(q, axis=1, keepdims=True)
            q_sum[q_sum == 0] = 1.0
            prob_mat[d_idx] = q / q_sum
        prob_marge = np.sum([prob_mat[d_idx] for d_idx in prob_mat.keys()], axis=0) / len(prob_mat)
        return prob_marge

    def get_sample_ll(self, sample, m, z_dict):
        # sample: {d_idx: [(k_idx, a_idx)]}, m: ndarray(n, n), z_dict: {d_idx: ndarray(n)}
        # return: float
        ll = 0
        for d_idx, k_a_list in sample.items():
            z_beta = np.power(z_dict[d_idx], self.beta)
            for k_idx, a_idx in k_a_list:
                prob = self.get_choice_prob(d_idx, k_idx, a_idx, m, z_beta=z_beta)
                ll += np.log(prob + (prob == 0))
        return ll

    def get_sample_acc(self, sample, m, z_dict):
        # sample: {d_idx: [(k_idx, a_idx)]}, m: ndarray(n, n), z_dict: {d_idx: ndarray(n)}
        # return: float
        right = 0
        total = 0
        for d_idx, k_a_list in sample.items():
            z_beta = np.power(z_dict[d_idx], self.beta)
            prob_mat = self.get_choice_prob_mat(m, z_beta=z_beta)
            for k_idx, a_idx in k_a_list:
                total += (prob_mat[k_idx, :] > 0).sum()
                right += np.argmax(prob_mat[k_idx, :]) == a_idx
        return right / total

    def get_ll(self, p, sample, mix_term=None, use_bias=False):
        # p: ndarray(n), sample: {d_idx: [(k_idx, a_idx)]}
        # return: float
        m = self.comp_m(p, mix_term=mix_term, use_bias=use_bias)
        z_dict = self.comp_z(m, sample)
        return self.get_sample_ll(sample, m, z_dict)

    def get_acc(self, p, sample, mix_term=None, use_bias=False):
        # p: ndarray(n), sample: {d_idx: [(k_idx, a_idx)]}
        # return: float
        m = self.comp_m(p, mix_term=mix_term, use_bias=use_bias)
        z_dict = self.comp_z(m, sample)
        return self.get_sample_acc(sample, m, z_dict)

    # visualization
    def write_result(self, res, sample, mix_term=None, use_bias=False, file=None):
        d_num = len(sample)
        sample_num = sum([len(k_a_list) for k_a_list in sample.values()])
        org_len = len(res.x) - (1 * (mix_term is not None)) - (1 * use_bias)
        ll0 = self.get_ll(res.x[:org_len], sample)
        ll = self.get_ll(res.x, sample, mix_term=mix_term, use_bias=use_bias)
        acc = self.get_acc(res.x, sample, mix_term=mix_term, use_bias=use_bias)
        # try:
        t_val = res.x / np.sqrt(np.diag(res.hess_inv))  # t値
        # except:
        #    fn = lambda x: -self.get_ll(x, sample, mix_term=mix_term, use_bias=use_bias)
        #    hess = hessian(fn)(res.x)
        #    t_val = res.x / np.sqrt(-np.diag(np.linalg.pinv(hess)))  # t値
        rho2 = (ll0 - ll) / ll0  # 尤度比
        rho2_adj = (ll0 - (ll - len(res.x))) / ll0  # 修正済み尤度比
        aic = -2 * (ll - len(res.x))  # 赤池情報量基準
        prop_names = self.prop_names + (["bias"] if use_bias else []) + (["mix_term"] if mix_term is not None else [])

        string = "---------- Estimation Results ----------\n"
        string += f"     d number = {d_num}\n"
        string += f"sample number = {sample_num}\n"
        string += f"    variables = {prop_names}\n"
        string += f"    param_std = {self.prop_vec_std} | {self.prop_mat_std}\n"
        string += f"    parameter = {res.x}\n"
        string += f"      t value = {t_val}\n"
        string += f"          jac = {res.jac}\n"
        string += f"           L0 = {ll0}\n"
        string += f"           LL = {ll}\n"
        string += f"          ACC = {acc}\n"
        string += f"         rho2 = {rho2}\n"
        string += f"adjusted rho2 = {rho2_adj}\n"
        string += f"          AIC = {aic}\n"
        string += f"         beta = {self.beta}\n"
        print(string)
        if file is not None:
            with open(file, "w") as f:
                f.write(string)

    # forward
    def forward(self, p, num, d_link_id=None, d_node_id=None, o_node_id=None, mix_term=None, use_bias=False, max_step=100):
        # num: int, d_node: node_id, o_node: node_id or [node_id]
        # return: {d_link_idx: [[path_link_idx]]}
        # d_node: None -> random, o_node: None -> random
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        nid2idx = {nid: i for i, nid in enumerate(self.nw_data.nids)}
        use_d_node = True
        if d_link_id is None and d_node_id is None:
            d_idx = np.random.choice(np.arange(len(self.nw_data.lids)))
        elif d_link_id is not None:
            d_idx = self.lid2idx[d_link_id]
            use_d_node = False
        elif d_node_id is not None:
            d_node = self.nw_data.nodes[d_node_id]
            if len(d_node.upstream) == 0:
                raise Exception(f"no upstream from {d_node_id}")
            d_link = d_node.upstream[0]
            d_idx = self.lid2idx[d_link.id]

        if use_d_node:  # all links connected to d_node is the destination
            d_idx_set = set([lid2idx[l.id] for l in list(self.nw_data.edges.values())[d_idx].end.upstream])
        else:
            d_idx_set = set([d_idx])

        if o_node_id is None:
            o_node_id = np.random.choice(self.nw_data.nids, size=num)
        elif type(o_node_id) is int:
            o_node_id = np.array([o_node_id] * num)

        sample = {d_idx: []}

        m = self.comp_m(p, mix_term=mix_term, use_bias=use_bias)
        z_dict = self.comp_z(m, sample, d_node=use_d_node)
        prob_mat = self.get_choice_prob_mat(m, z_dict=z_dict)

        path_dict = {d_idx: []}
        for i, tmp_o_node_id in enumerate(o_node_id):
            tmp_link = np.random.choice(self.nw_data.nodes[tmp_o_node_id].downstream)
            tmp_link_idx = self.lid2idx[tmp_link.id]
            tmp_path = [tmp_link_idx]

            for _ in range(max_step):
                prob = prob_mat[d_idx][tmp_link_idx, :]
                if np.sum(prob) == 0:
                    break
                tmp_link_idx = np.random.choice(np.arange(self.n), p=prob)
                tmp_path.append(tmp_link_idx)

                if tmp_link_idx in d_idx_set:
                    break
            path_dict[d_idx].append(tmp_path)
        return path_dict

    # inside function
    def _initialize_props(self):
        # (k,a):リンクkからリンクa
        n = len(self.nw_data.lids)
        link_props = [prop for prop in self.nw_data.link_props if prop not in Edge.base_prop]
        l = np.zeros((n, len(link_props) + 1), dtype=np.float32)  # length, attri1, attri2, ...
        for i, edge in enumerate(self.nw_data.edges.values()):
            l[i, 0] = edge.length
            for j, prop in enumerate(link_props):
                if prop in edge.prop:
                    l[i, 1 + j] = edge.prop[prop]

        angle = lil_array((n, n), dtype=np.float32)  # (n, n) angle
        for i, edge in enumerate(self.nw_data.edges.values()):
            for down in edge.end.downstream:
                j = self.lid2idx[down.id]
                angle[i, j] = down.angle - edge.angle

        self.prop_vec_std = np.std(l, axis=0)
        self.prop_vec_mean = np.mean(l, axis=0)
        self.prop_vec_std[self.prop_vec_std == 0] = 1.0
        self.prop_mat_std = [np.std(angle.toarray())]

        self.prop_vec = (l - self.prop_vec_mean) / self.prop_vec_std  # (n, len(link_props) + 1)
        self.prop_mat = [angle / self.prop_mat_std[0]]  # [(n, n)]
        self.n = n
        self.prop_len = self.prop_vec.shape[1] + len(self.prop_mat)
        self.prop_names = ["length"] + link_props + ["angle"]

        print(f"prop_vec: {self.prop_vec.shape}, prop_mat: {self.prop_mat[0].shape}*{len(self.prop_mat)}")
        print(f"prop_vec_std: {self.prop_vec_std}, prop_mat_std: {self.prop_mat_std}")
        print(f"prop_names: {self.prop_names}")

