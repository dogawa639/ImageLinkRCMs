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
            z_beta = {d_idx: np.power(z_dict[d_idx], self.beta) for d_idx in z_dict.keys()}
        prob_mat = dict()
        for d_idx in z_beta.keys():
            q = m * np.expand_dims(z_beta[d_idx], 0)
            q_sum = np.sum(q, axis=1, keepdims=True)
            q_sum[q_sum == 0] = 1.0
            prob_mat[d_idx] = q / q_sum
        return prob_mat

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

    def get_ll(self, p, sample, mix_term=None, use_bias=False):
        # p: ndarray(n), sample: {d_idx: [(k_idx, a_idx)]}
        # return: float
        m = self.comp_m(p, mix_term=mix_term, use_bias=use_bias)
        z_dict = self.comp_z(m, sample)
        return self.get_sample_ll(sample, m, z_dict)

    # visualization
    def write_result(self, res, sample, mix_term=None, use_bias=False, file=None):
        d_num = len(sample)
        sample_num = sum([len(k_a_list) for k_a_list in sample.values()])
        org_len = len(res.x) - (1 * (mix_term is not None)) - (1 * use_bias)
        ll0 = self.get_ll(res.x[:org_len], sample)
        ll = self.get_ll(res.x, sample, mix_term=mix_term, use_bias=use_bias)
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


# test
if __name__ == "__main__":
    import json
    import configparser
    from preprocessing.network import *
    from preprocessing.pp import *
    from models.general import FF
    from models.gnn import GAT, GT
    from torch import nn

    import matplotlib.pyplot as plt
    from logger import Logger

    print(os.getcwd())

    device = "mps"

    CONFIG = "/Users/dogawa/PycharmProjects/GANs/config/config_all.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    pp_path = json.loads(read_data["pp_path"])

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    pp_data = [PP(path, nw_data) for path in pp_path]
    seed = 100
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    # x_ped = np.array([-1.16, 0.29, 0.02, -0.51, 0.27], dtype=np.float32)  # 0: length / 100, -2: angle, -1: mix_term
    # x_car = np.array([-0.36, -0.13, -0.04, -0.64, 0.73], dtype=np.float32)  # 0: length / 100, -2: angle, -1: mix_term

    rl_ped = RL(pp_data[0], device=device, max_sample=10)
    rl_car = RL(pp_data[1], device=device, max_sample=10)

    sample_ped = rl_ped.get_sample()
    sample_car = rl_car.get_sample()

    z_test = True
    interaction_estimation = False
    v_ml_test = False

    if z_test:
        x = np.array([-0.95709107, -0.86090123, -0.50018837, 0.48622069])
        m = rl_ped.comp_m(x)
        z_dict = rl_ped.comp_z(m, sample_ped)

        k = list(z_dict.keys())[0]
        # z = Mz+dt  ->  z = (I-M)^(-1)dt
        z_pred = z_dict[k]
        v_pred = np.log(z_pred + (z_pred <= 0))
        dt = np.zeros((rl_ped.n, 1), dtype=np.float32)
        dt[k, 0] = 1.0
        m2 = m.copy()
        m2[k, :] = 0.0
        z_true = np.linalg.pinv(np.eye(rl_ped.n, dtype=np.float32) - m2) @ dt
        z_true = z_true.flatten()
        v_true = np.log(z_true + (z_true <= 0))
        u, s, vh = np.linalg.svd(np.eye(rl_ped.n, dtype=np.float32) - m2)
        print(s[0] / s[-1])

        # write_2d_ndarray("/Users/dogawa/PycharmProjects/GANs/debug/data/m.txt", m)

        fig = plt.figure()
        plt.scatter(z_true, z_pred)
        plt.xlabel("z_true")
        plt.ylabel("z_pred")
        plt.plot([0, 1], [0, 1], color="red", linestyle="dashed")
        plt.show()

        d_node = nw_data.nodes[nw_data.edges[nw_data.lids[k]].end.id]
        xlim = [-69400, -69000]
        ylim = [92850, 93300]
        fig, ax = plt.subplots(dpi=200)
        ax = nw_data.plot_link(nw_data.lids, v_true, title="v_true", ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()
        fig, ax = plt.subplots(dpi=200)
        ax = nw_data.plot_link(nw_data.lids, v_pred, title="v_pred", ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()

        lid2idx = {lid: i for i, lid in enumerate(nw_data.lids)}
        prob = np.zeros(len(nw_data.lids), dtype=np.float32)
        for node in nw_data.nodes.values():
            idxs = []
            zs = []
            v_total = 0.0
            for link in node.downstream:
                idxs.append(lid2idx[link.id])
                zs.append(z_pred[lid2idx[link.id]])
                v_total += z_pred[lid2idx[link.id]]
            if v_total > 0.0:
                prob[idxs] = np.array(zs) / v_total
        fig, ax = plt.subplots()
        ax = nw_data.plot_link(nw_data.lids, prob, title="prob", xlim=xlim, ylim=ylim, ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()

        paths = rl_ped.forward(x, 10, d_node_id=d_node.id)
        nw_data.plot_paths(paths)

    if interaction_estimation:

        def fn_ll_ped(x, rl_ped, sample_ped):
            # x = np.concatenate([x_ped, x_car])
            mix_term_ped = rl_car.comp_m(x[rl_ped.prop_len + 2:-1])
            return -rl_ped.get_ll(x[:rl_ped.prop_len + 2], sample_ped, mix_term=mix_term_ped, use_bias=True)


        def fn_ll_car(x, rl_car, sample_car):
            mix_term_car = rl_ped.comp_m(x[:rl_car.prop_len + 1], use_bias=True)
            return -rl_car.get_ll(x[rl_car.prop_len + 2:], sample_car, mix_term=mix_term_car)


        x = np.zeros((rl_ped.prop_len + 2) + (rl_car.prop_len + 1), dtype=np.float32)
        dx = 100
        x_ped_len = rl_ped.prop_len + 2
        x_car_len = rl_car.prop_len + 1
        log1 = Logger("../result/rl/log.txt", "")
        tol = 1e-6
        cnt = 0
        while dx > 1e-2 and cnt < 1:
            pre_x = x.copy()
            x0 = np.zeros(x_ped_len, dtype=np.float32)
            fn_ped = lambda x_ped: fn_ll_ped(np.concatenate((x_ped, x[x_ped_len:])), rl_ped, sample_ped)
            fprime_ped = lambda x: scipy.optimize.approx_fprime(x, fn_ped)
            opt_kwargs_ped = {"method": "BFGS", "jac": fprime_ped, "tol": tol}
            res_ped = scipy.optimize.minimize(fn_ped, x0, **opt_kwargs_ped)
            # x[:x_ped_len] = res_ped.x
            log1.add_log("x", x)
            log1.add_log("fn", res_ped.fun)

            x0 = np.zeros(x_car_len, dtype=np.float32)
            fn_car = lambda x_car: fn_ll_car(np.concatenate((x[:x_ped_len], x_car)), rl_car, sample_car)
            fprime_car = lambda x: scipy.optimize.approx_fprime(x, fn_car)
            opt_kwargs_car = {"method": "BFGS", "jac": fprime_car, "tol": tol}
            res_car = scipy.optimize.minimize(fn_car, x0, **opt_kwargs_car)
            x[x_ped_len:] = res_car.x
            log1.add_log("x", x)
            log1.add_log("fn", res_car.fun)

            dx = np.sqrt(np.mean(np.power(x - pre_x, 2)))
            log1.add_log("dx", dx)

            cnt += 1

        log1.close()

        x_ped = x[:x_ped_len]
        x_car = x[x_ped_len:]

        mix_term_car = rl_ped.comp_m(x_ped[:-1], use_bias=True)
        mix_term_ped = rl_car.comp_m(x_car[:-1])

        rl_ped.write_result(res_ped, sample_ped, mix_term=mix_term_ped, use_bias=True,
                            file="../result/rl/rl_ped_single.txt")
        rl_car.write_result(res_car, sample_car, mix_term=mix_term_car, file="../result/rl/rl_car_single.txt")

    if v_ml_test:
        x_ped = [-0.95709107, -0.86090123, -0.50018837, 0.48622069, -0.85128209, -0.77433027]  # x[:x_ped_len]
        x_car = [-0.31941202, -0.33455477, -0.05158337, 0.25108421, -0.01120311]  # x[x_ped_len:]

        M_ped = rl_ped.comp_m(x_ped, mix_term=mix_term_ped, use_bias=True)
        M_car = rl_car.comp_m(x_car, mix_term=mix_term_car)

        z_ped = rl_ped.comp_z(M_ped, sample_ped)
        z_car = rl_car.comp_z(M_car, sample_car)

        feature_dict = dict()
        input_channel = nw_data.feature_num + nw_data.context_feature_num
        # input_channel = nw_data.context_feature_num
        for d_idx in z_ped.keys():
            d_link_id = rl_ped.nw_data.lids[d_idx]
            d_node_id = rl_ped.nw_data.edges[d_link_id].end.id
            all_features = nw_data.get_all_feature_matrix(d_node_id, normalize=True)
            feature_dict[d_idx] = all_features

        value_model1 = GT(input_channel, input_channel,
                          tensor(nw_data.edge_graph > 0, dtype=torch.float32, device=device), output_atten=False,
                          sn=False, depth=3, atten_fn="dense").to(device)
        value_model2 = FF(input_channel, 1, act_fn=lambda x: torch.exp(x)).to(device)
        value_model = nn.Sequential(value_model1, value_model2)
        optimizer = torch.optim.Adam(value_model.parameters(), lr=0.001)
        sc = StandardScaler()
        losses = []
        for epoch in range(100):
            value_model.train()
            tmp = []
            for i, d_idx in enumerate(z_ped.keys()):
                y = tensor(z_ped[d_idx].reshape(-1, 1), device=device)
                x = tensor(feature_dict[d_idx], device=device)
                x = torch.unsqueeze(x, 0)

                y2 = value_model(x).reshape(-1, 1)
                loss = torch.mean(torch.pow(y - y2, 2))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tmp.append(loss.detach().cpu().item())
            losses.append(np.mean(tmp))

        value_model.eval()
        errors = []
        for i, d_idx in enumerate(z_ped.keys()):
            y = tensor(z_ped[d_idx].reshape(-1, 1), device=device)
            x = tensor(feature_dict[d_idx], device=device)
            x = torch.unsqueeze(x, 0)
            y2 = value_model(x).reshape(-1, 1)
            errors.append((y - y2).cpu().detach().numpy().flatten().tolist())
        print(y, y2)
        print(x.shape, y.shape, y2.shape)
        errors = sum(errors, [])
        # write_1d_array("/Users/dogawa/PycharmProjects/GANs/debug/data/y.txt", np.array(y.detach().cpu().numpy().flatten()))
        # write_1d_array("/Users/dogawa/PycharmProjects/GANs/debug/data/y2.txt", np.array(y2.detach().cpu().numpy().flatten()))
        # write_1d_array("/Users/dogawa/PycharmProjects/GANs/debug/data/errors.txt", np.array(errors))
        plt.hist(errors, bins=30)
        plt.xlabel("error")
        plt.show()
        plt.hist(y2.detach().cpu().numpy().flatten(), bins=30)
        plt.xlabel("y2")
        plt.show()
        plt.scatter(y.detach().cpu().numpy().flatten(), y2.detach().cpu().numpy().flatten())
        m1, M1 = y.detach().cpu().numpy().flatten().min(), y.detach().cpu().numpy().flatten().max()
        m, M = y2.detach().cpu().numpy().flatten().min(), y2.detach().cpu().numpy().flatten().max()
        m, M = 0.0, 0.1
        plt.xlim([m, M])
        plt.ylim([m, M])
        plt.plot([m, M], [m, M], color="red", linestyle="dashed")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.show()
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
