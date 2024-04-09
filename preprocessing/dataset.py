import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision.transforms.v2 as transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utility import mutual_information as mi

from scipy.sparse import csr_matrix, coo_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime

import os

from utility import read_csv, load_json, dump_json
from preprocessing.geo_util import MapSegmentation
from preprocessing.pp import PP

__all__ = ["GridDataset", "PPEmbedDataset", "MeshDataset", "MeshDatasetStatic", "MeshDatasetStaticSub","PatchDataset", "XImageDataset", "XYImageDataset", "XHImageDataset", "StreetViewDataset", "StreetViewXDataset", "ImageDataset", "calc_loss"]


class GridDataset(Dataset):
    # 3*3 grid choice set
    def __init__(self, pp_data, h_dim=10, normalize=True):
        # nw_data: NWDataCNN
        self.pp_data = pp_data
        self.nw_data = pp_data.nw_data
        self.h_dim = h_dim

        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}

        f = self.nw_data.feature_num
        c = self.nw_data.context_feature_num
        # inputs: [sum(sample_num), f+c, 3, 3]
        # masks: [sum(sample_num), 9]
        # next_links: [sum(sample_num), 9]
        inputs = np.zeros((0, f + c, 3, 3), dtype=np.float32)
        masks = np.zeros((0, 9), dtype=np.float32)
        next_links = np.zeros((0, 9), dtype=np.float32)
        link_idxs = np.zeros((0, 9), dtype=np.int32)
        hs = np.zeros((0, h_dim), dtype=np.float32)
        for tid in pp_data.tids:
            path = pp_data.path_dict[tid]["path"]
            d_node_id = pp_data.path_dict[tid]["d_node"]
            if len(path) == 0:
                continue
            sample_num = len(path) - 1
            trip_input = np.zeros((sample_num, f + c, 3, 3), dtype=np.float32)  # [link, state, state]
            trip_mask = np.zeros((sample_num,9), dtype=np.float32)  # [link, mask]
            trip_next_link = np.zeros((sample_num,9), dtype=np.float32)  # [link, one_hot]
            trip_link_idxs = np.zeros((sample_num,9), dtype=np.int32)  # [link, next_link_idxs]

            for i in range(sample_num):
                tmp_link = path[i]
                next_link = path[i+1]
                feature_mat, _ = self.nw_data.get_feature_matrix(tmp_link, normalize=normalize)
                context_mat, action_edge = self.nw_data.get_context_matrix(tmp_link, d_node_id, normalize=normalize)

                trip_input[i, 0:f, :, :] = feature_mat
                trip_input[i, f:f+c, :, :] = context_mat
                trip_mask[i, :] = [len(edges) > 0 and j != 4 for j, edges in enumerate(action_edge)]
                trip_next_link[i, :] = [next_link in edges for edges in action_edge]
                trip_link_idxs[i, :] = [lid2idx[np.random.choice(edges)] if len(edges) > 0 else 0 for edges in action_edge]

            trip_input = trip_input[np.isnan(trip_input).sum(axis=(1, 2, 3)) == 0]
            if len(trip_input) == 0:
                continue
            inputs = np.concatenate((inputs, trip_input), axis=0)
            masks = np.concatenate((masks, trip_mask), axis=0)
            next_links = np.concatenate((next_links, trip_next_link), axis=0)
            link_idxs = np.concatenate((link_idxs, trip_link_idxs), axis=0)
            hs = np.concatenate((hs, np.repeat(np.random.randn(1, h_dim), len(trip_input), axis=0)), axis=0).astype(np.float32)

        self.inputs = tensor(inputs, requires_grad=False)
        self.masks = tensor(masks, requires_grad=False)
        self.next_links = tensor(next_links, requires_grad=False)
        self.link_idxs = tensor(link_idxs, requires_grad=False)
        self.hs = tensor(hs, requires_grad=False)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # [f+c, 3, 3], [9], [9], [9]
        return self.inputs[idx], self.masks[idx], self.next_links[idx], self.link_idxs[idx], self.hs[idx]
    
    def split_into(self, ratio):
        # ratio: [train, test, validation]
        pp_data = self.pp_data.split_into(ratio)
        return [GridDataset(pp_data[i], self.h_dim) for i in range(len(ratio))]

    def get_all_input(self, d_node_id, normalize=True):
        # inputs : [link_num, f+c, 3, 3]
        # masks : [link_num, 9]
        # link_idxs : [link_num, 9]
        f = self.nw_data.feature_num
        c = self.nw_data.context_feature_num
        inputs = np.zeros((0, self.nw_data.feature_num + self.nw_data.context_feature_num, 3, 3), dtype=np.float32)
        masks = np.zeros((0, 9), dtype=np.float32)
        link_idxs = np.zeros((0, 9), dtype=np.int32)
        for i, lid in enumerate(self.nw_data.lids):
            feature_mat, _ = self.nw_data.get_feature_matrix(lid, normalize=normalize)
            context_mat, action_edge = self.nw_data.get_context_matrix(lid, d_node_id, normalize=normalize)

            inputs[i, 0:f, :, :] = feature_mat
            inputs[i, f:f + c, :, :] = context_mat
            masks[i, :] = [len(edges) > 0 and j != 4 for j, edges in enumerate(action_edge)]
            link_idxs[i, :] = [self.nw_data.lid2idx[np.random.choice(edges)] if len(edges) > 0 else -1 for edges in action_edge]
        return tensor(inputs, requires_grad=False), tensor(masks, requires_grad=False), tensor(link_idxs, requires_grad=False)


    def get_fake_batch(self, real_batch, g_output):
        mask = real_batch[1].view(-1, 3, 3)
        logits = torch.where(mask > 0, g_output, tensor(-9e15, dtype=torch.float32, device=g_output.device))
        next_link_prob = F.softmax(logits, dim=-1)
        next_links = torch.multinomial(next_link_prob.view(-1, g_output.shape[-1]), num_samples=1).squeeze()
        next_links_one_hot = F.one_hot(next_links, num_classes=g_output.shape[-1]).view(g_output.shape)
        return [real_batch[i] if i != 2 else next_links_one_hot for i in range(len(real_batch))]


class PPEmbedDataset(Dataset):
    # nw_data: NWDataGNN
    # Global state (link_num, graph_node_feature_num) + PP pos embedding, PP adj matrix (link_num, link_num)
    def __init__(self, pp_data, h_dim=10):
        # global state: output of GNN
        self.pp_data = pp_data
        self.h_dim = h_dim
        self.nw_data = pp_data.nw_data
        self.feature_mat = self.nw_data.get_feature_matrix()
        self.link_num, self.feature_num = self.feature_mat.shape
        self.context_num = self.nw_data.context_feature_num

        self.d_node_ids = [val["d_node"] for val in pp_data.path_dict.values()]
        self.pp_pos_embeddings = [self.get_pp_pos_embedding(val["path"]) for val in pp_data.path_dict.values()]
        self.pp_adj_matrices = [self.get_pp_adj_matrix(val["path"]) for val in pp_data.path_dict.values()]

        self.hs = torch.randn((len(self.pp_data), h_dim), dtype=torch.float32, requires_grad=False)

    def __len__(self):
        return len(self.pp_data)
    
    def __getitem__(self, idx):
        kwargs = {"dtype": torch.float32, "requires_grad": False}
        # [link_num, feature_num+context_num], [link_num, link_num]
        return (tensor(np.concatenate((self.feature_mat, self.nw_data.get_context_matrix(self.d_node_ids[idx])), axis=1), **kwargs),  # inputs
                tensor(self.nw_data.a_matrix.toarray(), **kwargs).to_sparse(),  # 隣接行列
                tensor(self.pp_adj_matrices[idx].toarray(), **kwargs).to_sparse(),  # next_link_adj_matrix
                tensor(self.hs[idx], **kwargs))  # h

    def get_pp_pos_embedding(self, path):
        # path: [link_id] or [edge obj]
        pos_embedding = np.zeros((self.link_num, self.feature_num), dtype=np.float32)
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for i, edge in enumerate(path):
            if type(edge) != int and type(edge) != np.int64:
                edge = edge.id
            if i % 2 == 0:
                pos_embedding[lid2idx[edge], :] = np.sin(np.arange(self.feature_num) / 10000 ** (i / self.feature_num))
            else:
                pos_embedding[lid2idx[edge], :] = np.cos(np.arange(self.feature_num) / 10000 ** ((i - 1) / self.feature_num))
        return csr_matrix(pos_embedding)
    
    def get_pp_adj_matrix(self, path):
        # path: [link_id] or [edge obj]
        row = []
        col = []
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for i, edge in enumerate(path):
            if type(edge) != int and type(edge) != np.int64:
                edge = edge.id
            if i < len(path) - 1:
                row.append(lid2idx[edge])
                if type(path[i+1]) != int and type(path[i+1]) != np.int64:
                    path[i+1] = path[i+1].id
                col.append(lid2idx[path[i+1]])
        adj_matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(self.link_num, self.link_num), dtype=np.float32).tocsr()
        return adj_matrix

    def get_all_input(self, d_node_id):
        # inputs : [link_num, feature_num+context_num]
        # a_matrix : [link_num, link_num]
        kwargs = {"dtype": torch.float32, "requires_grad": False}
        inputs = tensor(np.concatenate((self.feature_mat, self.nw_data.get_context_matrix(d_node_id)), axis=1), **kwargs)
        a_matrix = tensor(self.nw_data.a_matrix.toarray(), **kwargs)
        return inputs, a_matrix
    
    def get_dataset_from_logits(self, logits, d_node_ids):
        # logits: [trip_num, link_num, link_num] tensor(cpu, detached) each element is real
        # d_node_ids: [trip_num] list
        # return: PPEmbedDataset
        exp_logits = torch.exp(logits)
        origin = (exp_logits.sum(dim=1, keepdim=False) > 0) & (exp_logits.sum(dim=2, keepdim=False) == 0)  # どこからも入ってきていないが，出ていっているリンク (trip_num, link_num)
        origin = origin[(origin.sum(dim=1) > 0), :]  # originがないtripは除く
        trip_num = origin.shape[0]

        logits_origin = torch.einsum('ijk, ij -> ij', logits, origin)  # (trip_num, link_num)
        origin_prob = logits_origin.softmax(dim=-1)  # (trip_num, link_num)

        paths = [[] for i in range(trip_num)]
        path_links = [{} for i in range(trip_num)]
        prev_idxs = [None for i in range(trip_num)]

        for i in range(trip_num):
            origin_idx = torch.multinomial(origin_prob[i, :], num_samples=1)  # (1,)
            paths[i].append(self.nw_data.lids[origin_idx.item()])
            path_links[i].add(origin_idx.item())
            prev_idxs[i] = origin_idx.item()

        goon = False
        for _ in range(logits.shape[1]):
            for i in range(trip_num):
                prev_idx = prev_idxs[i]
                if exp_logits[i, prev_idx, :].sum() == 0 or self.nw_data.edges[self.nw_data.lids[prev_idx]] == d_node_ids[i]:  # 次に移動するリンクがない場合かd_nodeに到達した場合
                    continue
                
                next_idx = torch.multinomial(exp_logits[i, prev_idx, :], num_samples=1)  # (1,)
                if next_idx in path_links[i]:  # すでに通ったリンクの場合
                    continue
                paths[i].append(self.nw_data.lids[next_idx.item()])
                path_links[i].add(next_idx.item())
                prev_idxs[i] = next_idx.item()
                goon = True
            if not goon:
                break
        pp_data = [[i+1, paths[i][j], paths[i][j+1], paths[i][-1]] for i in range(trip_num) for j in range(len(paths[i]) - 1)]  # ID, a, k, b
        pp_data = PP(pd.DataFrame(pp_data, columns=["ID", "a", "k", "b"]), self.nw_data)

        return PPEmbedDataset(pp_data, h_dim=self.h_dim)

    def split_into(self, ratio):
        # ratio: [train, test, validation]
        pp_data = self.pp_data.split_into(ratio)
        return [PPEmbedDataset(pp_data[i], h_dim=self.h_dim) for i in range(len(ratio))]

    def get_fake_batch(self, real_batch, g_output):
        mask = real_batch[1]
        g_output = torch.where(mask > 0, g_output, tensor(-9e15, dtype=torch.float32, device=g_output.device))  # (trip_num, link_num, link_num)
        next_link_prob = F.softmax(g_output, dim=-1)
        next_links = torch.multinomial(next_link_prob.view(-1, g_output.shape[-1]), num_samples=1).squeeze()  # (trip_num * link_num)
        next_links_one_hot = F.one_hot(next_links, num_classes=g_output.shape[-1]).view(g_output.shape)
        return [real_batch[0], real_batch[1], next_links_one_hot, real_batch[3]]


class MeshDataset(Dataset):
    def __init__(self, mesh_traj_data, d, channel=None, return_id=False):
        # d: the number of grids to be considered in route choice
        # state: (bs, prop_dim, 2d+1, 2d+1)
        # context: (bs, context_num, 2d+1, 2d+1)
        # next_state: (bs, 2d+1, 2d+1), 0 or 1
        # mask: (bs, 2d+1, 2d+1), 0 or 1
        # positions: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
        # pis: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
        self.mesh_traj_data = mesh_traj_data
        self.d = d
        self.mnw_data = mesh_traj_data.mnw_data
        self.prop_dim = mesh_traj_data.mnw_data.prop_dim  # prop_dim: output_channel + common_channel
        self.output_channel = len(mesh_traj_data.data_list)
        self.max_agent_num = mesh_traj_data.get_max_agent_num()
        self.trip_nums = mesh_traj_data.get_trip_nums()
        self.times = mesh_traj_data.times

        self.channel = channel
        self.return_id = return_id

        self.common_state = mesh_traj_data.get_state(0)[self.output_channel:, :, :]  # common for all time
        self.common_mask = self.common_state.sum(axis=0) > 0

        self.state = None
        self.context = None
        self.next_state = None
        self.mask = None
        self.positions = None
        self.pis = None

        if channel is not None:
            self.set_channel(channel)

    def __len__(self):
        if self.channel is None:
            raise Exception("Please set channel first.")
        return self.trip_nums[self.channel]

    def __getitem__(self, idx):
        # state: (prop_dim, 2d+1, 2d+1)
        # context: (context_num, 2d+1, 2d+1)
        # next_state: (2d+1, 2d+1)
        # mask: (2d+1, 2d+1)
        # positions: (max_agent_num, output_channel, 2d+1, 2d+1)
        # pis: (max_agent_num, output_channel, 2d+1, 2d+1)
        if self.channel is None:
            raise Exception("Please set channel first.")
        if self.return_id:
            return self.state[idx], self.context[idx], self.next_state[idx], self.mask[idx], self.positions[idx], self.pis[idx], self.aids[idx]
        else:
            return self.state[idx], self.context[idx], self.next_state[idx], self.mask[idx], self.positions[idx], self.pis[idx]

    def set_channel(self, channel):
        self.channel = channel

        self.state = torch.zeros((self.trip_nums[channel], self.prop_dim, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        self.context = torch.zeros((self.trip_nums[channel], 1, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)  # context: distance
        self.next_state = torch.zeros((self.trip_nums[channel], 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        self.mask = torch.zeros((self.trip_nums[channel], 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        self.positions = torch.zeros((self.trip_nums[channel], self.max_agent_num, self.output_channel, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        self.pis = torch.zeros((self.trip_nums[channel], self.max_agent_num, self.output_channel, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        self.aids = torch.zeros((self.trip_nums[channel]), dtype=torch.int32)

        cnt = 0
        for i in range(len(self.mesh_traj_data.times)):
            prop = self.mesh_traj_data.get_state(i)[:self.output_channel]  # (prop_dim, H, W)
            idxs, next_idxs, d_idxs, aids = self.mesh_traj_data.get_action(i)  # (num_channel, num_agents, 2), (num_channel, num_agents, 2), (num_channel, num_agents, 2), (num_channel, num_agents)

            for j in range(idxs[channel].shape[0]):  # num_agents
                min_y = max(0, idxs[channel][j, 0] - self.d)
                max_y = min(prop.shape[1], idxs[channel][j, 0] + self.d + 1)
                min_x = max(0, idxs[channel][j, 1] - self.d)
                max_x = min(prop.shape[2], idxs[channel][j, 1] + self.d + 1)
                min_y_local = min_y - (idxs[channel][j, 0] - self.d)
                max_y_local = max_y - (idxs[channel][j, 0] - self.d)
                min_x_local = min_x - (idxs[channel][j, 1] - self.d)
                max_x_local = max_x - (idxs[channel][j, 1] - self.d)

                next_x = next_idxs[channel][j, 0] - min_x
                next_y = next_idxs[channel][j, 1] - min_y
                next_x = min(max(0, next_x), 2 * self.d)
                next_y = min(max(0, next_y), 2 * self.d)

                d_point = self.mnw_data.cells[idxs[channel][j, 0]][idxs[channel][j, 1]].center
                dist = self.mnw_data.distance_from(d_point)  # (H, W)

                self.state[cnt, :self.output_channel, min_y_local:max_y_local, min_x_local:max_x_local] = tensor(prop[:, min_y:max_y, min_x:max_x])
                self.state[cnt, self.output_channel:, min_y_local:max_y_local, min_x_local:max_x_local] = tensor(self.common_state[:, min_y:max_y, min_x:max_x])
                self.context[cnt, 0, min_y_local:max_y_local, min_x_local:max_x_local] = tensor(dist[min_y:max_y, min_x:max_x])
                self.next_state[cnt, next_y, next_x] = 1
                self.aids[cnt] = aids[channel][j]
                # add other agents at the same time
                self.mask[cnt, min_y_local:max_y_local, min_x_local:max_x_local] = 1.0  #tensor(self.common_state[:, min_y:max_y, min_x:max_x].sum(axis=0) == 0)
                for channel_tmp in range(self.output_channel):
                    for j2 in range(idxs[channel_tmp].shape[0]):  # num_agents
                        if channel_tmp == channel and j2 == j:
                            continue
                        cur_point2 = self.mnw_data.cells[idxs[channel_tmp][j2, 0]][idxs[channel_tmp][j2, 1]].center
                        next_point2 = self.mnw_data.cells[next_idxs[channel_tmp][j2, 0]][next_idxs[channel_tmp][j2, 1]].center
                        cur_point2_dist = tensor(self.mnw_data.distance_from(cur_point2)[min_y:max_y, min_x:max_x])
                        next_point2_dist = tensor(self.mnw_data.distance_from(next_point2)[min_y:max_y, min_x:max_x])
                        if (cur_point2_dist == 0).sum() > 0 and (next_point2_dist == 0).sum() > 0:
                            self.positions[cnt, j2, channel_tmp, min_y_local:max_y_local, min_x_local:max_x_local] = cur_point2_dist
                            self.pis[cnt, j2, channel_tmp, min_y_local:max_y_local, min_x_local:max_x_local] = next_point2_dist
                cnt += 1

    def split_into(self, ratio):
        # ratio: [train, test, validation]
        mesh_traj_data = self.mesh_traj_data.split_into(ratio)
        return [MeshDataset(mesh_traj_data[i], self.d, channel=self.channel) for i in range(len(ratio))]

    def get_mesh_dataset_one_agent(self, channel, aid):
        mesh_traj_data = self.mesh_traj_data.get_mesh_traj_one_agent(channel, aid)
        return MeshDataset(mesh_traj_data, self.d, channel=channel)

    def get_current_state(self, idxs, d_idxs):
        # for simulation
        # idxs: (num_agents, 2)
        # state: (prop_dim, 2d+1, 2d+1)
        # context: (context_num, 2d+1, 2d+1)
        points = [self.mnw_data.cells[idx[0]][idx[1]].center for idx in idxs]
        d_points = [self.mnw_data.cells[d_idx[0]][d_idx[1]].center for d_idx in d_idxs]

        state = torch.zeros((len(idxs), self.prop_dim, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        context = torch.zeros((len(idxs), 1, 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32)
        for i, idx in enumerate(idxs):
            min_x, min_y, max_x, max_y = self.mnw_data.get_surroundings_idxs(points[i], self.d)
            min_y_local = min_y - (idx[0] - self.d)
            max_y_local = max_y - (idx[0] - self.d)
            min_x_local = min_x - (idx[1] - self.d)
            max_x_local = max_x - (idx[1] - self.d)

            state[i, :, min_y_local:max_y_local, min_x_local:max_x_local] = self.mnw_data.get_surroundings_prop(points[i], self.d)
            context[i, 0, min_y_local:max_y_local, min_x_local:max_x_local] = self.mnw_data.distance_from(d_points[i])[min_y:max_y, min_x:max_x]
        return state, context


class MeshDatasetStatic:
    #
    state_mean = 0.05  # float or ndarray
    state_std = 0.1  # float or ndarray
    context_std = 150  # float or ndarray
    def __init__(self, mesh_traj_data, d):
        self.mesh_traj_data = mesh_traj_data  # MeshTrajStatic
        self.d = d

        self.mnw_data = mesh_traj_data.mnw_data
        self.prop_dim = mesh_traj_data.mnw_data.prop_dim  # prop_dim: output_channel + common_channel
        self.output_channel = len(mesh_traj_data.data_list)
        self.trip_nums = mesh_traj_data.get_trip_nums()

        self.common_state =tensor(mesh_traj_data.get_state(), dtype=torch.float32)  # common for all transportation

        self.state = [None for _ in range(self.output_channel)]  # state of the current and next mesh
        self.context = [None for _ in range(self.output_channel)]  # context: distance
        self.next_state = [None for _ in range(self.output_channel)]  # one_hot of next mesh
        self.mask = None
        self.idxs = [None for _ in range(self.output_channel)]  # idxs: (y_idx, x_idx) of current mesh

        self._set_values()

    def split_into(self, ratio):
        # ratio: [train, test, validation]
        mesh_traj_data = self.mesh_traj_data.split_into(ratio)
        return [MeshDatasetStatic(mesh_traj_data[i], self.d) for i in range(len(ratio))]

    def get_sub_datasets(self):
        return [MeshDatasetStaticSub(self.state[channel], self.context[channel], self.next_state[channel], self.mask, self.idxs[channel]) for channel in range(self.output_channel)]

    def get_normalization_params(self):
        # return: state_mean, state_std, context_std list(tensor)
        state_mean = []
        state_std = []
        context_std = []
        for channel in range(self.output_channel):
            state_mean_tmp = self.state[channel].mean(dim=(0, 2, 3), keepdim=True)
            state_std_tmp = self.state[channel].std(axis=(0, 2, 3), keepdim=True)
            context_std_tmp = self.context[channel].std(axis=(0, 2, 3), keepdim=True)

            state_std_tmp[state_std_tmp == 0] = 1.0
            context_std_tmp[context_std_tmp == 0] = 1.0

            state_mean.append(state_mean_tmp)
            state_std.append(state_std_tmp)
            context_std.append(context_std_tmp)
        return state_mean, state_std, context_std

    def normalize(self, state_mean, state_std, context_std):
        for channel in range(self.output_channel):
            self.state[channel] = (self.state[channel] - state_mean[channel]) / state_std[channel]
            min_context = self.context[channel].clone().detach().cpu().numpy().min(axis=(2, 3), keepdims=True)
            self.context[channel] = (self.context[channel] - tensor(min_context, dtype=torch.float32)) / context_std[channel]
            print(f"MeshDatasetStatic normalize: channel: {channel}")
            print(
                f"  Mean  state: {self.state[channel].mean(dim=(0, 2, 3))}, context: {self.context[channel].mean(dim=(0, 2, 3))}")
            print(
                f"  Std   state: {self.state[channel].std(dim=(0, 2, 3))}, context: {self.context[channel].std(dim=(0, 2, 3))}")

    def get_context_mi(self):
        return [mi(self.context[channel].clone().detach().numpy().flatten(), self.next_state[channel].clone().detach().numpy().flatten()) for channel in range(self.output_channel)]

    def _set_values(self):
        self.state = [torch.zeros((self.trip_nums[channel], self.prop_dim, 2 * self.d + 1, 2 * self.d + 1),
                                 dtype=torch.float32, requires_grad=False) for channel in range(self.output_channel)]
        self.context = [torch.full((self.trip_nums[channel], 1, 2 * self.d + 1, 2 * self.d + 1), 5000,
                                   dtype=torch.float32, requires_grad=False) for channel in range(self.output_channel)]  # context: distance
        self.next_state = [torch.zeros((self.trip_nums[channel], 2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32, requires_grad=False) for channel in range(self.output_channel)]
        self.mask = torch.ones((2 * self.d + 1, 2 * self.d + 1), dtype=torch.float32, requires_grad=False)
        self.mask[self.d, self.d] = 0
        self.idxs = [torch.zeros((int(self.trip_nums[channel]), 2), dtype=torch.long) for channel in range(self.output_channel)]
        print(f"MeshDatasetStatic initialize starts. Trip nums: {self.trip_nums}")
        for channel in range(self.output_channel):
            idxs = self.mesh_traj_data.mesh_idxs[channel]  # ["ID", "y_idx", "x_idx", "y_idx_next", "x_idx_next", "d_x", "d_y"]
            aids = np.unique(idxs["ID"].values)
            cnt = 0
            for aid in aids:
                target = idxs[idxs["ID"] == aid]
                d_point = tuple(target[["d_x", "d_y"]].values[0])
                dist = tensor(self.mnw_data.distance_from(d_point))  # (H, W)
                for i in target.index:
                    y_idx = target.loc[i, "y_idx"]
                    x_idx = target.loc[i, "x_idx"]
                    y_idx_next = target.loc[i, "y_idx_next"]
                    x_idx_next = target.loc[i, "x_idx_next"]

                    # index in whole mesh network grid
                    min_y = max(0, y_idx - self.d)
                    max_y = min(self.mnw_data.h_dim, y_idx + self.d + 1)
                    min_x = max(0, x_idx - self.d)
                    max_x = min(self.mnw_data.w_dim, x_idx + self.d + 1)
                    # index in (2*d+1, 2*d+1) grid
                    min_y_local = min_y - (y_idx - self.d)
                    max_y_local = max_y - (y_idx - self.d)
                    min_x_local = min_x - (x_idx - self.d)
                    max_x_local = max_x - (x_idx - self.d)
                    next_y = min(max(0, y_idx_next - y_idx + self.d), 2 * self.d)
                    next_x = min(max(0, x_idx_next - x_idx + self.d), 2 * self.d)

                    self.state[channel][cnt, :, min_y_local:max_y_local, min_x_local:max_x_local] = self.common_state[:, min_y:max_y, min_x:max_x]
                    self.context[channel][cnt, 0, min_y_local:max_y_local, min_x_local:max_x_local] = dist[min_y:max_y, min_x:max_x]

                    self.next_state[channel][cnt, next_y, next_x] = 1
                    self.idxs[channel][cnt, 0] = y_idx
                    self.idxs[channel][cnt, 1] = x_idx

                    cnt += 1

            self.state[channel] = (self.state[channel] - MeshDatasetStatic.state_mean) / MeshDatasetStatic.state_std
            min_context = self.context[channel].clone().detach().cpu().numpy().min(axis=(2, 3), keepdims=True)
            self.context[channel] = (self.context[channel] - tensor(min_context, dtype=torch.float32)) / MeshDatasetStatic.context_std
            print(f"MeshDatasetStatic initialize: channel: {channel}, cnt: {cnt}")
            print(f"  Mean  state: {self.state[channel].mean(dim=(0, 2, 3))}, context: {self.context[channel].mean(dim=(0, 2, 3))}")
            print(f"  Std   state: {self.state[channel].std(dim=(0, 2, 3))}, context: {self.context[channel].std(dim=(0, 2, 3))}")
            print(f"  Mut info  state: "
                  f"{[mi(self.state[channel].clone().detach().numpy()[:, i, :, :].flatten(), self.next_state[channel].clone().detach().numpy().flatten()) for i in range(self.prop_dim)]}, "
                  f"context: {mi(self.context[channel].clone().detach().numpy().flatten(), self.next_state[channel].clone().detach().numpy().flatten())}")


class MeshDatasetStaticSub(Dataset):
    def __init__(self, state, context, next_state, mask, idxs):
        self.state = state
        self.context = context
        self.next_state = next_state
        self.mask = mask
        self.idxs = idxs

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        return self.state[idx], self.context[idx], self.next_state[idx], self.mask, self.idxs[idx]


class PatchDataset(Dataset):
    # all model are preloaded
    def __init__(self, patches):
        # patches: [(C, H, W)]
        self.patches = patches

    def __getitem__(self, item):
        return tensor(self.patches[item], dtype=torch.float32)

    def __len__(self):
        return len(self.patches)


class ImageDatasetBase(Dataset):
    def __init__(self, crop=True, affine=True, transform_coincide=True, flip=True):
        self.crop = crop
        self.affine = affine
        self.transform_coincide = transform_coincide
        self.flip = flip

        self.preprocess = transforms.Compose([
                transforms.PILToTensor(),
                transforms.ToDtype(torch.float32),
                transforms.Normalize(mean=[110.95507746, 110.7599434, 102.66558818], std=[44.63149753, 45.53861262, 45.94082692]),
            ])

        if crop:
            self.rc = transforms.RandomCrop   #(input_shape, pad_if_needed=True)
        if affine:
            self.ra_params = [ # img_size is missing
                (-90, 90),  # degree
                (0.1, 0.1),  # translate
                (0.9, 1.1),  # scale
                (0, 0, 0, 0),  # shear
            ]
        else:
            self.ra_params = [  # img_size is missing
                (0, 0),  # degree
                (0, 0),  # translate
                (1.0, 1.0),  # scale
                (0, 0, 0, 0),  # shear
            ]
        self.ra = transforms.RandomAffine  #(*self.ra_params[0:-1])

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def transform_by_params(self, img_tensor, ra_params, rc_params, h_flip, v_flip):
        # mask: mask for transformed
        # img_tensor[idx] = transformed
        no_channel = False
        if img_tensor.dim() == 2:
            no_channel = True
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

        shape = img_tensor.shape  # (C, H, W)
        transformed = img_tensor
        mask = torch.ones(shape[-2:], dtype=torch.float32)
        idx = tensor(np.arange(shape[-2] * shape[-1], dtype=int).reshape(*shape[-2:]))
        # affine
        if self.affine:
            transformed = transforms.functional.affine(transformed, *ra_params)
            mask = self._get_mask(shape, ra_params)
            idx = self._get_affine_origin_idx(shape, ra_params)  # idx for original tensor
        # crop
        if self.crop:
            transformed = transforms.functional.crop(transformed, *rc_params)
            mask = transforms.functional.crop(mask, *rc_params)
            idx = transforms.functional.crop(idx, *rc_params)
        else:
            transformed = transforms.functional.center_crop(transformed, shape[1:])
            mask = transforms.functional.center_crop(mask, shape[1:])
            idx = transforms.functional.center_crop(idx, shape[1:])
        # flip
        if h_flip:
            transformed = transforms.functional.hflip(transformed)
            mask = transforms.functional.hflip(mask)
            idx = transforms.functional.hflip(idx)
        if v_flip:
            transformed = transforms.functional.vflip(transformed)
            mask = transforms.functional.vflip(mask)
            idx = transforms.functional.vflip(idx)

        if no_channel:
            img_tensor = torch.squeeze(img_tensor, dim=0)
            transformed = torch.squeeze(transformed, dim=0)
        return transformed, mask, idx

    def get_transform_params(self, img_tensor):
        ra_params = None
        rc_params = None
        h_flip = False
        v_flip = False
        input_shape = img_tensor.shape[-2:]

        if self.affine:
            ra_params = self.ra(*self.ra_params).get_params(*self.ra_params, input_shape)
        if self.crop:
            rc_params = self.rc(input_shape, pad_if_needed=True).get_params(img_tensor, input_shape)
        if self.flip:
            h_flip = torch.rand(1) < 0.5
            v_flip = torch.rand(1) < 0.5
        return ra_params, rc_params, h_flip, v_flip

    def affine_by_origin_idx(self, img_tensor, idx):
        # img_tensor (N, C, H, W) or (C, H, W)
        # idx_transformed (N, H, W) or (H, W)
        # img_tensor[idx] = transformed
        # return transformed
        if img_tensor.dim() == 2:
            raise ValueError("img_tensor must have channel dim.")
        idx = idx.to("cpu")
        idx = idx.unsqueeze(-3)
        img_channels = img_tensor.shape[-3]
        img_resolv = img_tensor.shape[-2] * img_tensor.shape[-1]
        # add batch dim
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if idx.dim() == 3:
            idx = idx.unsqueeze(0)

        idx = idx.repeat(1, img_channels, 1, 1)
        null_idx = idx < 0
        batch_num = (np.arange(img_tensor.numel()) / img_resolv).astype(int) * img_resolv  # when img shape (bs, C, H, W)
        batch_num = tensor(batch_num, dtype=int, device="cpu").view(img_tensor.shape)
        idx = idx + batch_num
        idx[null_idx] = 0

        transformed = tensor(np.zeros(img_tensor.shape), dtype=img_tensor.dtype, device=img_tensor.device)
        transformed.reshape(-1)[:] = img_tensor.reshape(-1)[idx.reshape(-1)]
        transformed.reshape(-1)[null_idx.reshape(-1)] = 0

        return transformed

    def padding2size(self, img_tensor, size):
        # center padding
        # size: (H, W)
        # img_tensor: (*, H, W)
        # return: (*, size[0], size[1])
        no_channel = False
        if img_tensor.dim() == 2:
            no_channel = True
            img_tensor = img_tensor.unsqueeze(0)
        if img_tensor.shape[-2] < size[0] or img_tensor.shape[-1] < size[1]:
            pad = (max(0, size[1] - img_tensor.shape[-1]), max(0, size[0] - img_tensor.shape[-2]))
            pad = (pad[0] // 2, pad[1] - pad[1] // 2, pad[0] - pad[0] // 2, pad[1] // 2)
            img_tensor = transforms.functional.pad(img_tensor, pad, padding_mode="constant", fill=0)
        if no_channel:
            img_tensor = img_tensor.squeeze(0)
        return img_tensor

    def _get_mask(self, shape, ra_params):
        # ra_params: (degree, translate, scale, shear, img_size)
        # return: (H, W)
        mask = torch.ones((1, *shape[-2:]), dtype=torch.float32)
        if self.affine:
            mask = transforms.functional.affine(mask, *ra_params)
        mask = mask.squeeze(0)
        return mask

    def _get_affine_origin_idx(self, shape, ra_params):
        # idx of tensor before transform for each pixel after transform
        # image_org[idx_transformed] = image_transformed
        # return (H, W)
        idx = tensor(np.arange(shape[-2] * shape[-1], dtype=int).reshape((1, *shape[-2:])))
        if self.affine:
            idx = transforms.functional.affine(idx, *ra_params,
                                                       interpolation=transforms.InterpolationMode.NEAREST, fill=-1)
        idx = idx.unsqueeze(0)
        return idx

class XImageDataset(ImageDatasetBase):
    # model structure:
    #     if corresponds  base_dir/**/*.png for base_dir in base_dirs (**, * are the same)
    #     else:
    #         base_dir/**/*.png
    def __init__(self, base_dirs, corresponds=False, expansion=1, *args, **kwargs):
        # base_dirs: [dir]
        # corresponds: if True, base_dirs have same structure
        # args, kwargs: crop=True, affine=True, transform_coincide=True, flip=True
        super().__init__(*args, **kwargs)
        self.base_dirs = base_dirs
        self.corresponds = corresponds
        self.expansion = expansion if self.affine or self.crop or self.flip else 1
        self.args = args
        self.kwargs = kwargs

        self._load_data_paths()  # self.files: [[absolute paths]]

    def __len__(self):
        return len(self.files[0]) * self.expansion

    def __getitem__(self, item):
        # (img_tensor, transformed, mask, idx)
        # img_tensor, transformed: (C, H, W)
        # mask, idx: (H, W)
        item = item // self.expansion

        data_list = []
        for i in range(len(self.files)):
            img_tensor = self.preprocess(Image.open(self.files[i][item]))  # (C, H, W)

            if i == 0:  # same transformation for all model
                params = self.get_transform_params(img_tensor)
            transformed, mask, idx = self.transform_by_params(img_tensor, *params)
            data_list.append((img_tensor, transformed, mask, idx))
        return tuple(data_list)

    def split_into(self, ratio):
        # ratio: tuple
        if sum(ratio) > 1.0:
            raise ValueError(f"Sum of ratio must be less than or equal to 1.")
        shuffled_idxs = np.random.permutation(len(self.files[0]))
        num = [int(rat * len(self.files[0])) for rat in ratio]
        cnt = 0
        data_list = []
        for i in range(len(num)):
            tmp_obj = XImageDataset([], corresponds=self.corresponds, expansion=self.expansion, *self.args, **self.kwargs)
            tmp_obj.base_dirs = self.base_dirs
            tmp_idxs = sorted(shuffled_idxs[cnt:cnt+num[i]])
            tmp_obj.files = [[self.files[j][idx] for idx in tmp_idxs] for j in range(len(self.files))]
            data_list.append(tmp_obj)
            cnt += num[i]
        return data_list

    #visualization
    def show_samples(self, num_samples=1):
        items = np.random.choice(len(self), num_samples, replace=False)
        fig = plt.figure(tight_layout=True)
        for i, item in enumerate(items):
            data_tuple = self[item]
            for j, (img_tensor, _, _, _) in enumerate(data_tuple):
                ax = fig.add_subplot(num_samples, len(self.files), i * len(self.files) + j + 1)
                img = img_tensor.permute(1, 2, 0).numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
        plt.show()

    # inside function
    def _load_data_paths(self):
        # self.files: [[path]]
        self.files = [set() for _ in range(len(self.base_dirs))]  # absolute paths
        for i, base_dir in enumerate(self.base_dirs):
            for cur_dir, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(".png"):
                        cur_path = os.path.join(cur_dir, file)
                        cur_path = os.path.relpath(cur_path, base_dir)
                        self.files[i].add(cur_path)  # relative path to base_dir
        if self.corresponds:
            file_set = set.intersection(*self.files)
            file_list = list(file_set)
            self.files = [[os.path.join(base_dir, cur_path) for cur_path in file_list] for base_dir in self.base_dirs]  # (num_basedir, *)
        else:
            self.files = [list(file_set) for file_set in self.files]
            self.files = [[os.path.join(base_dir, cur_path) for i, base_dir in enumerate(self.base_dirs) for cur_path in self.files[i]]]  # (1, *)


class XYImageDataset(XImageDataset):
    # y_file: [Path, ...], csv
    #
    def __init__(self, base_dirs, y_file, *args, **kwargs):
        # x_dirs: [dir]
        # y_file: file
        super().__init__(base_dirs, *args, **kwargs)
        self.y_file = y_file
        if type(y_file) is str:
            y_df = read_csv(y_file)
        else:
            y_df = pd.DataFrame(y_file)
        y_df.set_index("Path", drop=True, inplace=True)

        self.y = [[tensor(y_df.loc[cur_path, :].values, dtype=torch.float32) for cur_path in file_list] for file_list in self.files]  # [[1d tensor]] corresponding to self.files

    def __getitem__(self, item):
        # ((img_tensor, transformed, mask, idx)), [y]
        item = item // self.expansion
        data_tuple = super().__getitem__(item)
        y_tuple = tuple([self.y[i][item] for i in range(len(self.base_dirs))])
        return data_tuple, y_tuple

    def split_into(self, ratio):
        # ratio: tuple
        if sum(ratio) > 1.0:
            raise ValueError(f"Sum of ratio must be less than or equal to 1.")
        shuffled_idxs = np.random.permutation(len(self.files[0]))
        num = [int(rat * len(self.files[0])) for rat in ratio]
        cnt = 0
        data_list = []
        for i in range(len(num)):
            tmp_obj = XYImageDataset([], self.y_file, corresponds=self.corresponds, expansion=self.expansion, *self.args, **self.kwargs)
            tmp_obj.base_dirs = self.base_dirs
            tmp_idxs = sorted(shuffled_idxs[cnt:cnt+num[i]])
            tmp_obj.files = [[self.files[j][idx] for idx in tmp_idxs] for j in range(len(self.files))]
            tmp_obj.y = [[self.y[j][idx] for idx in tmp_idxs] for j in range(len(self.files))]
            data_list.append(tmp_obj)
            cnt += num[i]
        return data_list

    # visualization
    def show_samples(self, num_samples=1):
        items = np.random.choice(len(self), num_samples, replace=False)
        fig = plt.figure(tight_layout=True)
        for i, item in enumerate(items):
            data_tuple, y_tuple = self[item]
            for j, (img_tensor, _, _, _) in enumerate(data_tuple):
                ax = fig.add_subplot(num_samples, len(self.files), i * len(self.files) + j + 1)
                img = img_tensor.permute(1, 2, 0).numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
            print(f"Data {i}: ", y_tuple)
        plt.show()


class XHImageDataset(ImageDatasetBase):
    # x and pixel-wise one-hot
    # base_dirs_x[i]/**/*.png corresponds to base_dirs_h[i]/**/*.png
    def __init__(self, base_dirs_x, base_dirs_h, expansion=1, *args, **kwargs):
        # args, kwargs: crop=True, affine=True, transform_coincide=True, flip=True
        if len(base_dirs_x) != len(base_dirs_h):
            raise ValueError("base_dirs_x and base_dirs_h must have the same length.")
        super().__init__(*args, **kwargs)
        self.preprocess_h = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ToDtype(torch.long)
        ])
        self.base_dirs_x = base_dirs_x
        self.base_dirs_h = base_dirs_h
        self.expansion = expansion
        self.args = args
        self.kwargs = kwargs

        self._load_data_paths()  # self.files_x, self.files_h: [absolute paths]

    def __len__(self):
        return len(self.files_x) * self.expansion

    def __getitem__(self, item):
        # (img_tensor, transformed, mask, idx)
        # img_tensor_x, transformed_x: (C, H, W)
        # img_tensor_h, transformed_h: (H, W) or (3, H, W)
        # mask, idx: (H, W)
        item = item // self.expansion

        img_tensor_x = self.preprocess(Image.open(self.files_x[item]))  # (C, H, W)

        params = self.get_transform_params(img_tensor_x)
        transformed_x, mask_x, idx_x = self.transform_by_params(img_tensor_x, *params)

        img_tensor_h = self.preprocess_h(Image.open(self.files_h[item]))
        transformed_h, mask_h, idx_h = self.transform_by_params(img_tensor_h, *params)

        if img_tensor_h.shape[0] == 1:
            img_tensor_h = torch.squeeze(img_tensor_h, dim=0)
            transformed_h = torch.squeeze(transformed_h, dim=0)
        return (img_tensor_x, transformed_x, mask_x, idx_x), (img_tensor_h, transformed_h, mask_h, idx_h)

    def split_into(self, ratio):
        # ratio: tuple
        if sum(ratio) > 1.0:
            raise ValueError(f"Sum of ratio must be less than or equal to 1.")
        shuffled_idxs = np.random.permutation(len(self.files_x))
        num = [int(rat * len(self.files_x)) for rat in ratio]
        cnt = 0
        data_list = []
        for i in range(len(num)):
            tmp_obj = XHImageDataset([], [], expansion=self.expansion, *self.args, **self.kwargs)
            tmp_obj.base_dirs_x = self.base_dirs_x
            tmp_obj.base_dirs_h = self.base_dirs_h
            tmp_idxs = sorted(shuffled_idxs[cnt:cnt+num[i]])
            tmp_obj.files_x = [self.files_x[idx] for idx in tmp_idxs]
            tmp_obj.files_h = [self.files_h[idx] for idx in tmp_idxs]
            data_list.append(tmp_obj)
            cnt += num[i]
        return data_list

    # visualization
    def show_samples(self, num_samples=1):
        items = np.random.choice(len(self), num_samples, replace=False)
        fig = plt.figure(tight_layout=True)
        for i, item in enumerate(items):
            data_tuple_x, data_tuple_h = self[item]
            for j, (img_tensor, _, _, _) in enumerate([data_tuple_x, data_tuple_h]):
                ax = fig.add_subplot(num_samples, 2, i * 2 + j + 1)
                if len(img_tensor.shape) == 3:
                    img = img_tensor.permute(1, 2, 0).numpy()
                else:
                    img = img_tensor.numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
        plt.show()

    def _load_data_paths(self):
        # self.files: [[path]]
        files_x = [set() for _ in range(len(self.base_dirs_x))]  # realative paths
        for i, base_dir in enumerate(self.base_dirs_x):
            for cur_dir, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(".png"):
                        cur_path = os.path.join(cur_dir, file)
                        cur_path = os.path.relpath(cur_path, base_dir)
                        files_x[i].add(cur_path)  # relative path to base_dir
        files_h = [set() for _ in range(len(self.base_dirs_h))]  # realative paths
        for i, base_dir in enumerate(self.base_dirs_h):
            for cur_dir, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(".png"):
                        cur_path = os.path.join(cur_dir, file)
                        cur_path = os.path.relpath(cur_path, base_dir)
                        files_h[i].add(cur_path)  # relative path to base_dir

        file_list = [list(set.intersection(files_x[i], files_h[i])) for i in range(len(self.base_dirs_x))]
        self.files_x = [os.path.join(self.base_dirs_x[i], cur_path) for i in range(len(file_list)) for cur_path in file_list[i]]  # 1d array
        self.files_h = [os.path.join(self.base_dirs_h[i], cur_path) for i in range(len(file_list)) for cur_path in file_list[i]]   # 1d array


class StreetViewDataset(ImageDatasetBase):
    def __init__(self, config_json, link_prop, input_shape=(256, 256), expansion=1, *args, **kwargs):
        # config_json: {"out_dir": out_dir, "lids": {undir_id: [lids]}, "images": [[relative_path]], "size":HxW}
        # link_prop: columns [LinkID, ...]
        # args, kwargs: crop=True, affine=True, transform_coincide=True, flip=True
        super().__init__(*args, **kwargs)
        if type(config_json) is str:
            config_json = load_json(config_json)
        if type(link_prop) is str:
            link_prop = read_csv(link_prop)

        self.input_shape = input_shape
        self.expansion = expansion
        self.args = args
        self.kwargs = kwargs
        if len(link_prop) == 0:
            return

        self.lids = link_prop['LinkID'].values.astype(int)
        self.y = link_prop.drop('LinkID', axis=1, inplace=False).values
        self.lid2idx = {lid: i for i, lid in enumerate(self.lids)}
        self.files = [[None] * 4 for _ in range(len(self.lids))]  # [[abs path 1, ..., abs path 4]], len(lids)
        for i, tmp_lids in enumerate(config_json["lids"].values()):
            if len(tmp_lids) == 0:
                continue
            for j in range(4):
                if config_json["images"][i][j] is not None:
                    if int(tmp_lids[0]) in self.lid2idx:
                        self.files[self.lid2idx[int(tmp_lids[0])]][j] = os.path.join(config_json["out_dir"], config_json["images"][i][j])
                    else:
                        print(f"Link property for lid {int(tmp_lids[0])} is missing.")
                if len(tmp_lids) > 1:
                    j_inv = (j + 2) % 4
                    if config_json["images"][i][j_inv] is not None:
                        if int(tmp_lids[1]) in self.lid2idx:
                            self.files[self.lid2idx[int(tmp_lids[1])]][j] = os.path.join(config_json["out_dir"], config_json["images"][i][j_inv])
                        else:
                            print(f"Link property for lid {int(tmp_lids[1])} is missing.")

    def __len__(self):
        return len(self.lids) * self.expansion

    def __getitem__(self, item):
        # ((img_tensor, transformed, mask, idx)), y (1d tensor)
        item = item // self.expansion

        data_list = []
        for i in range(4):
            if self.files[item][i] is None:
                img_tensor = torch.zeros((3, *self.input_shape), dtype=torch.float32)
            else:
                img_tensor = self.preprocess(Image.open(self.files[item][i]))  # (C, H, W)
            if i == 0:  # same transformation for all model
                params = self.get_transform_params(img_tensor)
            transformed, mask, idx = self.transform_by_params(img_tensor, *params)
            data_list.append((img_tensor, transformed, mask, idx))

        y_tensor = tensor(self.y[item, :], dtype=torch.float32)
        return tuple(data_list), y_tensor

    def split_into(self, ratio):
        if sum(ratio) > 1.0:
            raise ValueError(f"Sum of ratio must be less than or equal to 1.")
        shuffled_idxs = np.random.permutation(len(self.lids))
        num = [int(rat * len(self.lids)) for rat in ratio]
        cnt = 0
        data_list = []
        for i in range(len(num)):
            tmp_obj = StreetViewDataset({}, {}, expansion=self.expansion, *self.args, **self.kwargs)
            tmp_idxs = sorted(shuffled_idxs[cnt:cnt+num[i]])
            tmp_obj.lids = [self.lids[j] for j in tmp_idxs]
            tmp_obj.lid2idx = {lid: j for j, lid in enumerate(tmp_obj.lids)}
            tmp_obj.files = [self.files[j] for j in tmp_idxs]
            tmp_obj.y = self.y[tmp_idxs, :]

            data_list.append(tmp_obj)
            cnt += num[i]
        return data_list

    # visualization
    def show_samples(self, num_samples=1):
        items = np.random.choice(len(self), num_samples, replace=False)
        fig = plt.figure(tight_layout=True)
        for i, item in enumerate(items):
            data_tuple, y_tensor = self[item]
            for j, (img_tensor, _, _, _) in enumerate(data_tuple):
                ax = fig.add_subplot(num_samples, 4, i * 4 + j + 1)
                if len(img_tensor.shape) == 3:
                    img = img_tensor.permute(1, 2, 0).numpy()
                else:
                    img = img_tensor.numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
            print(f"Data {i}: ", y_tensor)
        plt.show()

class StreetViewXDataset(StreetViewDataset):
    def __init__(self, config_json, base_dirs_x, link_prop, input_shapes=((256, 256), (128, 128)), expansion=1, *args, **kwargs):
        # config_json: {"out_dir": out_dir, "lids": {undir_id: [lids]}, "images": [[relative_path]], "size":HxW}
        # base_dirs_x: [dir] base_dirs_x[i]/{lid}.png
        # link_prop: columns [LinkID, ...]
        # args, kwargs: crop=True, affine=True, transform_coincide=True, flip=True
        super().__init__(config_json, link_prop, input_shape=input_shapes[0], expansion=expansion, *args, **kwargs)
        self.input_shapes = input_shapes
        self.base_dirs_x = base_dirs_x
        self.files_x = [[os.path.join(base_dir, f"{lid}.png") if os.path.exists(os.path.join(base_dir, f"{lid}.png")) else None for lid in self.lids] for base_dir in self.base_dirs_x]

    def __getitem__(self, item):
        # ((img_tensor, transformed, mask, idx)), y (1d tensor)
        item = item // self.expansion
        data_list, y_tensor = super().__getitem__(item)
        data_list_x = []
        for i in range(len(self.files_x)):
            tmp_path = self.files_x[i][item]
            if tmp_path is not None:
                img_tensor_x = self.preprocess(Image.open(tmp_path))
            else:
                img_tensor_x = torch.zeros((3, *self.input_shapes[i+1]), dtype=torch.float32)
            if i == 0:  # same transformation for all model
                params = self.get_transform_params(img_tensor_x)
            transformed_x, mask_x, idx_x = self.transform_by_params(img_tensor_x, *params)
            data_list_x.append((img_tensor_x, transformed_x, mask_x, idx_x))
        return tuple(data_list), tuple(data_list_x), y_tensor

    def split_into(self, ratio):
        if sum(ratio) > 1.0:
            raise ValueError(f"Sum of ratio must be less than or equal to 1.")
        shuffled_idxs = np.random.permutation(len(self.lids))
        num = [int(rat * len(self.lids)) for rat in ratio]
        cnt = 0
        data_list = []
        for i in range(len(num)):
            tmp_obj = StreetViewXDataset({}, [], {}, expansion=self.expansion, *self.args, **self.kwargs)
            tmp_idxs = sorted(shuffled_idxs[cnt:cnt+num[i]])
            tmp_obj.base_dirs_x = self.base_dirs_x
            tmp_obj.lids = [self.lids[j] for j in tmp_idxs]
            tmp_obj.lid2idx = {lid: j for j, lid in enumerate(tmp_obj.lids)}
            tmp_obj.files = [self.files[j] for j in tmp_idxs]
            tmp_obj.files_x = [[self.files_x[k][j] for j in tmp_idxs] for k in range(len(self.files_x))]
            tmp_obj.y = self.y[tmp_idxs, :]

            data_list.append(tmp_obj)
            cnt += num[i]
        return data_list

    # visualization
    def show_samples(self, num_samples=1):
        items = np.random.choice(len(self), num_samples, replace=False)
        fig = plt.figure(tight_layout=True)
        col = 4 + len(self.files_x)
        for i, item in enumerate(items):
            data_tuple, data_tuple_x, y_tensor = self[item]
            for j, (img_tensor, _, _, _) in enumerate(data_tuple):
                ax = fig.add_subplot(num_samples, col, i * col + j + 1)
                if len(img_tensor.shape) == 3:
                    img = img_tensor.permute(1, 2, 0).numpy()
                else:
                    img = img_tensor.numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
            for j, (img_tensor, _, _, _) in enumerate(data_tuple_x):
                ax = fig.add_subplot(num_samples, col, i * col + j + 5)
                if len(img_tensor.shape) == 3:
                    img = img_tensor.permute(1, 2, 0).numpy()
                else:
                    img = img_tensor.numpy()
                min_val = img.min()
                max_val = img.max()
                img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                ax.imshow(img)
                ax.set_axis_off()
            print(f"Data {i}: ", y_tensor)
        plt.show()


# deprecated
class ImageDataset(Dataset):
    # semantic segmentation
    # for unsupervised learning
    def __init__(self, files, additional_files=None, output_files=None, input_shape=(520, 520), num_sample=200):
        # files:[png file]
        # output_files:[npy file] (optional)
        super(ImageDataset, self).__init__()
        self.files = files
        self.additional_files = additional_files
        self.output_files = output_files
        if not additional_files is None:
            if len(files) != len(additional_files):
                print("files and additional_files must have same length.")
                raise(ValueError)
        if not output_files is None:
            if len(files) != len(output_files):
                print("files and output_files must have same length.")
                raise(ValueError)
        self.input_shape = input_shape
        self.num_sample = num_sample

        self.ra_params = [
            (-90, 90),  # degree
            (0.1, 0.1),  # translate
            (0.9, 1.1),  # scale
            (-1, 1, -1, 1),  # shear
            self.input_shape  # img_size
        ]

        self.rc = transforms.RandomCrop(input_shape)  # random crop
        self.ra = transforms.RandomAffine(*self.ra_params[0:-1])

        self._load_data()  # self.input_tensors
        self._split_data(num_sample=num_sample)  # self.cropped_tensor

    def __del__(self):
        if not self.output_files is None:
            for pt_file in self.output_tensors:
                os.remove(pt_file)
            for pt_file in self.cropped_output:
                os.remove(pt_file)

    def __getitem__(self, i):
        tmp_tensor = self.cropped_tensor[i]
        additional_tensor = tensor([]) if len(self.cropped_additional) == 0 else self.cropped_additional[i]
        one_hot_tensor =tensor([]) if len(self.cropped_output) == 0 else torch.load(self.cropped_output[i])
        return self._random_transform(tmp_tensor, additional_tensor, one_hot_tensor)
    
    def __len__(self):
        return self.cropped_tensor.shape[0]

    def affine_by_origin_idx(self, img, idx_transformed):
        # img (N, C, H, W)
        # idx_transformed (N, 1, H, W)
        idx_transformed = idx_transformed.to("cpu")
        img_channels = img.shape[1]
        img_resolv = img.shape[2] * img.shape[3]

        idx_transformed = idx_transformed.repeat(1, img_channels, 1, 1)
        null_idx = idx_transformed < 0
        batch_num = (np.arange(img.numel()) / img_resolv).astype(int) * img_resolv  # when img shape (bs, C, H, W)
        batch_num = tensor(batch_num, dtype=int, device="cpu")
        batch_num = batch_num.view(img.shape)
        idx_transformed = idx_transformed + batch_num.reshape(img.shape)
        idx_transformed[null_idx] = 0

        new_img = tensor(np.zeros(img.shape), dtype=img.dtype, device=img.device)
        new_img.reshape(-1)[:] = img.reshape(-1)[idx_transformed.reshape(-1)]
        new_img.reshape(-1)[null_idx.reshape(-1)] = 0

        return new_img

    def resplit_data(self):
        self._split_data(num_sample=self.num_sample)

    def _load_data(self):
        input_tensors = []
        additional_tensors = []
        # input files
        for file in self.files:
            input_image = Image.open(file)
            input_image = input_image.convert("RGB")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[110.95507746, 110.7599434, 102.66558818], std=[44.63149753, 45.53861262, 45.94082692]),
            ])

            input_tensor = preprocess(input_image)  # (3, H, W)
            input_tensors.append(input_tensor)
        # additional input
        if self.additional_files is not None:
            for file in self.additional_files:
                input_image = Image.open(file)
                input_image = input_image.convert("RGB")
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[5., 5., 5.], std=[2., 2. ,2.]),
                ])

                input_tensor = preprocess(input_image)  # (3, H, W)
                additional_tensors.append(input_tensor)
        self.input_tensors = input_tensors
        self.additional_tensors = additional_tensors

        # output file (supervised)
        output_tensors = []
        self.map_seg = None
        if self.output_files is not None:
            map_seg = MapSegmentation(self.output_files)
            self.map_seg = map_seg
            for file in self.output_files:
                output_tensor = torch.tensor(map_seg.convert_file(file), dtype=torch.uint8)
                pt_file = file.replace(".png", ".pt")
                torch.save(output_tensor, pt_file)
                output_tensors.append(pt_file)
        self.output_tensors = output_tensors

    def _split_data(self, num_sample=200):
        # split model into some pices with shape (3, H, W)
        input_shape = self.input_shape

        cropped_tensor = torch.tensor(np.zeros((num_sample * len(self.input_tensors), 3, input_shape[0], input_shape[1]), dtype=np.float32))
        cropped_additional = torch.tensor(np.zeros((num_sample * len(self.input_tensors), 3, input_shape[0], input_shape[1]), dtype=np.float32))
        class_num = 0
        if not self.output_files is None:
            class_num = self.map_seg.class_num
        cropped_output = []  # path of npy files
        for i in range(len(self.input_tensors)):
            input_tensor = self.input_tensors[i]
            if input_tensor.shape[1] < input_shape[0] or input_tensor.shape[2] < input_shape[1]:
                print("Too small input image.")
                raise(ValueError)
            for num in range(num_sample):
                rc_params = transforms.RandomCrop.get_params(input_tensor, input_shape)
                cropped_tensor[i * num_sample + num, :, :, :] = transforms.functional.crop(input_tensor, *rc_params)
                if not self.additional_files is None:
                    cropped_additional[i * num_sample + num, :, :, :] = transforms.functional.crop(self.additional_tensors[i], *rc_params)
                if not self.output_files is None:
                    pt_tile = self.output_tensors[i].replace(".pt", f"_{num}.pt")
                    cropped_output_tmp = transforms.functional.crop(torch.load(self.output_tensors[i]), *rc_params)
                    torch.save(cropped_output_tmp, pt_tile)
                    cropped_output.append(pt_tile)
        self.cropped_tensor = cropped_tensor
        self.cropped_additional = cropped_additional
        self.cropped_output = cropped_output

    def _get_affine_origin_idx(self, ra_params):
        # idx of tensor before transform for each pixel after transform
        idx = tensor(np.arange(self.input_shape[0] * self.input_shape[1], dtype=int).reshape((1, self.input_shape[0], self.input_shape[1])))
        idx_transformed = transforms.functional.affine(idx, *ra_params, interpolation=transforms.InterpolationMode.NEAREST, fill=-1)

        return idx_transformed

    def _get_mask(self, ra_params):
        # mask for img_tensor
        mask = tensor(np.ones((1, self.input_shape[0], self.input_shape[1]), dtype=np.float32))
        mask = transforms.functional.affine(mask, *ra_params)  # mask for both tensor

        return mask

    def _random_transform(self, img_tensor, additional_tensor, one_hot_tensor):
        # tensor (3, H, W)
        ra_params = self.ra.get_params(*self.ra_params)

        transformed = transforms.functional.affine(img_tensor, *ra_params)
        if len(additional_tensor) == 0:
            additional_transformed = tensor([])
        else:
            additional_transformed = transforms.functional.affine(additional_tensor, *ra_params)
        if len(one_hot_tensor) == 0:
            one_hot_transformed = tensor([])
        else:
            one_hot_tensor = transforms.functional.affine(one_hot_tensor, *ra_params)

        mask = self._get_mask(ra_params)
        idx_transformed = self._get_affine_origin_idx(ra_params) # idx for original tensor

        rand = torch.rand(2)
        if rand[0] < 0.5:
            # vflip
            transformed = transforms.functional.vflip(transformed)
            if len(additional_tensor) > 0:
                additional_transformed = transforms.functional.vflip(additional_transformed)
            if len(one_hot_tensor) > 0:
                one_hot_transformed = transforms.functional.vflip(one_hot_transformed)
            mask = transforms.functional.vflip(mask)
            idx_transformed = transforms.functional.vflip(idx_transformed)
        if rand[1] < 0.5:
            # hflip
            transformed = transforms.functional.hflip(transformed)
            if len(additional_tensor) > 0:
                additional_transformed = transforms.functional.hflip(additional_transformed)
            if len(one_hot_tensor) > 0:
                one_hot_transformed = transforms.functional.hflip(one_hot_transformed)
            mask = transforms.functional.hflip(mask)
            idx_transformed = transforms.functional.hflip(idx_transformed)

        return img_tensor, transformed, additional_tensor, additional_transformed, one_hot_tensor, one_hot_transformed, mask, idx_transformed
    
def calc_loss(model, dataset, dataloader, loss_fn, device, optimizer=None, scheduler=None, model_additional=None, optimizer_additional=None, scheduler_additionl=None, loss_fn_kwargs={}):
    if not optimizer is None:
        model.train()
    else:
        model.eval()

    tmp_loss = []
    tmp_loss_add = []
    for img_tensor, transformed, additional_tensor, additional_transformed, one_hot_tensor, one_hot_transformed, mask, idx_transformed in dataloader:
        img_tensor = img_tensor.to(device)
        transformed = transformed.to(device)
        additional_tensor = additional_tensor.to(device)
        additional_transformed =additional_transformed.to(device)
        one_hot_tensor = one_hot_tensor.to(device)
        one_hot_transformed = one_hot_transformed.to(device)
        mask = mask.to(device)

        if not optimizer is None:
            optimizer.zero_grad()
        if not optimizer_additional is None:
            optimizer_additional.zero_grad()

        y, y2 = _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset)

        loss = loss_fn(y, y2, **loss_fn_kwargs)
        loss_add = tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
        if additional_tensor.max() > 0:
            if not model_additional is None:
                y_add, y2_add = _get_y_y2(model_additional, img_tensor, transformed, mask, idx_transformed, dataset)
                loss_add = loss_fn(y_add, y2_add, **loss_fn_kwargs) # loss for additional file
                loss_add = loss_add + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
                if not optimizer_additional is None:
                    loss_add.backward(retain_graph=True)
                    optimizer_additional.step()
                loss = loss + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
            else:
                y_add, y2_add = _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset)
                loss = loss + loss_fn(y, y2, **loss_fn_kwargs) # loss for additional file
                loss = loss + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
        if one_hot_tensor.max() > 0:
            loss = loss + loss_fn(y, one_hot_tensor, **loss_fn_kwargs)

        if not optimizer is None:
            loss.backward(retain_graph=True)
            optimizer.step()
        if not optimizer_additional is None:
            loss_add.backward(retain_graph=True)
            optimizer_additional.step()

        tmp_loss.append(loss.detach().cpu().item())
        tmp_loss_add.append(loss_add.detach().cpu().item())
    if not scheduler is None:
        scheduler.step()
    if not scheduler_additionl is None:
        scheduler_additionl.step()
    return tmp_loss, tmp_loss_add

def _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset):
    y = model(img_tensor)["out"]  # input (N, C, H, W)
    y = dataset.affine_by_origin_idx(y, idx_transformed)  # g
    y = y * mask
    y2 = model(transformed)["out"]  # filter (N, C, H, W)
    y2 = y2 * mask
    return y, y2


