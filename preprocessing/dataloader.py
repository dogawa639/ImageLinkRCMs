import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.sparse import csr_matrix, coo_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os

__all__ = ["GridDataset", "PPEmbedDataset"]


class GridDataset(Dataset):
    # 3*3 grid choice set
    def __init__(self, pp_data, nw_data, normalize=True):
        # nw_data: NWDataCNN
        f = nw_data.feature_num
        c = nw_data.context_feature_num
        # inputs: [sum(sample_num), f+c, 3, 3]
        # masks: [sum(sample_num), 9]
        # next_links: [sum(sample_num), 9]
        inputs = np.zeros((0, f + c, 3, 3), dtype=np.float32)
        masks = np.zeros((0, 9), dtype=np.float32)
        next_links = np.zeros((0, 9), dtype=np.float32)
        for tid in pp_data.tids:
            path = pp_data.path_dict[tid]["path"]
            d_node_id = pp_data.path_dict[tid]["d_node"]
            if len(path) == 0:
                continue
            sample_num = len(path) - 1
            trip_input = np.zeros((sample_num, f + c, 3, 3), dtype=np.float32)  # [link, state, state]
            trip_mask = np.zeros((sample_num,9), dtype=np.float32)  # [link, mask]
            trip_next_link = np.zeros((sample_num,9), dtype=np.float32)  # [link, one_hot]

            for i in range(sample_num):
                tmp_link = path[i]
                next_link = path[i+1]
                feature_mat, _ = nw_data.get_feature_matrix(tmp_link, normalize=normalize)
                context_mat, action_edge = nw_data.get_context_matrix(tmp_link, d_node_id, normalize=normalize)

                trip_input[i, 0:f, :, :] = feature_mat
                trip_input[i, f:f+c, :, :] = context_mat
                trip_mask[i, :] = [len(edges) > 0 for edges in action_edge]
                trip_next_link[i, :] = [next_link in edges for edges in action_edge]

            trip_input = trip_input[np.isnan(trip_input).sum(axis=(1, 2, 3)) == 0]
            if len(trip_input) == 0:
                continue
            inputs = np.concatenate((inputs, trip_input), axis=0)
            masks = np.concatenate((masks, trip_mask), axis=0)
            next_links = np.concatenate((next_links, trip_next_link), axis=0)

        self.inputs = inputs
        self.masks = masks
        self.next_links = next_links

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.masks[idx], self.next_links[idx]  # [f+c, 3, 3], [9], [9]


class PPEmbedDataset(Dataset):
    # Grobal state (link_num, graph_node_feature_num) + PP pos embedding, PP adj matrix (link_num, link_num)
    def __init__(self, pp_data, grobal_state, nw_data):
        # grobal state: output of GNN
        self.pp_data = pp_data
        self.grobal_state = grobal_state
        self.nw_data = nw_data
        self.link_num, self.feature_num = grobal_state.shape

        self.pp_pos_embeddings = [self.get_pp_pos_embedding(path) for path in pp_data.load_edge_list()]
        self.pp_adj_matrices = [self.get_pp_adj_matrix(path) for path in pp_data.load_edge_list()]

    def __len__(self):
        return len(self.pp_data)
    
    def __getitem__(self, idx):
        return self.grobal_state + self.pp_pos_embeddings[idx].toarray(), self.pp_adj_matrices[idx].toarray()  # [link_num, feature_num], [link_num, link_num]

    def get_pp_pos_embedding(self, path):
        # path: [link_id] or [edge obj]
        pos_embedding = np.zeros((self.link_num, self.feature_num), dtype=np.float32)
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for i, edge in enumerate(path):
            if type(edge) != int:
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
            if type(edge) != int:
                edge = edge.id
            if i < len(path) - 1:
                row.append(lid2idx[edge])
                col.append(lid2idx[path[i+1]])
        adj_matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(self.link_num, self.link_num), dtype=np.float32).to_csr()
        return adj_matrix
    

    


