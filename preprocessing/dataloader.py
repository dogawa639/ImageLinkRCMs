import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import os

__all__ = ["GridDataset"]


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


class EmbeddingDataset(Dataset):
    def __init__(self, pp_data, nw_data, normalize=True):
        # nw_data: NWDataCNN
        f = nw_data.feature_num
        # inputs: [sum(sample_num), f+c]
        # next_links: [sum(sample_num)]
        inputs = np.zeros((0, f + c), dtype=np.float32)
        next_links = np.zeros((0), dtype=np.float32)
        for tid in pp_data.tids:
            path = pp_data.path_dict[tid]["path"]
            d_node_id = pp_data.path_dict[tid]["d_node"]
            if len(path) == 0:
                continue
            sample_num = len(path) - 1
            trip_input = np.zeros((sample_num, f + c), dtype=np.float32)