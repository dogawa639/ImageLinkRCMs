import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import norm

from sklearn.neighbors import KDTree

import geopandas as gpd
import shapely

import datetime

from utility import *
__all__ = ["MeshTraj"]


class MeshTraj:
    def __init__(self, data_list, mnw_data):
        # non-agent property of network is already set.
        # data_list dimension should correspond to the property dimension of mesh network.
        data_list = [read_csv(data) if type(data) == str else data for data in data_list]  # [ID, x, y, time, type], len: num_transportations
        data_list = [data[mnw_data.contains(data[["x", "y"]].values)] for data in data_list]  # remove data out of mesh network
        for i in range(len(data_list)):
            data_list[i]["time"] = pd.to_datetime(data_list[i]["time"])
            data_list[i] = data_list[i].sort_values("time")
            ids = np.unique(data_list[i]["ID"].values)
            for tmp_id in ids:
                idx = data_list[i]["ID"] == tmp_id
                data_list[i].loc[idx, ["next_x", "next_y"]] = data_list[i].loc[idx, ["x", "y"]].shift(-1).values
                data_list[i].loc[idx, ["d_x", "d_y"]] = data_list[i].loc[idx, ["x", "y"]].iloc[-1, :].values
                data_list[i] = data_list[i].dropna()
        self.times = sorted(np.unique(np.concatenate([data["time"].values for data in data_list])))

        self.data_list = data_list
        self.mnw_data = mnw_data  # mesh network

    def __len__(self):
        return len(self.times)

    def get_trip_nums(self):
        return np.array([len(np.unique(data["ID"].values)) for data in self.data_list])

    def get_max_agent_num(self):
        return np.max([np.sum([np.sum(self.data_list[i]["time"] == t) for i in range(len(self.data_list))]) for t in self.times])

    def split_into(self, ratio):
        # ratio: [train, val, test] sum = 1
        if sum(ratio) > 1:
            raise ValueError("The sum of ratio must be less or equal to 1.")
        total_data_list = [[] for _ in range(len(ratio))]
        for i in range(len(self.data_list)):
            ids = np.unique(self.data_list[i]["ID"].values)
            trip_nums = (len(ids) * np.array(ratio)).astype(int)
            trip_nums = trip_nums.cumsum()
            trip_nums[-1] = len(ids)
            ids_shuffled = np.random.permutation(ids)
            for j in range(len(ratio)):
                if j == 0:
                    tmp_ids = ids_shuffled[:trip_nums[j]]
                else:
                    tmp_ids = ids_shuffled[trip_nums[j - 1]:trip_nums[j]]
                tmp_data = self.data_list[i].loc[self.data_list[i]["ID"].isin(tmp_ids), :]
                total_data_list[j].append(tmp_data)
        result = []
        for j in range(len(ratio)):
            result.append(MeshTraj(total_data_list[j], self.mnw_data))
        return result

    def get_state(self, idx):
        t = self.times[idx]
        data_list = [data[data["time"] == t] for data in self.data_list]
        prop = self.mnw_data.get_prop_array()  # (max_y_idx - min_y_idx, max_x_idx - min_x_idx, prop_dim)
        for i, data in enumerate(data_list):
            idxs = self.mnw_data.get_idx(data[["x", "y"]].values)  # (num_points, 2)
            for idx in idxs:
                prop[idx[0], idx[1], i] += 1
        return prop

    def get_action(self, idx):
        t = self.times[idx]
        data_list = [data[data["time"] == t] for data in self.data_list]
        idxs = []
        next_idxs = []
        d_idxs = []
        for i, data in enumerate(data_list):
            idxs_tmp = self.mnw_data.get_idx(data[["x", "y"]].values)  # (num_agents, 2)
            next_idxs_tmp = self.mnw_data.get_idx(data[["next_x", "next_y"]].values)  # (num_agents, 2)
            d_idx_tmp = self.mnw_data.get_idx(data[["d_x", "d_y"]].values)  # (num_agents, 2)
            idxs.append(idxs_tmp)
            next_idxs.append(next_idxs_tmp)
            d_idxs.append(d_idx_tmp)
        return idxs, next_idxs, d_idxs

    # visualize
    def show_action(self, idx, save_path=None):
        fig = plt.figure(figsize=(6.4, 2.4 * self.mnw_data.prop_dim))
        prop = self.get_state(idx)
        idxs, next_idxs = self.get_action(idx)
        for i in range(self.mnw_data.prop_dim):
            ax = fig.add_subplot(self.mnw_data.prop_dim, 1, i + 1)
            ax.set_title("Prop {}".format(i))
            ax.set_aspect('equal')
            ax.imshow(prop[:, :, i])
            if i < len(idxs):
                for j in range(len(idxs[i])):
                    ax.quiver(idxs[i][j][1] + 0.5, idxs[i][j][0] + 0.5, next_idxs[i][j][1] - idxs[i][j][1], next_idxs[i][j][0] - idxs[i][j][0], angles='xy', scale_units='xy', scale=1, color="red")
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)

