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
    def __init__(self, data_list, mnw_data, time_resolution=1.0):
        # non-agent property of network is already set.
        # data_list dimension should correspond to the property dimension of mesh network.
        self.time_resolution = datetime.timedelta(seconds=time_resolution)  # [s] time to aggregate data

        start_time = None
        data_list = [read_csv(data) if type(data) == str else data for data in data_list]  # [ID, x, y, time], len: num_transportations
        data_list = [data[mnw_data.contains(data[["x", "y"]].values)] for data in data_list]  # remove data out of mesh network
        # get start_time
        for i in range(len(data_list)):
            data_list[i]["time"] = pd.to_datetime(data_list[i]["time"])
            data_list[i] = data_list[i].sort_values("time")
            if len(data_list[i]) == 0:
                continue
            if start_time is None:
                start_time = data_list[i]["time"].iloc[0]
            else:
                if start_time > data_list[i]["time"].iloc[0]:
                    start_time = data_list[i]["time"].iloc[0]
        self.start_time = start_time.to_pydatetime()
        data_list = self._aggregate_time(data_list)  # aggregate data_list by time_resolution. data_list  # [ID, x, y, time]

        for i in range(len(data_list)):
            ids = np.unique(data_list[i]["ID"].values)
            for tmp_id in ids:
                idx = data_list[i]["ID"] == tmp_id
                data_list[i].loc[idx, ["next_x", "next_y"]] = data_list[i].loc[idx, ["x", "y"]].shift(-1).values
                data_list[i].loc[idx, ["d_x", "d_y"]] = data_list[i].loc[idx, ["x", "y"]].iloc[-1, :].values
            data_list[i] = data_list[i].dropna()
        self.data_list = data_list  # [ID, x, y, next_x, next_y, d_x, d_y, time]

        self.times = sorted(np.unique(np.concatenate([data["time"].values for data in data_list])))

        self.mnw_data = mnw_data  # mesh network
        for i in range(len(data_list)):
            print("MeshTraj {}: data_list num: {}, time num: {}, agent_num: {}".format(i, len(self.data_list[i]), len(self.times), len(np.unique(self.data_list[i]["ID"].values))))

    def __len__(self):
        return len(self.times)

    def get_trip_nums(self):
        return np.array([len(data) for data in self.data_list])

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

    def get_mesh_traj_one_agent(self, channel, aid):
        data_list = [data[data["ID"] == aid] if i == channel else data for i, data in enumerate(self.data_list)]
        return MeshTraj(data_list, self.mnw_data, time_resolution=self.time_resolution.total_seconds())

    def get_state(self, idx):
        t = self.times[idx]
        data_list = [data[data["time"] == t] for data in self.data_list]
        prop = self.mnw_data.get_prop_array()  # (max_y_idx - min_y_idx, max_x_idx - min_x_idx, prop_dim)
        for i, data in enumerate(data_list):
            idxs = self.mnw_data.get_idx(data[["x", "y"]].values)  # (num_points, 2)
            for tmp_idx in idxs:
                prop[i, tmp_idx[0], tmp_idx[1]] += 1
        return prop

    def get_action(self, idx):
        # idx: time index
        # idxs: (num_agents, 2) current position index
        # next_idxs: (num_agents, 2) next position index
        # d_idxs: (num_agents, 2) destination index
        # aids: (num_agents) agent id
        t = self.times[idx]
        data_list = [data[data["time"] == t] for data in self.data_list]
        idxs = []
        next_idxs = []
        d_idxs = []
        aids = []
        for i, data in enumerate(data_list):
            idxs_tmp = self.mnw_data.get_idx(data[["x", "y"]].values)  # (num_agents, 2)
            next_idxs_tmp = self.mnw_data.get_idx(data[["next_x", "next_y"]].values)  # (num_agents, 2)
            d_idx_tmp = self.mnw_data.get_idx(data[["d_x", "d_y"]].values)  # (num_agents, 2)
            idxs.append(idxs_tmp)
            next_idxs.append(next_idxs_tmp)
            d_idxs.append(d_idx_tmp)
            aids.append(data["ID"].values)
        return idxs, next_idxs, d_idxs, aids

    def get_traj_idx_one_agent(self, channel, aid):
        data = self.data_list[channel][self.data_list[channel]["ID"] == aid]
        return self.mnw_data.get_idx(data[["x", "y"]].values)  # (num_points, 2)

    def get_aggregated(self, paths=None):
        if paths is not None:
            for i in range(len(self.data_list)):
                self.data_list[i].to_csv(paths[i], index=False)
        return self.data_list

    # visualize
    def show_action(self, idx, save_path=None):
        # idx: time index
        fig = plt.figure(figsize=(6.4, 2.4 * self.mnw_data.prop_dim))
        prop = self.get_state(idx)
        idxs, next_idxs, d_idxs, aids = self.get_action(idx)
        for i in range(self.mnw_data.prop_dim):
            ax = fig.add_subplot(self.mnw_data.prop_dim, 1, i + 1)
            ax.set_title("Prop {}".format(i))
            ax.set_aspect('equal')
            ax.imshow(prop[i, :, :])
            if i < len(idxs):
                for j in range(len(idxs[i])):
                    ax.quiver(idxs[i][j][1], idxs[i][j][0], next_idxs[i][j][1] - idxs[i][j][1], next_idxs[i][j][0] - idxs[i][j][0], angles='xy', scale_units='xy', scale=1, color="red")
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()

    def show_actions(self, prop_idx, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_title("Prop {}".format(prop_idx))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        xs = []
        ys = []
        ts = []
        us = []
        vs = []
        ws = []
        for idx in range(len(self)):
            idxs, next_idxs, d_idxs, aids = self.get_action(idx)
            if prop_idx < len(idxs):
                for j in range(len(idxs[prop_idx])):
                    xs.append(idxs[prop_idx][j][1])
                    ys.append(idxs[prop_idx][j][0])
                    ts.append(idx)
                    us.append(next_idxs[prop_idx][j][1] - idxs[prop_idx][j][1])
                    vs.append(next_idxs[prop_idx][j][0] - idxs[prop_idx][j][0])
                    ws.append(1)
        ax.quiver(xs, ys, ts, us, vs, ws, linewidth=1.5, length=1.0, arrow_length_ratio=0.1,
                                  color="red")

        if save_path is not None:
            fig.savefig(save_path)
        plt.show()

    def show_agent_num(self, save_path=None):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Agent num")
        ax.set_xlabel("t")
        ax.set_ylabel("num")
        for channel in range(len(self.data_list)):
            num = [(self.data_list[channel]["time"] == self.times[idx]).sum() for idx in range(len(self))]
            ax.plot(self.times, num, label=f"channel {channel}")
        ax.legend()
        if save_path is not None:
            fig.savefig(save_path)
        plt.show()

    # inside function
    def _aggregate_time(self, data_list):
        new_data_list = []
        for i in range(len(data_list)):
            time_step = data_list[i]["time"]
            time_step = time_step.apply(lambda x: (x - self.start_time) // self.time_resolution)

            ids = np.unique(data_list[i]["ID"].values)
            new_val = []
            for tmp_id in ids:
                idx = data_list[i]["ID"] == tmp_id
                ts = sorted(np.unique(time_step[idx].values))
                if len(ts) < 2:
                    continue
                for t in ts:
                    target = data_list[i].loc[idx & (time_step == t), :]  # [ID, x, y, time]
                    new_val.append([tmp_id, np.mean(target["x"].values), np.mean(target["y"].values), t * self.time_resolution + self.start_time])

            new_data_list.append(pd.DataFrame(new_val, columns=["ID", "x", "y", "time"]))
        return new_data_list


class MeshTrajStatic:
    def __init__(self, data_list, mnw_data):
        # non-agent property of network is already set.
        # information of agents is not included in mnw_data property.
        data_list = [read_csv(data) if type(data) == str else data for data in data_list]  # [ID, x, y, time], len: num_transportations
        data_list = [data[mnw_data.contains(data[["x", "y"]].values)] for data in data_list]  # remove data out of mesh network
        self.data_list = data_list
        self.mnw_data = mnw_data

        self._set_mesh_idxs()  # set mesh transition data for each agent

    def _set_mesh_idxs(self):
        self.mesh_idxs = []  # list(df[ID, idx, idx_next, d_x, d_y])
        for i, data in enumerate(self.data_list):
            ids = np.unique(data["ID"].values)
            mesh_idx = []
            for tmp_id in ids:
                target = data[data["ID"] == tmp_id]
                idx = self.mnw_data.get_idx(target[["x", "y"]].values)  # np.array(num_points, 2)
                # remove the duplicated idx
                diff = np.diff(idx, axis=0).sum(1)
                diff = np.concatenate([np.array([True]), (diff != 0).astype(bool)])
                idx = idx[diff]





