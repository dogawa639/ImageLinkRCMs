import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import os

from models.unet import UNet
from models.general import FF, SLN

# still debugging
class MeshSimulator:
    def __init__(self, generators, dataset, device="cpu"):
        self.generators = generators
        self.dataset = dataset

        self.mnw_data = dataset.mnw_data
        self.output_channel = dataset.output_channel
        self.device = device

        self.current_agents = None
        self.finish_agents = None

    def initialize(self):
        self.current_agents = [[] for _ in range(self.output_channel)]  # [[{"id":id, "path": [idx], "d_idx": idx, "start_timestep": timestep}]]
        self.finish_agents = [[] for _ in range(self.output_channel)]
        for i in range(self.output_channel):
            self.mnw_data.clear_prop(i)

    def simulate(self, od_data, max_timestep, out_dir=None):
        # dataset: MeshDataset
        # od_data: [{timestep : [(o_idx_x, o_idx_y, d_idx_x, d_idx_y)]}]. len(od_data) == output_channel
        # output: [df([ID, x, y, next_x, next_y, d_x, d_y, timestep])]
        self.initialize()
        for timestep in range(max_timestep):
            for channel in range(self.output_channel):
                self._add_agent(od_data[channel], timestep, channel)

            for channel in range(self.output_channel):
                self._update(channel)
        result = []
        for channel in range(self.output_channel):
            result.append(self._get_result(channel))
        if out_dir is not None:
            for i, df in enumerate(result):
                df.to_csv(os.path.join(out_dir, f"transportation_{i}.csv"), index=False)
        return result

    # inside function
    def _add_agent(self, od_data, timestep, channel):
        # dataset: MeshDataset
        # od_data: {timestep : [(o_idx_x, o_idx_y, d_idx_x, d_idx_y)]}
        if timestep not in od_data:
            return
        for i, od in enumerate(od_data[timestep]):
            tmp_id = len(self.current_agents[channel]) + len(self.finish_agents[channel]) + 1
            self.current_agents[channel].append({"id": tmp_id, "path": [(od[0], od[1])], "d_idx": (od[2], od[3]), "start_timestep": timestep})
            self.mnw_data.add_prop(np.array([self.mnw_data.get_center_points((od[0], od[1]))]), np.array([channel]))

    def _update(self, channel):
        # dataset: MeshDataset
        # od_data: {timestep : [(o_idx_x, o_idx_y, d_idx_x, d_idx_y)]}
        # output: [ID, x, y, next_x, next_y, d_x, d_y, timestep]
        # move agents
        inputs = None
        masks = None
        min_xs = []
        min_ys = []
        if len(self.current_agents[channel]) == 0:
            return
        for agent in self.current_agents[channel]:
            point = self.mnw_data.get_center_points(agent["path"][-1])
            d_point = self.mnw_data.get_center_points(agent["d_idx"])

            min_x, min_y, max_x, max_y = self.mnw_data.get_surroundings_idxs(point, self.dataset.d)
            min_y_local = min_y - (agent["path"][-1][0] - self.dataset.d)
            max_y_local = max_y - (agent["path"][-1][0] - self.dataset.d)
            min_x_local = min_x - (agent["path"][-1][1] - self.dataset.d)
            max_x_local = max_x - (agent["path"][-1][1] - self.dataset.d)

            input = np.zeros((self.mnw_data.prop_dim + 1, 2 * self.dataset.d + 1, 2 * self.dataset.d + 1), dtype=np.float32)  # (total_feature, 2d+1, 2d+1)
            state = self.mnw_data.get_surroundings_prop(point, self.dataset.d)  # (prop_dim, max_y_idx - min_y_idx, max_x_idx - min_x_idx)
            dist = self.mnw_data.distance_from(d_point)[min_y:max_y, min_x:max_x]  # (max_y_idx - min_y_idx, max_x_idx - min_x_idx)
            input[:self.mnw_data.prop_dim, min_y_local:max_y_local, min_x_local:max_x_local] = state
            input[self.mnw_data.prop_dim, min_y_local:max_y_local, min_x_local:max_x_local] = dist
            input = np.expand_dims(input, axis=0)  # (1, total_feature, 2d+1, 2d+1)
            mask = np.zeros((1,  2 * self.dataset.d + 1, 2 * self.dataset.d + 1), dtype=np.float32) # (1, 2d+1, 2d+1)
            mask[:, min_y_local:max_y_local, min_x_local:max_x_local] = state[self.output_channel:, :, :].sum(axis=0, keepdims=True) == 0.0
            if inputs is None:
                inputs = input
            else:
                inputs = np.concatenate((inputs, input), axis=0)
            if masks is None:
                masks = mask
            else:
                masks = np.concatenate((masks, mask), axis=0)
            min_xs.append(min_x)
            min_ys.append(min_y)
        inputs = tensor(inputs, dtype=torch.float32).to(self.device)  # (agent_num, total_feature, 2d+1, 2d+1)
        masks = tensor(masks, dtype=torch.float32).to(self.device)  # (agent_num, 2d+1, 2d+1)
        out = self.generators[channel](inputs)  # output: (bs, 2d+1, 2d+1)
        out = torch.where(masks > 0, out, tensor(-9e15, dtype=torch.float32, device=out.device))
        pi = F.softmax(out.view(out.shape[0], -1), dim=1)  # (agent_num, 2d+1 * 2d+1)
        elem = torch.multinomial(pi, 1).squeeze(dim=1).cpu().detach().numpy()

        rm_idx = []
        for i, agent in enumerate(self.current_agents[channel]):
            new_idx = min_ys[i] + elem[i] // (self.dataset.d * 2 + 1), min_xs[i] + elem[i] % (self.dataset.d * 2 + 1)
            if agent["d_idx"] == new_idx:
                self.mnw_data.remove_prop(self.mnw_data.get_center_points(agent["path"][-1]), np.array([channel]))
                self.finish_agents[channel].append(agent)
                rm_idx.append(i)
            else:
                self.mnw_data.move_props(np.array([self.mnw_data.get_center_points(agent["path"][-1])]), np.array([self.mnw_data.get_center_points(new_idx)]), np.array([channel]))
                agent["path"].append(new_idx)
        self.current_agents[channel] = [agent for i, agent in enumerate(self.current_agents[channel]) if i not in rm_idx]

    def _get_result(self, channel):
        # dataset: MeshDataset
        # od_data: {timestep : [(o_idx_x, o_idx_y, d_idx_x, d_idx_y)]}
        # output: [ID, x, y, next_x, next_y, d_x, d_y, timestep]
        result = []
        for agent in self.finish_agents[channel]:
            for i in range(len(agent["path"]) - 1):
                result.append([agent["id"], *self.mnw_data.get_center_points(agent["path"][i]), *self.mnw_data.get_center_points(agent["path"][i + 1]), *self.mnw_data.get_center_points(agent["d_idx"]), agent["start_timestep"] + i])
        for agent in self.current_agents[channel]:
            for i in range(len(agent["path"]) - 1):
                result.append([agent["id"], *self.mnw_data.get_center_points(agent["path"][i]), *self.mnw_data.get_center_points(agent["path"][i + 1]), *self.mnw_data.get_center_points(agent["d_idx"]), agent["start_timestep"] + i])
        return pd.DataFrame(result, columns=["ID", "x", "y", "next_x", "next_y", "d_x", "d_y", "timestep"])

