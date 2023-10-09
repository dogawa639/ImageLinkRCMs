import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["PP"]

class PP:
    def __init__(self, data_file, network):
        # all property of network is already set.
        self.data = pd.read_csv(data_file)  # [ID, a, k, b(last link)]
        self.network = network

        self.path_dict = PP.get_path_dict(self.data, self.network)  # {"tid": {"path": [link_id], "d_node": node_id}}
        self.tids = list(self.path_dict.keys())

    def __len__(self):
        return len(self.path_dict)

    def load_edge_list(self):
        idx = 0
        while True:
            if idx >= len(self.path_dict):
                break
            tid = self.tids[idx]
            path = self.path_dict[tid]["path"]
            edge_list = [self.network.edges[link_id] for link_id in path]

            yield edge_list
            idx += 1

    @staticmethod
    def get_path_dict(data, network):
        # data is soreted by ID and time
        trip_ids = sorted(data["ID"].unique())
        real_data = dict()

        tid = 1
        cnt = 0
        for t in trip_ids:
            target = data.loc[(data["ID"] == t), :]
            val = target[["k", "a"]].values

            tmp_path = [(i, v[0]) for i, v in enumerate(val) if v[0] in network.edges]
            tmp_path2 = {i: v[1] for i, v in enumerate(val) if v[1] in network.edges}
            if len(tmp_path) == 0:
                continue
            cnt += 1
            path = []
            for idx, (i, tmp_edge) in enumerate(tmp_path):
                if i - 1 in tmp_path2:  # if the previous link is in the network
                    path.append(tmp_edge)
                    if i not in tmp_path2:  # if the next link is not in the network
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": network.edges[last_link].end.id}
                        tid += 1
                    elif idx == len(tmp_path) - 1:  # if the next link is in the network and this is the last link
                        path.append(tmp_path2[i])
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": network.edges[last_link].end.id}
                        tid += 1
                else:  # if the previous link is not in the network
                    path = [tmp_edge]
        print("path_samples: ", tid - 1)
        print("trip_num: ", cnt)
        return real_data