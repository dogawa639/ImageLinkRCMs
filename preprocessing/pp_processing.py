import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["PP"]

class PP:
    def __init__(self, data, network):
        # all property of network is already set.
        if type(data) == str:
            data = pd.read_csv(data)  # [ID, a, k, b(last link)]
        self.data = data  # [ID, a, k, b(last link)]
        self.network = network

        self.path_dict, self.tid2org, self.org2tid = PP.get_path_dict(self.data, self.network)  # {"tid": {"path": [link_id], "d_node": node_id}}
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

    def split_into(self, ratio):
        # ratio: [train, val, test] sum = 1
        if sum(ratio) != 1:
            raise ValueError("The sum of ratio must be 1.")
        trip_nums = (len(self.path_dict) * np.array(ratio)).astype(int)
        trip_nums = trip_nums.cumsum()
        trip_nums[-1] = len(self.path_dict)
        tids_shuffled = np.random.permutation(self.tids)
        result = []
        for i in range(len(ratio)):
            if i == 0:
                tmp_tids = tids_shuffled[:trip_nums[i]]
            else:
                tmp_tids = tids_shuffled[trip_nums[i - 1]:trip_nums[i]]
            tmp_orgs = [self.tid2org[tid] for tid in tmp_tids]
            tmp_data = self.data.loc[self.data["ID"].isin(tmp_orgs), :]
            tmp_pp = PP(tmp_data, self.network)
            result.append(tmp_pp)
        return result


    @staticmethod
    def get_path_dict(data, network):
        # real_data: {"tid": {"path": [link_id], "d_node": node_id}}
        # tid2org: {"tid": "org_id"}
        # org2tid: {"org_id": ["tid"]}
        # data is soreted by time
        trip_ids = sorted(data["ID"].unique())
        real_data = dict()
        tid2org = dict()
        org2tid = {org: [] for org in trip_ids}

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
                        tid2org[tid] = t
                        org2tid[t].append(tid)
                        tid += 1
                    elif idx == len(tmp_path) - 1:  # if the next link is in the network and this is the last link
                        path.append(tmp_path2[i])
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": network.edges[last_link].end.id}
                        tid2org[tid] = t
                        org2tid[t].append(tid)
                        tid += 1
                else:  # if the previous link is not in the network
                    path = [tmp_edge]
        print("path_samples: ", tid - 1)
        print("trip_num: ", cnt)
        return real_data, tid2org, org2tid


# test
if __name__ == "__main__":
    import configparser
    from preprocessing.network_processing import *

    CONFIG = "/Users/dogawa/PycharmProjects/GANs/config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    pp_path = read_data["pp_path"]

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    pp_data = PP(pp_path, nw_data)
    pp_train, pp_test = pp_data.split_into([0.8, 0.2])
    print(len(pp_data), len(pp_train), len(pp_test))

    print(pp_data.path_dict[1])
    print(pp_data.tids[1])

