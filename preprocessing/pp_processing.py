import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy.sparse import csr_matrix
from scipy.stats import norm

from sklearn.neighbors import KDTree

import geopandas as gpd
import shapely

import datetime

from utility import *

__all__ = ["PP"]


class PP:
    def __init__(self, data, network):
        # all property of network is already set.
        if type(data) == str:
            data = pd.read_csv(data)  # [ID, a, k, b(last link)] or [ID, tid, a, k, b(last link)]
        self.data = data  # [ID, a, k, b(last link)] or [ID, tid, a, k, b(last link)]
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

    def write_geo_file(self, file_path, driver=None):
        # [seq, tid, geometry]
        result = []
        for tid, trip in self.path_dict.items():
            geom = [(self.network.edges[link_id].start.lon, self.network.edges[link_id].start.lat) for link_id in trip["path"]]
            geom.append((self.network.edges[trip["path"][-1]].end.lon, self.network.edges[trip["path"][-1]].end.lat))
            geom = shapely.geometry.LineString(geom)
            if "tid" in trip:
                result.append([tid, trip["tid"], geom])
            else:
                result.append([tid, -1, geom])
        gdf = gpd.GeoDataFrame(result, columns=["seq", "ID", "geometry"])
        gdf.to_file(file_path, driver=driver)

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
            org_tid = None
            if "tid" in data.columns:
                org_tid = target["tid"].values[0]

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
                        if org_tid is not None:
                            real_data[tid]["tid"] = org_tid
                        tid2org[tid] = t
                        org2tid[t].append(tid)
                        tid += 1
                    elif idx == len(tmp_path) - 1:  # if the next link is in the network and this is the last link
                        path.append(tmp_path2[i])
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": network.edges[last_link].end.id}
                        if org_tid is not None:
                            real_data[tid]["tid"] = org_tid
                        tid2org[tid] = t
                        org2tid[t].append(tid)
                        tid += 1
                else:  # if the previous link is not in the network
                    path = [tmp_edge]
        print("path_samples: ", tid - 1)
        print("trip_num: ", cnt)
        return real_data, tid2org, org2tid

    @staticmethod
    def map_matching(trip_path, loc_path, nw_data, utm_num, thresh= 60, loc_time_format="%Y-%m-%d %H:%M:%S.%f", trip_time_format="%Y-%m-%d %H:%M:%S"):
        # thresh: sec to regard as the separated trip
        coord = Coord(utm_num)
        trip_data = pd.read_csv(trip_path)  # ID,ユーザーID,作成日時,出発時刻,到着時刻,更新日時,有効性,目的コード（active）
        loc_data = pd.read_csv(loc_path)  # ID,accuracy,bearing,speed,ユーザーID,作成日時,経度,緯度,記録日時,高度

        trip_data["出発時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["出発時刻"].values)
        trip_data["到着時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["到着時刻"].values)
        loc_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, loc_time_format))(loc_data["記録日時"].values)

        loc_data[["x", "y"]] = np.array([coord.to_utm(loc_data.loc[i, "lat"], loc_data.loc[i, "lon"]) for i in loc_data.index])
        node_pints = shapely.geometry.MultiPoint([(node.x, node.y) for node in nw_data.nodes.values()])
        convex_hull = node_pints.convex_hull
        loc_data = loc_data.loc[[convex_hull.contains(shapely.geometry.Point(x)) for x in loc_data[["x", "y"]].values], :]

        user_loc_data = {uid: Trip(uid, 0) for uid in loc_data["ユーザーID"].unique()}
        for uid in user_loc_data.keys():
            target = loc_data.loc[loc_data["ユーザーID"] == uid, :]
            user_loc_data[uid].set_points(target["記録日時"].values, target[["x", "y"]].values)

        tree = KDTree(np.array([(node.x, node.y) for node in nw_data.nodes.values()]), leaf_size=2)

        def forward(path, i, link_dist_inv, candidates, near_link, gps_points):
            if i == 0 and len(candidates) == 0:
                return path
            elif len(candidates) == 0:
                down_set = set(path[:-1].end.downstream) - set(path[-1])
                candidates = (link_dist_inv[i-1, :] > 0).toarray().tolist()[0]
                candidates[near_link.index(path[-1])] = False
                link_dist_inv[i-1, :] = 0
                return forward(path[:-1], i-1, link_dist_inv, candidates)
            for j in range(len(candidates)):
                if candidates[j]:
                    dist = heron_vertex((gps_points[i, 0], gps_points[i, 1]),
                                        (near_link[j].start.x, near_link[j].start.y),
                                        (near_link[j].end.x, near_link[j].end.y)) / near_link[j].length *2
                    link_dist_inv[i, j] = 1 / dist
            next_link = near_link[np.argmax(link_dist_inv[i, :])]
            if len(path) == 0 or next_link != path[-1]:
                path.append(near_link[np.argmax(link_dist_inv[i, :])])
            if i == len(gps_points) - 1:
                return path
            down_set = set(path[:-1].end.downstream)
            candidates = [nl in down_set for nl in near_link]
            return forward(path, i+1, link_dist_inv, candidates)

        result = []  # [seq, k, a, b, tid] k, a, b: link_id
        seq = 1
        for idx in trip_data.index:
            tid = trip_data.loc[idx, "ID"]
            uid = trip_data.loc[idx, "ユーザーID"]
            dep_time = trip_data.loc[idx, "出発時刻"]
            end_time = trip_data.loc[idx, "到着時刻"]
            target = user_loc_data[uid].clip(dep_time, end_time)
            if len(target) == 0:
                continue
            trip_list = target.split_by_thresh(thresh)
            for trip in trip_list:
                near_nodes = tree.query(trip_list.gps_points, k=3, return_distance=False).flatten().unique()
                near_link = set(sum([nw_data.nodes[node_id].downstream for node_id in near_nodes], []))

                path = []
                link_dist_inv = csr_matrix((len(trip.gps_points), len(near_link)))
                path = forward(path, 0, link_dist_inv, [nl in near_link for nl in near_link], near_link, trip.gps_points)
                path = [link.id for link in path]

                if len(path) > 1:
                    tmp_result = [[seq, path[i], path[i+1], path[-1].id, tid] for i in range(len(path)-1)]
                    result.extend(tmp_result)
                    seq += 1
        return pd.DataFrame(result, columns=["ID", "k", "a", "b", "tid"])  # original trip id is tid


class Trip:
    def __init__(self, uid, tid):
        self.uid = uid  # user id
        self.tid = tid  # trip id

        self.gps_times = None # [time]
        self.gps_points = None  # [(lon, lat)] or [(x, y)]

        self.dep_time = None
        self.end_time = None

    def __len__(self):
        return len(self.gps_times)

    def set_points(self, times, points):
        self.gps_times = np.array(times)
        self.gps_points = np.array(points)

        idxs = np.argsort(self.gps_times)
        self.gps_times = self.gps_times[idxs]
        self.gps_points = self.gps_points[idxs]

        self.dep_time = self.gps_times[0]
        self.end_time = self.gps_times[-1]

    def clip(self, start_time=None, end_time=None):
        if start_time is None:
            start_time = self.dep_time
        if end_time is None:
            end_time = self.end_time
        idxs = np.where((self.gps_times >= start_time) & (self.gps_times <= end_time))[0]
        new_trip = Trip(self.uid, self.tid)
        new_trip.set_points(self.gps_times[idxs], self.gps_points[idxs])
        return new_trip
    def split(self, time):
        if time < self.dep_time or time > self.end_time:
            raise ValueError("time must be between dep_time and end_time.")
        idx = np.where(self.gps_times <= time)[0][-1]
        trip1 = Trip(self.uid, self.tid)
        trip2 = Trip(self.uid, self.tid)
        trip1.set_points(self.gps_times[:idx], self.gps_points[:idx])
        trip2.set_points(self.gps_times[idx:], self.gps_points[idx:])
        return trip1, trip2

    def split_by_thresh(self, thresh):
        # thresh: sec to regard as the separated trip
        idxs = np.where(np.diff(self.gps_times) > datetime.timedelta(seconds=thresh))[0]
        trips = []
        if len(idxs) == 0:
            trips.append(self)
        else:
            idxs = idxs + 1
            idxs = np.append(idxs, len(self.gps_times))
            idxs = np.insert(idxs, 0, 0)
            for i in range(len(idxs) - 1):
                trips.append(self.clip(self.gps_times[idxs[i]], self.gps_times[idxs[i + 1]]))
        return trips


# test
if __name__ == "__main__":
    import json
    import configparser
    from preprocessing.network_processing import *

    CONFIG = "/Users/dogawa/PycharmProjects/GANs/config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    #pp_path = json.loads(read_data["pp_path"])
    pp_path = ["/Users/dogawa/Desktop/bus/mapmatching/output/matsuyama_proc/walk/result_free.csv"]

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    pp_data = [PP(path, nw_data) for path in pp_path]
    pp_train_test = [tmp_pp_data.split_into([0.8, 0.2]) for tmp_pp_data in pp_data]
    print(len(pp_data[0]), len(pp_train_test[0][0]), len(pp_train_test[0][1]))

    print(pp_data[0].path_dict[1])
    print(pp_data[0].tids[1])
    pp_data[0].write_geo_file("/Users/dogawa/PycharmProjects/GANs/data/test/pp_ped.geojson", driver="GeoJSON")

