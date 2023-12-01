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
    def __init__(self, data, nw_data):
        # all property of network is already set.
        if type(data) == str:
            data = pd.read_csv(data)  # [ID, a, k, b(last link)] or [ID, tid, a, k, b(last link)]
        self.data = data  # [ID, a, k, b] or [ID, tid, a, k, b], b: last link
        self.nw_data = nw_data

        self.path_dict, self.tid2org, self.org2tid = PP.get_path_dict(self.data, self.nw_data)  # {"tid": {"path": [link_id], "d_node": node_id}}
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
            edge_list = [self.nw_data.edges[link_id] for link_id in path]

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
            tmp_pp = PP(tmp_data, self.nw_data)
            result.append(tmp_pp)
        return result

    # visualization
    def write_geo_file(self, file_path, driver=None):
        # [seq, tid, geometry]
        result = []
        for tid, trip in self.path_dict.items():
            geom = [(self.nw_data.edges[link_id].start.lon, self.nw_data.edges[link_id].start.lat) for link_id in trip["path"]]
            geom.append((self.nw_data.edges[trip["path"][-1]].end.lon, self.nw_data.edges[trip["path"][-1]].end.lat))
            geom = shapely.geometry.LineString(geom)
            if "tid" in trip:
                result.append([tid, trip["tid"], geom])
            else:
                result.append([tid, -1, geom])
        gdf = gpd.GeoDataFrame(result, columns=["seq", "ID", "geometry"])
        gdf.to_file(file_path, driver=driver)

    def plot_path(self, trip_id, trip_path, loc_path, trip_time_format="%Y-%m-%d %H:%M:%S", loc_time_format="%Y-%m-%d %H:%M:%S.%f"):
        # trip_id: original id if exists, otherwise tid
        # trip_path: path of trip data
        # loc_path: path of location data
        if "tid" not in self.data.columns:
            raise ValueError("this method requires original trip id in data.")
        coord = Coord(self.nw_data.utm_num)
        trip_data = pd.read_csv(trip_path)
        loc_data = pd.read_csv(loc_path)

        trip_data = trip_data[trip_data["ID"] == trip_id]
        uid = trip_data["ユーザーID"].values[0]
        loc_data = loc_data[loc_data["ユーザーID"] == uid]

        trip_data["出発時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(
            trip_data["出発時刻"].values)
        trip_data["到着時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(
            trip_data["到着時刻"].values)
        loc_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, loc_time_format))(
            loc_data["記録日時"].values)

        loc_data = loc_data[(loc_data["記録日時"] >= trip_data["出発時刻"].values[0]) & (loc_data["記録日時"] <= trip_data["到着時刻"].values[0])]
        loc_data[["x", "y"]] = np.array([coord.to_utm(loc_data.loc[i, "lon"], loc_data.loc[i, "lat"]) for i in loc_data.index])

        fig, ax = plt.subplots(dpi=200)
        tids = self.org2tid[trip_id]
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for tid in tids:
            idxs = [lid2idx[lid] for lid in self.path_dict[tid]["path"]]
            path = {idxs[-1]: idxs}
            c = np.random.random()
            colors = {idxs[-1]: c}
            self.nw_data.plot_paths(path, ax=ax, colors=colors)
        ax.scatter(loc_data["x"], loc_data["y"], s=1, c="r")
        plt.show()

    @staticmethod
    def get_path_dict(data, nw_data):
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

            tmp_path = [(i, v[0]) for i, v in enumerate(val) if v[0] in nw_data.edges]
            tmp_path2 = {i: v[1] for i, v in enumerate(val) if v[1] in nw_data.edges}
            if len(tmp_path) == 0:
                continue
            cnt += 1
            path = []
            for idx, (i, tmp_edge) in enumerate(tmp_path):
                if i - 1 in tmp_path2:  # if the previous link is in the network
                    path.append(tmp_edge)
                    if i not in tmp_path2:  # if the next link is not in the network
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": nw_data.edges[last_link].end.id}
                        if org_tid is not None:
                            real_data[tid]["tid"] = org_tid
                        tid2org[tid] = t
                        org2tid[t].append(tid)
                        tid += 1
                    elif idx == len(tmp_path) - 1:  # if the next link is in the network and this is the last link
                        path.append(tmp_path2[i])
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": nw_data.edges[last_link].end.id}
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
    def map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code, out_file=None, thresh= 60, loc_time_format="%Y-%m-%d %H:%M:%S.%f", feeder_time_format="%Y-%m-%d %H:%M:%S", trip_time_format="%Y-%m-%d %H:%M:%S"):
        # thresh: sec to regard as the separated trip
        coord = Coord(nw_data.utm_num)
        trip_data = read_csv_with_encoding(trip_path)  # ID,ユーザーID,作成日時,出発時刻,到着時刻,更新日時,有効性,目的コード（active）
        feeder_data = read_csv_with_encoding(feeder_path)  # ID,トリップID,ユーザーID,作成日時,操作タイプ(1:出発、5:移動手段変更),更新日時,有効性,移動手段コード,記録日時
        loc_data = read_csv_with_encoding(loc_path)  # ID,accuracy,bearing,speed,ユーザーID,作成日時,経度,緯度,記録日時,高度

        trip_data["出発時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["出発時刻"].values)
        trip_data["到着時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(trip_data["到着時刻"].values)
        feeder_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, feeder_time_format))(feeder_data["記録日時"].values)
        loc_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, loc_time_format))(loc_data["記録日時"].values)

        # loc data clipping by network
        loc_data[["x", "y"]] = np.array([coord.to_utm(loc_data.loc[i, "lon"], loc_data.loc[i, "lat"]) for i in loc_data.index])
        node_pints = shapely.geometry.MultiPoint([(node.x, node.y) for node in nw_data.nodes.values()])
        convex_hull = node_pints.convex_hull
        loc_data = loc_data.loc[[convex_hull.contains(shapely.geometry.Point(x)) for x in loc_data[["x", "y"]].values], :]

        user_loc_data = {uid: Trip(uid, 0) for uid in loc_data["ユーザーID"].unique()}
        for uid in user_loc_data.keys():
            target = loc_data[loc_data["ユーザーID"] == uid]
            user_loc_data[uid].set_points(target["記録日時"].values, target[["x", "y"]].values)

        tree = KDTree(np.array([(node.x, node.y) for node in nw_data.nodes.values()]), leaf_size=2)

        # recursive function
        def forward(path, i, link_dist_inv, candidates, near_link, gps_points):
            # path: [link]
            # i: index of gps
            # link_dist_inv: [gps_point, link]
            # candidates: [link] candidates for link of i-th gps
            # near_link: [link] candidates in path
            # gps_points: [(x, y)] gps points
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
            # feeder 記録日時 is departure time of the unlinked trip
            # trip 到着時刻 is arrival time of the linked trip
            end_time = trip_data.loc[idx, "到着時刻"]
            feeder_tmp = feeder_data[(feeder_data["ユーザーID"] == uid) & (feeder_data["トリップID"] == tid)]
            feeder_tmp = feeder_tmp.sort_values("ID")
            for j, f_idx in enumerate(feeder_tmp.index):
                feeder_dep_time = feeder_tmp.loc[f_idx, "記録日時"]
                if feeder_tmp.loc[f_idx, "移動手段コード"] != mode_code:
                    continue
                if j < len(feeder_tmp) - 1:
                    feeder_end_time = feeder_tmp.loc[feeder_tmp.index[j+1], "記録日時"]
                else:
                    feeder_end_time = end_time

                target = user_loc_data[uid].clip(feeder_dep_time, feeder_end_time)
                if len(target) == 0:
                    continue
                trip_list = target.split_by_thresh(thresh)  # split the feeder trip when the time interval is too long
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
        df = pd.DataFrame(result, columns=["ID", "k", "a", "b", "tid"])  # original trip id is tid
        if out_file is not None:
            df.to_csv(out_file, index=False)
        return df


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
    pp_path = json.loads(read_data["pp_path"])

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    pp_data = PP(pp_path[0], nw_data)
    pp_train_test = pp_data.split_into([0.8, 0.2])
    print(len(pp_data), len(pp_train_test[0]), len(pp_train_test[1]))

    print(pp_data.path_dict[1])
    print(pp_data.tids[1])
    #pp_data.write_geo_file("/Users/dogawa/PycharmProjects/GANs/data/test/pp_ped.geojson", driver="GeoJSON")
    loc_path = "/Users/dogawa/Desktop/bus/R4松山ｰ新宿観光PP調査（PP調査データ）/locationMatsuyama2022_2.csv"
    feeder_path = "/Users/dogawa/Desktop/bus/R4松山ｰ新宿観光PP調査（PP調査データ）/feederMatsuyama2022.csv"
    trip_path = "/Users/dogawa/Desktop/bus/R4松山ｰ新宿観光PP調査（PP調査データ）/tripMatsuyama2022.csv"
    mode_code = 100
    PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code, out_file="/Users/dogawa/PycharmProjects/GANs/debug/data/pp_ped.csv")


