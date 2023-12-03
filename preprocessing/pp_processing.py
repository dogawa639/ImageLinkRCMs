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

__all__ = ["PP"]


class PP:
    def __init__(self, data, nw_data):
        # all property of network is already set.
        if type(data) == str:
            data = pd.read_csv(data)  # [ID, a, k, b] or [ID, tid, a, k, b], b: last link
        self.data = data  # [ID, a, k, b] or [ID, org, a, k, b], b: last link
        self.nw_data = nw_data

        self.path_dict = PP.get_path_dict(self.data, self.nw_data)  # {"tid": {"path": [link_id], "d_node": node_id, "id": ID(, "org": org)}}
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
            tmp_data = self.data.loc[self.data["ID"].isin(tmp_tids), :]
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
            if "org" in trip:
                result.append([tid, trip["id"], trip["org"], geom])
            else:
                result.append([tid, trip["id"], -1, geom])
        gdf = gpd.GeoDataFrame(result, columns=["tid", "ID", "org", "geometry"])
        gdf.to_file(file_path, driver=driver)

    def plot_path(self, trip_path, loc_path, trip_id=None, org_trip_id=None, trip_time_format="%Y-%m-%d %H:%M:%S", loc_time_format="%Y-%m-%d %H:%M:%S.%f"):
        # trip_id: ID
        # trip_path: path of trip data
        # loc_path: path of location data
        if trip_id is None and org_trip_id is None:
            raise ValueError("trip_id or org_trip_id must be set.")
        if org_trip_id is not None and "org" not in self.data.columns:
            raise ValueError("method requires original trip id in data.")

        fig, ax = plt.subplots(dpi=200)

        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        target = self.data[self.data["ID"] == trip_id] if trip_id is not None else self.data[self.data["org"] == org_trip_id]
        b_links = np.unique(target["b"].values)
        for b in b_links:
            tmp_target = target[target["b"] == b]
            links = np.unique(tmp_target[["k", "a"]].values)
            idxs = [lid2idx[lid] for lid in links]
            path = {idxs[-1]: [idxs]}
            c = np.random.random()
            colors = {idxs[-1]: c}
            self.nw_data.plot_paths(path, ax=ax, colors=colors)

        if org_trip_id is not None:
            coord = Coord(self.nw_data.utm_num)
            trip_data = read_csv_with_encoding(trip_path)
            loc_data = read_csv_with_encoding(loc_path)

            trip_data = trip_data[trip_data["ID"] == org_trip_id]
            uid = trip_data["ユーザーID"].values[0]
            loc_data = loc_data[loc_data["ユーザーID"] == uid]

            trip_data["出発時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(
                trip_data["出発時刻"].values)
            trip_data["到着時刻"] = np.vectorize(lambda x: datetime.datetime.strptime(x, trip_time_format))(
                trip_data["到着時刻"].values)
            loc_data["記録日時"] = np.vectorize(lambda x: datetime.datetime.strptime(x, loc_time_format))(
                loc_data["記録日時"].values)

            loc_data = loc_data[(loc_data["記録日時"] >= trip_data["出発時刻"].values[0]) & (loc_data["記録日時"] <= trip_data["到着時刻"].values[0])]
            loc_data[["x", "y"]] = np.array([coord.to_utm(loc_data.loc[i, "経度"], loc_data.loc[i, "緯度"]) for i in loc_data.index])

            ax.scatter(loc_data["x"], loc_data["y"], s=1, c="r", alpha=0.5)
        plt.show()

    @staticmethod
    def get_path_dict(data, nw_data):
        # real_data: {"tid": {"path": [link_id], "d_node": node_id, "id": ID(, "org": org)}}
        # data is soreted by time
        trip_ids = sorted(data["ID"].unique())
        use_org = "org" in data.columns
        real_data = dict()

        tid = 1
        cnt = 0
        for t in trip_ids:
            target = data[data["ID"] == t]
            val = target[["k", "a"]].values

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
                        real_data[tid] = {"path": path, "d_node": nw_data.edges[last_link].end.id, "id": t}
                        if use_org:
                            real_data[tid]["org"] = target["org"].values[0]
                        tid += 1
                    elif idx == len(tmp_path) - 1:  # if the next link is in the network and this is the last link
                        path.append(tmp_path2[i])
                        last_link = path[-1]
                        real_data[tid] = {"path": path, "d_node": nw_data.edges[last_link].end.id, "id": t}
                        if use_org:
                            real_data[tid]["org"] = target["org"].values[0]
                        tid += 1
                else:  # if the previous link is not in the network
                    path = [tmp_edge]
        print("path_samples: ", tid - 1)
        print("trip_num: ", cnt)
        return real_data

    @staticmethod
    def map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code, out_file=None, thresh=60, dist_thresh=None, loc_time_format="%Y-%m-%d %H:%M:%S.%f", feeder_time_format="%Y-%m-%d %H:%M:%S", trip_time_format="%Y-%m-%d %H:%M:%S"):
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
        node_list = list(nw_data.nodes.values())
        tree = KDTree(np.array([(node.x, node.y) for node in node_list]), leaf_size=2)

        loc_data[["x", "y"]] = np.array([coord.to_utm(loc_data.loc[i, "経度"], loc_data.loc[i, "緯度"]) for i in loc_data.index])
        link_line = shapely.geometry.MultiLineString([[(edge.start.x, edge.start.y), (edge.end.x, edge.end.y)] for edge in nw_data.edges.values()])
        buf_dist = dist_thresh if dist_thresh is not None else 30.0
        buffer = link_line.buffer(buf_dist)
        loc_data = loc_data.loc[[buffer.contains(shapely.geometry.Point(x)) for x in loc_data[["x", "y"]].values], :]

        # recursive function
        def forward(near_link, gps_points, dist_thresh, prev_links, costs, i, candidates):
            # prev_links: [gps_points, link]
            # costs: corresponding to near_link, (i-1)-th gpu point
            # i: index of gps
            # candidates: [bool] candidates for link of i-th gps, each elem corresponds to near_link
            # near_link: [link] candidates in path
            # gps_points: [(x, y)] gps points
            # return: finish, kwargs
            finish = False
            new_costs = np.full(len(near_link), np.inf, dtype=float)
            new_candidates = set()

            for j, is_candidate in enumerate(candidates):
                if is_candidate:
                    p0 = np.array([gps_points[i, 0], gps_points[i, 1]])
                    p1 = np.array([near_link[j].start.x, near_link[j].start.y])
                    p2 = np.array([near_link[j].end.x, near_link[j].end.y])
                    dist = heron_vertex(p0, p1, p2) / near_link[j].length * 2.0
                    l0 = np.sqrt(dist ** 2 + near_link[j].length ** 2)
                    l1 = np.linalg.norm(p1 - p0)
                    l2 = np.linalg.norm(p2 - p0)
                    if l1 > l0:  # p0 is out of the link (end point side)
                        dist = l2
                    elif l2 > l0:  # p0 is out of the link (start point side)
                        dist = l1
                    if i + 1 < len(gps_points):
                        v = np.array([gps_points[i + 1, 0] - gps_points[i, 0], gps_points[i + 1, 1] - gps_points[i, 1]])
                    else:
                        v = np.array([gps_points[i, 0] - gps_points[i - 1, 0], gps_points[i, 1] - gps_points[i - 1, 1]])
                    mul = np.min(np.abs(np.power(np.linalg.norm(v), 2) / np.dot(v, near_link[j].e)), 2.0)
                    dist = dist * mul
                    j_prev = prev_links[i, j]
                    if j_prev < 0:  # no previous link for (i-th gps, link j)
                        new_costs[j] = dist
                    else:
                        new_costs[j] = costs[j_prev] + dist
                    if dist_thresh is not None and dist > dist_thresh:
                        continue

                    down_links = set(near_link[j].end.downstream + [near_link[j]])
                    for j_next, nl in enumerate(near_link):  # candidates for (i+1)-th gps
                        if nl not in down_links:
                            continue
                        if i + 1 >= len(prev_links):
                            continue
                        if prev_links[i + 1, j_next] < 0:  # new candidate for (i+1)-th gps
                            prev_links[i + 1, j_next] = j
                            new_candidates.add(j_next)
                        elif new_costs[prev_links[i + 1, j_next]] > new_costs[j]:  # j is better prev_link for j_next
                            prev_links[i + 1, j_next] = j

            new_candidates = [j_next in new_candidates for j_next in np.arange(len(near_link))]
            if sum(new_candidates) == 0:  # no possible down link
                finish = True
            kwargs = {"prev_links": prev_links, "costs": new_costs, "i": i+1, "candidates": new_candidates}
            return finish, kwargs

        user_loc_data = {uid: Trip(uid, 0) for uid in loc_data["ユーザーID"].unique()}
        for uid in user_loc_data.keys():
            target = loc_data[loc_data["ユーザーID"] == uid]
            user_loc_data[uid].set_points(target["記録日時"].values, target[["x", "y"]].values)

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

                if uid not in user_loc_data:
                    continue
                target = user_loc_data[uid].clip(feeder_dep_time, feeder_end_time)
                if len(target) == 0:
                    continue
                trip_list = target.split_by_thresh(thresh)  # split the feeder trip when the time interval is too long
                for trip in trip_list:
                    if len(trip) < 1:
                        continue
                    near_node_idxs = np.unique(tree.query(trip.gps_points, k=3, return_distance=False))
                    near_link = (sum([node_list[node_idx].downstream for node_idx in near_node_idxs], [])
                                 + sum([node_list[node_idx].upstream for node_idx in near_node_idxs], []))

                    tmp_arg = (near_link, trip.gps_points, dist_thresh)
                    prev_links = np.full((len(trip.gps_points), len(near_link)), -1, dtype=int)
                    costs = np.zeros(len(near_link), dtype=float)
                    tmp_kwargs = {"prev_links": prev_links, "costs": costs, "i": 0, "candidates": [True] * len(near_link)}
                    for _ in range(len(trip.gps_points)):
                        finish, tmp_kwargs = forward(*tmp_arg, **tmp_kwargs)
                        if finish:
                            break
                    path = []
                    for k in range(tmp_kwargs["i"] - 1, -1, -1):
                        if k == tmp_kwargs["i"] - 1:
                            j_tmp = np.argmin(tmp_kwargs["costs"])
                        else:
                            j_tmp = tmp_kwargs["prev_links"][k+1, j_tmp]
                        path.append(near_link[j_tmp].id)
                    path = path[::-1]

                    if len(path) > 1:
                        tmp_result = [[seq, path[i], path[i+1], path[-1], tid] for i in range(len(path)-1) if path[i] != path[i+1]]
                        if len(tmp_result) > 0:
                            result.extend(tmp_result)
                            seq += 1
        df = pd.DataFrame(result, columns=["ID", "k", "a", "b", "org"])  # original trip id is org
        if out_file is not None:
            df.to_csv(out_file, index=False)
        return df


class Trip:
    def __init__(self, uid, tid):
        self.uid = uid  # user id
        self.tid = tid  # trip id

        self.gps_times = []  # [time]
        self.gps_points = []  # [(lon, lat)] or [(x, y)]

        self.dep_time = None
        self.end_time = None

    def __len__(self):
        return len(self.gps_times)

    def set_points(self, times, points):
        if len(times) == 0:
            self.gps_times = []  # [time]
            self.gps_points = []  # [(lon, lat)] or [(x, y)]

            self.dep_time = None
            self.end_time = None
            return
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
        idxs = np.where(np.diff(self.gps_times) > np.timedelta64(thresh, "m"))[0]
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
    import os
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
    pp_data_dir = "/Users/dogawa/Desktop/bus/R4松山ｰ新宿観光PP調査（PP調査データ）/20220907"
    loc_path = os.path.join(pp_data_dir, "t_loc_data.csv")
    feeder_path = os.path.join(pp_data_dir, "t_locfeeder.csv")
    trip_path = os.path.join(pp_data_dir, "t_trip.csv")
    mode_code = 100
    #PP.map_matching(trip_path, feeder_path, loc_path, nw_data, mode_code, out_file="/Users/dogawa/PycharmProjects/GANs/debug/data/pp_ped.csv")
    pp_new = PP("/Users/dogawa/PycharmProjects/GANs/debug/data/pp_ped.csv", nw_data)
    pp_new.plot_path(trip_path, loc_path, org_trip_id=222334)

