# python 3.7 or later required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csgraph

from sklearn.preprocessing import StandardScaler

from utility import load_json, dump_json, Coord

__all__ = ["NetworkCNN", "NetworkGNN"]


# Define a class for nodes
class Node:
    utm_num = 4
    coord = Coord(utm_num)

    def __init__(self, id, lon, lat):
        self.id = id
        self.lon = lon
        self.lat = lat

        self.x, self.y = self.coord.to_utm(lon, lat)
        self.downstream = []

        self.prop = dict()

    def add_downstream(self, downstream_edge):
        if downstream_edge not in self.downstream:
            self.downstream.append(downstream_edge)

    def set_prop(self, prop):
        for k, v in prop.items():
            if k in self.prop:
                self.prop[k] = v

    @property
    def prop_values(self):
        return list(self.prop.values())


class Edge:
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end
        self.length = self.get_length(start, end)
        self.angle = self.get_angle(start, end)

        self.prop = {"length": self.length, "angle": self.angle}

    def set_prop(self, prop):
        for k, v in prop.items():
            if k in self.prop:
                self.prop[k] = v

    def set_action_edges(self):
        # action_edges: [[edge]] 9 elements ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2))
        self.action_edges = [[] for i in range(9)]
        self.action_edges[4].append(self)
        for edge_down in self.end.downstream:
            idx = self.get_action_index(edge_down.angle, self.angle)
            self.action_edges[idx].append(edge_down)

    def get_action_edges_mask(self):
        return np.array([len(edges) > 0 for edges in self.action_edges])
    def get_d_edge_mask(self, d_edge):
        if type(d_edge) != int:  # d_edge: edge obj
            d_edge = d_edge.id
        action_edge_ids = [[edge.id for edge in edges] for edges in self.action_edges]
        mask = np.array([d_edge in edge_ids for edge_ids in action_edge_ids])
        return mask


    @staticmethod
    def get_length(start, end):
        return np.sqrt((start.x - end.x) ** 2 + (start.y - end.y) ** 2)

    @staticmethod
    def get_angle(start, end):
        # angle: 0-360
        dx = end.x - start.x
        dy = end.y - start.y
        if dx == 0:
            if dy > 0:
                return 90.0
            else:
                return 270.0
        arctan = np.arctan(dy / dx) * 180.0 / np.pi
        if dx < 0:
            arctan += 180.0
        return arctan % 360.0

    @staticmethod
    def get_action_index(angle_down, angle_up):
        # return the index in 3*3 feature matrix
        d_theta = (angle_down - angle_up) % 360.0
        d_theta_idx = d_theta // 22.5
        if d_theta_idx == 0 or d_theta_idx == 15:
            return 1
        elif d_theta_idx <= 2:
            return 0
        elif d_theta_idx <= 4:
            return 3
        elif d_theta_idx <= 6:
            return 6
        elif d_theta_idx <= 8:
            return 7
        elif d_theta_idx <= 10:
            return 8
        elif d_theta_idx <= 12:
            return 5
        elif d_theta_idx <= 14:
            return 2

    @property
    def prop_values(self):
        return list(self.prop.values())

    @property
    def center(self):
        return (self.start.x + self.end.x) / 2.0, (self.start.y + self.end.y) / 2.0


# Define a class for networks
# abstract class
class NetworkBase:
    def __init__(self, node_path, link_path, node_prop_path=None, link_prop_path=None):
        self.node = pd.read_csv(node_path)  # columns=['id', 'lon', 'lat']，行番号がnodeのインデックス
        self.link = pd.read_csv(link_path)  # columns=['id', 'start', 'end']，行番号がlinkのインデックス

        self._trim_duplicate()  # remove duplicated link
        self._trim_nodes()  # remove nodes not used

        self.nodes, self.nids = self._set_nodes()  # dict key: id, value: Node
        self.edges, self.lids = self._set_edges()  # dict key: id, value: Edge
        self._set_downstream()  # add downstream edges to each node
        self._set_edge_nodes()  # (start, end): linkId

        self._set_graphs()  # adj cost matrix [start,end], adj cost matrix 線グラフ
        self._set_shortest_path_all()

        self.node_props = []
        self.link_props = []

        self.sc_node = StandardScaler()
        self.sc_link = StandardScaler()

        if node_prop_path is not None:
            self.set_node_prop(node_prop_path=node_prop_path)
        if link_prop_path is not None:
            self.set_link_prop(link_prop_path=link_prop_path)

    def set_node_prop(self, node_prop=None, node_prop_path=None):
        if node_prop_path is not None:
            node_prop = pd.read_csv(node_prop_path)  # columns=['NodeID', prop1,...]
        if node_prop is None:
            print("node_prop is None.")
            return
        node_props = node_prop.columns.to_list()
        node_props.remove("NodeID")
        self.node_props = self.node_props + node_props

        for i, nid in enumerate(node_prop["NodeID"].values):
            if nid in self.nodes:
                tmp_prop = dict()
                for prop in enumerate(node_props):
                    tmp_prop[prop] = node_prop[prop].values[i]
                self.nodes[nid].set_prop(tmp_prop)

    def set_link_prop(self, link_prop=None, link_prop_path=None):
        if link_prop_path is not None:
            link_prop = pd.read_csv(link_prop_path)
        if link_prop is None:
            print("link_prop is None.")
            return
        link_props = link_prop.columns.to_list()
        link_props.remove("LinkID")
        self.link_props = self.link_props + link_props

        for i, lid in enumerate(link_prop["LinkID"].values):
            if lid in self.edges:
                tmp_prop = dict()
                for prop in enumerate(link_props):
                    tmp_prop[prop] = link_prop[prop].values[i]
                self.edges[lid].set_prop(tmp_prop)

    def get_shortest_path(self, start, end):
        # start, end: node id
        nid2idx = {nid: i for i, nid in enumerate(self.nids)}
        s = nid2idx[start]
        e = nid2idx[end]
        if np.isinf(self.dist_matrix[s, e]):
            return None, None
        tmp = e
        path = []
        while tmp != s:
            pre = self.predecessors[s, tmp]
            path.append(self.edge_nodes[(self.nids[pre], self.nids[tmp])])
            tmp = pre
        path.reverse()
        return path, self.dist_matrix[s, e].astype(np.float32)

    def get_shortest_path_edge(self, start, end):
        # start, end: link id
        lid2idx = {lid: i for i, lid in enumerate(self.lids)}
        s = lid2idx[start]
        e = lid2idx[end]
        if np.isinf(self.dist_matrix_edge[s, e]):
            return None, None
        tmp = e
        path = []
        while tmp != s:
            pre = self.predecessors_edge[s, tmp]
            path.append(self.lids[pre])
            tmp = pre
        path.reverse()
        return path, self.dist_matrix_edge[s, e].astype(np.float32)

    def get_sc_params_node(self):
        return self.sc_node.mean_, self.sc_node.scale_
    def get_sc_params_link(self):
        return self.sc_link.mean_, self.sc_link.scale_

    def set_sc_params_node(self, params):
        self.sc_node.mean_ = params[0]
        self.sc_node.scale_ = params[1]

    def set_sc_params_link(self, params):
        self.sc_link.mean_ = params[0]
        self.sc_link.scale_ = params[1]

    # functions for inside processing
    def _trim_duplicate(self):
        dup = self.link.duplicated(subset="id", keep="first")
        self.link = self.link.loc[(~dup), :]

    def _trim_nodes(self):
        edge_node_set = set(np.unique(np.reshape(self.link[["start", "end"]].values, (-1))))
        val = self.node["id"].values
        idx = np.array([v in edge_node_set for v in val])
        self.node = self.node.loc[idx, :]

    def _set_nodes(self):
        nodes = dict()
        values = self.node[['id', 'lon', 'lat']].values
        for i in range(len(self.node)):
            nodes[values[i, 0]] = Node(*values[i])
        nids = self.node["id"].values
        return nodes, nids

    def _set_edges(self):
        edges = dict()
        values = self.link[['id', 'start', 'end']].values
        for i in range(len(self.link)):
            start = self.nodes[values[i, 1]]
            end = self.nodes[values[i, 2]]
            edges[values[i, 0]] = Edge(values[i, 0], start, end)
        lids = self.link["id"].values
        return edges, lids

    def _set_downstream(self):
        for edge in self.edges.values():
            self.nodes[edge.start.id].add_downstream(edge)

    def _set_edge_nodes(self):
        val = self.link[["id", "start", "end"]].values
        self.edge_nodes = {(v[1], v[2]): v[0] for v in val}

    def _set_graphs(self):
        self.graph = np.zeros((len(self.node), len(self.node)), dtype=np.float32)
        nid2idx = {nid: i for i, nid in enumerate(self.nids)}

        for edge in self.edges.values():
            s = nid2idx[edge.start.id]
            e = nid2idx[edge.end.id]
            self.graph[s, e] = edge.length

        self.edge_graph = np.zeros((len(self.lids), len(self.lids)), dtype=np.float32)
        lid2idx = {lid: i for i, lid in enumerate(self.lids)}
        for i, edge in enumerate(self.edges.values()):
            for edge_down in edge.end.downstream:
                self.edge_graph[i, lid2idx[edge_down.id]] = (edge.length + edge_down.length) / 2.0

    def _set_shortest_path_all(self):
        kwrds = {"directed": True, "return_predecessors": True}

        graph = scipy.sparse.csr_matrix(self.graph)
        self.dist_matrix, self.predecessors = csgraph.floyd_warshall(graph, **kwrds)

        edge_graph = scipy.sparse.csr_matrix(self.edge_graph)
        self.dist_matrix_edge, self.predecessors_edge = csgraph.floyd_warshall(edge_graph, **kwrds)

    @staticmethod
    def get_angle(start, end):
        # angle: 0-360
        dx = end.x - start.x
        dy = end.y - start.y
        if dx == 0:
            if dy > 0:
                return 90.0
            else:
                return 270.0
        arctan = np.arctan(dy / dx) * 180.0 / np.pi
        if dx < 0:
            arctan += 180.0
        return arctan % 360

    @property
    def feature_num(self):
        return len(self.link_props)


class NetworkCNN(NetworkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_action_edges()  # add action edges to each edge

        self.sc_link = StandardScaler()

    def get_feature_matrix(self, lid, normalize=False):
        # feature_num*3*3 matrix: center element represents the lid
        # features: length, props
        feature_mat = np.zeros((9, self.feature_num), dtype=np.float32)
        edge_tar = self.edges[lid]
        # fill feature mat
        for i in range(9):
            edge_downs = edge_tar.action_edges[i]
            for edge_down in edge_downs:
                for j, link_prop in enumerate(self.link_props):
                    if link_prop in edge_down.prop:
                        feature_mat[i, j] += edge_down.prop[link_prop]
            feature_mat[i, :] /= len(edge_downs) + (len(edge_downs) == 0)
        if normalize:
            feature_mat = np.concatenate((feature_mat, np.zeros(9, self.context_feature_num)), axis=1)
            feature_mat = self.sc_link.transform(feature_mat)
            feature_mat = feature_mat[:, :self.feature_num]

        feature_mat[np.isnan(feature_mat)] = 0.0
        feature_mat = feature_mat.transpose(0, 1).reshape(self.feature_num, 3, 3)

        # create action_edge
        action_edge = [[edge_down.id for edge_down in edge_tar.action_edges[i]] for i in range(9)]  # [[edge_id]]

        return feature_mat, action_edge

    def get_context_matrix(self, lid, d_node_id, normalize=False):
        # context_feature_num*3*3 matrix: center element represents the lid
        # features: shortest_path_to_destination, angle_from_destination(0-360)
        context_mat = np.zeros((9, self.context_feature_num), dtype=np.float32)
        edge_tar = self.edges[lid]
        # fill context_mat
        for i in range(9):
            edge_downs = edge_tar.action_edges[i]
            for edge_down in edge_downs:
                _, shortest_distance = self.get_shortest_path(edge_down.end.id, d_node_id)
                if shortest_distance is None:
                    shortest_distance = 2000
                d_angle = 0
                if edge_down.end.id != d_node_id:
                    d_angle = self.get_angle(edge_down.end, self.nodes[d_node_id])
                    d_angle = np.abs((d_angle - edge_down.angle) % 360 - 180)
                context_mat[i, 0] += shortest_distance
                context_mat[i, 1] += d_angle
            context_mat[i, :] /= len(edge_downs) + (len(edge_downs) == 0)

        if normalize:
            context_mat = np.concatenate((np.zeros(9, self.feature_num), context_mat), axis=1)
            context_mat = self.sc_link.transform(context_mat)
            context_mat = context_mat[:, self.feature_num:]

        context_mat[np.isnan(context_mat)] = 0.0
        context_mat = context_mat.transpose(0, 1).reshape(self.context_feature_num, 3, 3)

        # create action_edge
        action_edge = [[edge_down.id for edge_down in edge_tar.action_edges[i]] for i in range(9)]  # [[edge_id]]

        return context_mat, action_edge

    def get_all_feature_matrix(self, d_node_id, normalize=False):
        # all_features: [link_num, f+c]
        all_features = np.zeros((len(self.edges), self.feature_num + self.context_feature_num), dtype=np.float32)
        for i, lid in enumerate(self.lids):
            edge = self.edges[lid]
            all_features[i, 0] = edge.length  # length of link
            if not edge.prop is None:
                for j, link_prop in enumerate(self.link_props):
                    if link_prop in edge.prop:
                        all_features[i, 1 + j] = edge.prop[link_prop]
            _, shortest_distance = self.get_shortest_path(edge.end.id, d_node_id)
            if shortest_distance is None:
                shortest_distance = 2000
            all_features[i, self.feature_num] = shortest_distance  # shortest distance to the destination
            d_angle = 0
            if edge.end.id != d_node_id:
                d_angle = self.get_angle(edge.end, self.nodes[d_node_id])
                d_angle = np.abs((d_angle - edge.angle) % 360 - 180)
            all_features[i, self.feature_num + 1] = d_angle  # angle from the destinaiton

        if normalize:
            all_features = self.sc_link.transform(all_features)
        return all_features
    def set_link_prop(self, **kwargs):
        super().set_link_prop(**kwargs)

        features = None
        for i in range(30):
            d_node_id = self.nids[np.random.randint(len(self.nids))]
            tmp_features = self.get_all_feature_matrix(d_node_id)
            tmp_features = tmp_features.astype(np.float32)
            if features is None:
                features = tmp_features
            else:
                features = np.concatenate((features, tmp_features), axis=0)
        self.sc_link.fit(features)  # feature_num + context_num

    # functions for inside processing
    def _set_action_edges(self):
        for edge in self.edges.values():
            edge.set_action_edges()

    @property
    def context_feature_num(self):
        # shortest path length, angle
        return 2


class NetworkGNN(NetworkBase):
    # basically using the edge graph
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_laplacian_matrix()

        self.sc_link = StandardScaler()

    def get_feature_matrix(self, normalize=False):
        # feature of all edges (link_num, feature_num)
        feature_mat = np.zeros((len(self.edges), self.feature_num), dtype=np.float32)
        for i, edge in enumerate(self.edges.values()):
            for j, link_prop in enumerate(self.link_props):
                if link_prop in edge.prop:
                    feature_mat[i, j] = edge.prop[link_prop]

        if normalize:
            feature_mat = self.sc_link.transform(feature_mat)
        return feature_mat

    def set_link_prop(self, **kwargs):
        super().set_link_prop(**kwargs)

        feature_mat = self.get_feature_matrix()
        self.sc_link.fit(feature_mat)

    # functions for inside processing
    def _set_laplacian_matrix(self):
        # 位置エンコーディング用
        self.d_matrix = np.zeros((len(self.edges), len(self.edges)), dtype=np.float32)
        self.a_matrix = np.zeros((len(self.edges), len(self.edges)), dtype=np.float32)
        lid2idx = {lid: i for i, lid in enumerate(self.lids)}
        for i, edge in enumerate(self.edges.values()):
            for edge_down in edge.end.downstream:
                self.a_matrix[i, lid2idx[edge_down.id]] = 1.0
            self.d_matrix[i, i] = len(edge.end.downstream)
        self.laplacian_matrix = self.d_matrix - self.a_matrix

        d_inv_half = np.diag(np.power(np.diag(self.d_matrix + (self.d_matrix == 0)), -0.5))
        self.norm_laplacian_matrix = np.identity(len(self.edges)) - np.matmul(np.matmul(d_inv_half, self.a_matrix),
                                                                              d_inv_half)

        d_plus = np.identity(len(self.edges)) + self.d_matrix
        a_plus = np.identity(len(self.edges)) + self.a_matrix
        d_plus_inv_half = np.diag(np.power(np.diag(d_plus), -0.5))  # (I+D)^-0.5
        self.renorm_laplacian_matrix = np.matmul(np.matmul(d_plus_inv_half, a_plus), d_plus_inv_half)
