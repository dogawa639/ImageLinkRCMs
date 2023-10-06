# python 3.7 or later required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.sparse import csgraph

from utility import load_json, dump_json, Coord

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
        self._set_action_edges()  # add action edges to each edge
        self._set_edge_nodes()  # (start, end): linkId

        self._set_graphs()  # adj cost matrix [start,end], adj cost matrix 線グラフ
        self._set_shortest_path_all()

        self.node_props = []
        self.link_props = []
        if node_prop_path is not None:
            self.set_node_prop(node_prop_path)
        if link_prop_path is not None:
            self.set_link_prop(link_prop_path)

    def set_node_prop(self, node_prop_path):
        node_prop = pd.read_csv(node_prop_path)  # columns=['NodeID', prop1,...]
        node_props = node_prop.columns.to_list()
        node_props.remove("NodeID")
        self.node_props = self.node_props + node_props

        for i, nid in enumerate(node_prop["NodeID"].values):
            if nid in self.nodes:
                tmp_prop = dict()
                for prop in enumerate(node_props):
                    tmp_prop[prop] = node_prop[prop].values[i]
                self.nodes[nid].set_prop(tmp_prop)

    def set_link_prop(self, link_prop_path):
        link_prop = pd.read_csv(link_prop_path)
        link_props = link_prop.columns.to_list()
        link_props.remove("LinkID")
        self.link_props = self.link_props + link_props

        for i, lid in enumerate(link_prop["LinkID"].values):
            if lid in self.edges:
                tmp_prop = dict()
                for prop in enumerate(link_props):
                    tmp_prop[prop] = link_prop[prop].values[i]
                self.edges[lid].set_prop(tmp_prop)


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

    def _set_action_edges(self):
        for edge in self.edges.values():
            edge.set_action_edges()

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



