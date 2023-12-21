# python 3.7 or later required

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapely
import osmnx as ox

import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import csgraph

from sklearn.preprocessing import StandardScaler

from utility import load_json, dump_json, Coord
from preprocessing.osm_util import NX2GDF

__all__ = ["Node", "Edge", "NetworkBase", "NetworkCNN", "NetworkGNN"]


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
        self.upstream = []

        self.prop = dict()

    def add_downstream(self, downstream_edge):
        if downstream_edge not in self.downstream:
            self.downstream.append(downstream_edge)

    def add_upstream(self, upstream_edge):
        if upstream_edge not in self.upstream:
            self.upstream.append(upstream_edge)

    def set_prop(self, prop):
        for k, v in prop.items():
            if k in self.prop:
                self.prop[k] = v

    def clear_connectivity(self):
        self.downstream = []
        self.upstream = []

    @property
    def prop_values(self):
        return list(self.prop.values())


class Edge:
    base_prop = ["length", "angle"]

    def __init__(self, id, start, end, car=True, ped=True):
        self.id = id
        self.start = start
        self.end = end
        self.length = self.get_length(start, end)
        self.angle = self.get_angle(start, end)  # 0-360

        self.car = car  # whether car can pass
        self.ped = ped  # whether pedestrian can pass
        if self.length == 0:
            self.e = (0, 0)
        else:
            self.e = ((end.x - start.x) / self.length, (end.y - start.y) / self.length)
        self.n = (-self.e[1], self.e[0])
        self.undir_id = -1  # undirected link id

        self.prop = {"length": self.length, "angle": self.angle}

    def set_prop(self, prop):
        for k, v in prop.items():
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

    def clear_undir_id(self):
        self.undir_id = -1

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

    @property
    def center_lonlat(self):
        return Node.coord.from_utm(*self.center)


# Define a class for networks
# abstract class
class NetworkBase:
    def __init__(self, node=None, link=None, node_prop_path=None, link_prop_path=None, utm_num=4, mode="ped", nw_data=None):
        # mode: "ped" or "car"
        if nw_data is None and (node is None or link is None):
            raise Exception("nw_data or node and link should be set.")
        if mode not in ["ped", "car"]:
            raise Exception("mode should be ped or car.")
        if nw_data is not None:
            self.node = nw_data.node
            self.link = nw_data.link
            utm_num = nw_data.utm_num

        self.utm_num = utm_num
        Node.utm_num = utm_num
        Node.coord = Coord(utm_num)

        if type(node) == str:
            self.node = pd.read_csv(node)  # columns=['id', 'lon', 'lat']，行番号がnodeのインデックス
        if type(link) == str:
            self.link = pd.read_csv(link)  # columns=['id', 'start', 'end'(, 'car', 'ped')]，行番号がlinkのインデックス

        self._trim_duplicate()  # remove duplicated link
        self._trim_nodes()  # remove nodes not used

        self.nodes, self.nids = self._set_nodes()  # dict key: id, value: Node
        self.edges_all, self.lids_all = self._set_edges()  # dict key: id, value: Edge

        self._set_edge_nodes()  # (start, end): linkId
        self._set_undir_id()  # set undir_id for all edges

        self.set_mode(mode)  # self.edges, self.lids are set.

        self.node_props = []
        self.link_props = Edge.base_prop

        self.sc_node = StandardScaler()
        self.sc_link = StandardScaler()

        if node_prop_path is not None:
            if type(node_prop_path) == str:
                self.set_node_prop(node_prop_path=node_prop_path)
            else:
                self.set_node_prop(node_prop=node_prop_path)
        if link_prop_path is not None:
            if type(link_prop_path) == str:
                self.set_link_prop(link_prop_path=link_prop_path)
            else:
                self.set_link_prop(link_prop=link_prop_path)

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
        link_props_remove = []  # remove props that cannot be converted to float
        for prop in link_props:
            if prop in Edge.base_prop:
                link_props_remove.append(prop)
            else:
                try:
                    link_prop[prop] = link_prop[prop].astype(np.float32)
                except:
                    link_props_remove.append(prop)
        link_props = [prop for prop in link_props if prop not in link_props_remove]

        self.link_props = self.link_props + link_props

        for i, lid in enumerate(link_prop["LinkID"].values):
            if lid in self.edges_all:
                tmp_prop = dict()
                for prop in link_props:
                    tmp_prop[prop] = link_prop[prop].values[i]
                self.edges_all[lid].set_prop(tmp_prop)

    def set_mode(self, mode):
        self.mode = mode

        self.edges = {lid: edge for lid, edge in self.edges_all.items() if (mode == "ped" and edge.ped) or (mode == "car" and edge.car)}
        self.lids = list(self.edges.keys())

        self._clear_connectivity()
        self._set_downstream()  # add downstream edges to each node
        self._set_upstream()  # add upstream edges to each node

        self._set_graphs()  # adj cost matrix [start,end], adj cost matrix 線グラフ
        self._set_shortest_path_all()

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

    def get_path_prop(self, path):
        props = np.zeros(len(self.link_props) - len(Edge.base_prop), dtype=np.float32)
        cnt = 0
        for prop in self.link_props:
            if prop in Edge.base_prop:
                continue
            for lid in path:
                props[cnt] += self.edges[lid].prop[prop]
            cnt += 1

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

    def plot_link(self, link_ids, colors=None, alphas=None, title=None, ax=None, xlim=None, ylim=None, cmap="jet", show_colorbar=True):
        width, headwidth, headlength = 0.002, 2, 2.5
        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(dpi=200)
            show_fig = True
        ax.set_aspect('equal')
        if colors is None:
            colors = ["black" for _ in link_ids]
        else:
            cm = plt.get_cmap(cmap)
            v_max, v_min = np.max(colors), np.min(colors)
            colors = [cm((color - v_min) / (v_max - v_min)) for color in colors]
            if show_colorbar:
                norm = plt.Normalize(vmin=v_min, vmax=v_max)
                sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
                plt.colorbar(sm, ax=ax)
        if alphas is None:
            alphas = [1.0 for _ in link_ids]

        xs = []
        ys = []
        us = []
        vs = []
        for i, lid in enumerate(link_ids):
            edge = self.edges[lid]
            dx, dy = 0, 0
            if len(self.undir_edges[edge.undir_id]) > 1:  # inv link exists
                dx, dy = edge.n
                dx *= 2
                dy *= 2
            xs.append(edge.start.x + dx)
            ys.append(edge.start.y + dy)
            us.append(edge.end.x - edge.start.x)
            vs.append(edge.end.y - edge.start.y)
        ax.quiver(xs, ys, us, vs, color=colors, alpha=alphas, width=width, headwidth=headwidth, headlength=headlength, angles="xy", scale_units="xy", scale=1)
        if title is not None:
            ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if show_fig:
            plt.show()
            return
        return ax

    def plot_paths(self, paths, colors=None, **kwargs):
        # paths: {d_link_idx: [[path_link_idx]]}
        # colors: {d_link_idx: color}
        # kwargs: ax, title, xlim, ylim, cmap
        if colors is None:
            colors = {d_idx: np.random.random() for d_idx in paths.keys()}
        link_ids = self.lids
        colors_total = [0.0 for _ in link_ids]
        alphas_total = [0.5 for _ in link_ids]
        d_node_x = []
        d_node_y = []
        d_node_c = []
        for d_idx in paths.keys():
            for path in paths[d_idx]:
                for idx in path:
                    colors_total[idx] = colors[d_idx]
                    alphas_total[idx] = 1.0
                    d_node = self.edges[self.lids[d_idx]].end
                    d_node_x.append(d_node.x)
                    d_node_y.append(d_node.y)
                    d_node_c.append(colors[d_idx])

        return_ax = True
        kwargs["show_colorbar"] = False
        if "ax" not in kwargs:
            fig, ax = plt.subplots(dpi=200)
            kwargs["ax"] = ax
            return_ax = False
        if "cmap" not in kwargs:
            kwargs["cmap"] = "jet"

        ax = self.plot_link(link_ids, colors_total, alphas_total, **kwargs)
        cm = plt.get_cmap(kwargs["cmap"])
        v_max, v_min = np.max(colors_total), np.min(colors_total)
        d_node_c = [cm((c - v_min) / (v_max - v_min)) for c in d_node_c]
        ax.scatter(d_node_x, d_node_y, s=2, c=d_node_c)
        if return_ax:
            return ax
        plt.show()
        return

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
        car = self.link["car"].values if "car" in self.link.columns else [True] * len(self.link)
        ped = self.link["ped"].values if "ped" in self.link.columns else [True] * len(self.link)
        for i in range(len(values)):
            start = self.nodes[values[i, 1]]
            end = self.nodes[values[i, 2]]
            edges[values[i, 0]] = Edge(values[i, 0], start, end, car=car[i], ped=ped[i])
        lids = self.link["id"].values
        return edges, lids

    def _clear_connectivity(self):
        for node in self.nodes.values():
            node.clear_connectivity()

    def _set_downstream(self):
        for edge in self.edges.values():
            self.nodes[edge.start.id].add_downstream(edge)

    def _set_upstream(self):
        for edge in self.edges.values():
            self.nodes[edge.end.id].add_upstream(edge)

    def _set_edge_nodes(self):
        self.edge_nodes = {(edge.start.id, edge.end.id): edge.id for edge in self.edges_all.values()}

    def _set_undir_id(self):
        self.undir_edges = dict()  # key: undir_id, value: [lid]
        undir_id = 1
        for k, v in self.edge_nodes.items():
            edge = self.edges_all[v]
            if (k[1], k[0]) in self.edge_nodes:
                inv_edge = self.edges_all[self.edge_nodes[(k[1], k[0])]]
                if inv_edge.undir_id > 0:  # 逆方向リンクがすでに追加されていた場合
                    edge.undir_id = inv_edge.undir_id
                    self.undir_edges[edge.undir_id].append(v)
                    continue
            edge.undir_id = undir_id
            self.undir_edges[undir_id] = [v]
            undir_id += 1

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

    @staticmethod
    def crop(nw_data, polygon_coord, utm_num):
        # polygon_coord: [(lon, lat)]
        # utm_num: 平面直交座標系番号
        # return: cropped network
        coord = Coord(utm_num)
        polygon_coord_xy = [coord.to_utm(lon, lat) for lon, lat in polygon_coord]
        polygon = shapely.geometry.Polygon(polygon_coord_xy)
        # link: [id, start, end]
        # node: [id, lon, lat]
        # link_prop: [LinkID, prop1, prop2, ...]
        nodes = [node.id for node in nw_data.nodes.values() if polygon.contains(shapely.geometry.Point(node.x, node.y))]
        link = [edge for edge in nw_data.edges.values() if edge.start.id in nodes and edge.end.id in nodes]
        link_prop = [[edge.id] + [edge.prop[k] if k in edge.prop else 0.0 for k in nw_data.link_props if k not in Edge.base_prop] for edge in link]

        node = pd.DataFrame([[node.id, node.lon, node.lat] for node in nw_data.nodes.values() if node.id in nodes], columns=["id", "lon", "lat"])
        link = pd.DataFrame([[edge.id, edge.start.id, edge.end.id] for edge in link], columns=["id", "start", "end"])
        link_prop = pd.DataFrame(link_prop, columns=["LinkID"] + [k for k in nw_data.link_props if k not in Edge.base_prop])

        new_nw_data = NetworkBase(node, link, link_prop_path=link_prop)
        return new_nw_data
    
    @staticmethod
    def get_from_osm(polygon_coord, utm_num, node_path, link_path, link_prop_path=None):
        # osmnx
        # poligon_coord: [(lon, lat)]
        # utm_num: 平面直交座標系番号
        # node_path: nodeのcsvファイルのパス [id, lon, lat]
        # link_path: linkのcsvファイルのパス [id, start, end(, car, ped, start_lon, start_lat, end_lon, end_lat)]
        # return: node, link
        epsg = 2442 + utm_num
        bounding_box = shapely.geometry.Polygon(polygon_coord)
        filter = (
            f'["highway"]["area"!~"yes"]'
            f'["highway"!~"abandoned|bus_guideway|construction|cycleway|elevator|'
            f'escalator|footway|no|planned|platform|proposed|raceway|razed|service|track"]'
            f'["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
        )
        net = ox.graph.graph_from_polygon(bounding_box, simplify=True, retain_all=False, custom_filter=filter)

        gdfs = NX2GDF(net, utm_num, tolerance=5.0)
        _, edge_gdf, no_exist_gdf = gdfs.to_epsg(epsg)
        # car availability
        ped_highway = {"bridleway", "corridor", "path", "pedestrian", "steps"}  # pedestrian only
        edge_gdf["car"] = True
        edge_gdf.loc[edge_gdf["highway"].isin(ped_highway), "car"] = False
        no_exist_gdf["car"] = False
        # ped availability
        edge_gdf["ped"] = edge_gdf["highway"] != "motorway"
        no_exist_gdf["ped"] = no_exist_gdf["highway"] != "motorway"
        no_exist_gdf = no_exist_gdf[no_exist_gdf["ped"]]
        edge_gdf["ped"] = True

        edge_gdf = pd.concat([edge_gdf, no_exist_gdf], axis=0)
        edge_gdf = edge_gdf[~edge_gdf["geometry"].duplicated()]
        edge_gdf["geometry"] = edge_gdf["geometry"].to_crs(epsg=4326)
        edge_gdf["id"] = np.arange(len(edge_gdf)) + 1

        node_set = set(sum([[*geom.coords] for geom in edge_gdf["geometry"]], []))
        nid2coord = {i+1: node for i, node in enumerate(node_set)}
        coord2nid = {v: k for k, v in nid2coord.items()}
        edge_gdf["start"] = edge_gdf["geometry"].apply(lambda x: coord2nid[x.coords[0]])
        edge_gdf["end"] = edge_gdf["geometry"].apply(lambda x: coord2nid[x.coords[-1]])

        node_df = pd.DataFrame({"id": list(nid2coord.keys()),
                                "lon": [coord[0] for coord in nid2coord.values()],
                                "lat": [coord[1] for coord in nid2coord.values()]})
        node_df["id"] = node_df["id"].astype(int)

        link_df = edge_gdf[["id", "start", "end", "car", "ped"]]
        link_df.loc[:, ["id", "start", "end"]] = link_df.loc[:, ["id", "start", "end"]].astype(int)

        # coord
        link_df.loc[:, "start_lon"] = link_df["start"].apply(lambda x: nid2coord[x][0])
        link_df.loc[:, "start_lat"] = link_df["start"].apply(lambda x: nid2coord[x][1])
        link_df.loc[:, "end_lon"] = link_df["end"].apply(lambda x: nid2coord[x][0])
        link_df.loc[:, "end_lat"] = link_df["end"].apply(lambda x: nid2coord[x][1])

        node_df.to_csv(node_path, index=False)
        link_df.to_csv(link_path, index=False)

        if link_prop_path is not None:
            cols = [col for col in edge_gdf.columns if not col in ["id", "start", "end", "u", "v",  "key", "osmid", "geometry"]]
            edge_prop = edge_gdf[["id"] + cols]
            edge_prop.columns = ["LinkID"] + cols

            edge_prop.to_csv(link_prop_path, index=False)

    @property
    def feature_num(self):
        return len(self.link_props)

    @property
    def link_num(self):
        return len(self.lids)


class NetworkCNN(NetworkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_action_edges()  # add action edges to each edge

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
            feature_mat = np.concatenate((feature_mat, np.zeros((9, self.context_feature_num), dtype=np.float32)), axis=1)
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
                path, shortest_distance = self.get_shortest_path(edge_down.end.id, d_node_id)
                path_prop = self.get_path_prop(path)
                if shortest_distance is None:
                    shortest_distance = 2000
                d_angle = 0
                if edge_down.end.id != d_node_id:
                    d_angle = self.get_angle(edge_down.end, self.nodes[d_node_id])
                    d_angle = np.abs((d_angle - edge_down.angle) % 360 - 180)
                context_mat[i, 0] += shortest_distance
                context_mat[i, 1] += d_angle
                context_mat[i, 2:] += path_prop
            context_mat[i, :] /= len(edge_downs) + (len(edge_downs) == 0)

        if normalize:
            context_mat = np.concatenate((np.zeros((9, self.feature_num), dtype=np.float32), context_mat), axis=1)
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
        return 2 + len(self.link_props) - len(Edge.base_prop)


class NetworkGNN(NetworkBase):
    # basically using the edge graph
    def __init__(self, k=50, *args, **kwargs):
        # k eig vectors to be used for pos encoding
        super().__init__(*args, **kwargs)
        self.k = k
        self._set_laplacian_matrix()

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
        # 位置エンコーディング用 (線グラフ)
        self.d_matrix = csr_matrix((len(self.edges), len(self.edges)), dtype=np.float32)  # 次数行列
        self.a_matrix = csr_matrix((len(self.edges), len(self.edges)), dtype=np.float32)  # 隣接行列
        lid2idx = {lid: i for i, lid in enumerate(self.lids)}
        for i, edge in enumerate(self.edges.values()):
            for edge_down in edge.end.downstream:
                self.a_matrix[i, lid2idx[edge_down.id]] = 1.0
            self.d_matrix[i, i] = len(edge.end.downstream)
        self.laplacian_matrix = self.d_matrix - self.a_matrix

        d_diag = self.d_matrix.diagonal()
        d_diag[d_diag != 0] = np.power(d_diag[d_diag != 0], -0.5)
        d_inv_half = coo_matrix((d_diag, (range(len(self.edges)), range(len(self.edges)))), shape=(len(self.edges), len(self.edges))).tocsr()
        self.norm_laplacian_matrix = sparse.identity(len(self.edges), dtype=np.float32, format="csr") - d_inv_half * self.a_matrix * d_inv_half  # I - D^-0.5 * A * D^-0.5
        norm_eig_val, self.norm_eig_vec = sparse.linalg.eigs(self.norm_laplacian_matrix, k=self.k, which="SR", tol=0)
        self.norm_eig_vec = self.norm_eig_vec[:, norm_eig_val.argsort()]  # ndarray (link_num, k)

        d_plus = sparse.identity(len(self.edges), dtype=np.float32, format="csr") + self.d_matrix
        a_plus = sparse.identity(len(self.edges), dtype=np.float32, format="csr") + self.a_matrix

        d_plus_diag = np.power(d_plus.diagonal(), -0.5)
        d_plus_inv_half = coo_matrix((d_plus_diag, (range(len(self.edges)), range(len(self.edges)))), shape=(len(self.edges), len(self.edges))).tocsr()  # (I+D)^-0.5
        self.renorm_laplacian_matrix = d_plus_inv_half * a_plus * d_plus_inv_half  # (I+D)^-0.5 * (I+A) * (I+D)^-0.5
        renorm_eig_val, self.renorm_eig_vec = sparse.linalg.eigs(self.renorm_laplacian_matrix, k=self.k, which="SR", tol=0)
        self.renorm_eig_vec = self.renorm_eig_vec[:, renorm_eig_val.argsort()]  # ndarray (link_num, k)

