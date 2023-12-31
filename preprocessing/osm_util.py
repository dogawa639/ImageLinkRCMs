import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
import shapely.geometry

__all__ = ["NX2GDF"]


class NX2GDF:
    def __init__(self, G, utm_num, tolerance=5.0):
        # G: networkx multi di graph
        # nodes: geopandas dataframe geometry:shapely.geometry.point.Point
        # edges: geopandas dataframe geometry:shapely.geometry.linestring.LineString
        self.G = G
        self.tolerance = tolerance
        self.utm_num = utm_num
        self.epsg = 2442 + utm_num
        self.nodes, self.edges = ox.graph_to_gdfs(G)
        self.edges = self.edges.reset_index()

        # simplify geometry
        self.edges.set_crs(epsg=4326, inplace=True)
        self.edges.to_crs(epsg=self.epsg, inplace=True)
        self.edges["geometry"] = self.edges["geometry"].simplify(tolerance=tolerance)
        self.edges.to_crs(epsg=4326, inplace=True)

        self._add_edge_info()
        self._divide_MultiLineString()
        self._divide_LineString()
        self._create_bidirection()

        self.nodes.set_crs(epsg=4326, inplace=True)
        self.edges.set_crs(epsg=4326, inplace=True)
        self.no_exist_edges.set_crs(epsg=4326, inplace=True)


    def save(self, node_file, edge_file, no_exist_edge_file=None):
        self.nodes.to_file(node_file)
        self.edges.to_file(edge_file)
        if no_exist_edge_file is not None:
            self.no_exist_edges.to_file(no_exist_edge_file)

    def to_epsg(self, epsg):
        return self.nodes.to_crs(epsg=epsg, inplace=False), self.edges.to_crs(epsg=epsg, inplace=False), self.no_exist_edges.to_crs(epsg=epsg, inplace=False)

    def _add_edge_info(self):
        # lanes, highway
        if "lanes" not in self.edges.columns:
            self.edges["lanes"] = 1
        else:
            self.edges["lanes"] = self.edges["lanes"].apply(NX2GDF.get_lane)
        self.edges["maxspeed"] = self.edges["highway"].apply(NX2GDF.get_maxspeed)

    def _divide_MultiLineString(self):
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.values
        geoms = self.edges["geometry"].values
        append_val = None
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            if type(tmp_geom) == shapely.geometry.MultiLineString:
                val[i, geom_idx] = tmp_geom.geoms[0]
                if len(tmp_geom.geoms) == 1:
                    continue
                tmp_val = np.repeat(val[[i], :], len(tmp_geom.geoms) - 1, axis=0)
                for j in range(1, len(tmp_geom.geoms)):
                    tmp_val[j-1, geom_idx] = tmp_geom.geoms[j]
                if append_val is None:
                    append_val = tmp_val
                else:
                    append_val = np.concatenate((append_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges = gpd.GeoDataFrame(val, columns=self.edges.columns)

    def _divide_LineString(self):
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.values
        geoms = self.edges["geometry"].values
        append_val = None
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            geom_list = [shapely.geometry.LineString(tmp_geom.coords[j:j+2]) for j in range(len(tmp_geom.coords)-1)]
            val[i, geom_idx] = geom_list[0]
            if len(geom_list) == 1:
                continue
            tmp_val = np.repeat(val[[i], :], len(geom_list)-1, axis=0)
            for j in range(1, len(geom_list)):
                tmp_val[j-1, geom_idx] = geom_list[j]
            if append_val is None:
                append_val = tmp_val
            else:
                append_val = np.concatenate((append_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges =  gpd.GeoDataFrame(val, columns=self.edges.columns)

    def _create_bidirection(self):
        cols = self.edges.columns.tolist()
        geom_idx = cols.index("geometry")
        cols.remove("geometry")
        val = self.edges.values
        geoms = self.edges["geometry"].values
        reversed = self.edges["reversed"].values
        motorway = self.edges["highway"].values == "motorway"
        oneway = self.edges["oneway"].values
        append_val = None
        no_exist_val = None  # opposite direction of oneway
        for i in range(len(self.edges)):
            tmp_geom = geoms[i]
            geoms[i] = shapely.geometry.LineString(tmp_geom.coords[::-1])  # reversed geometry
            if reversed[i]:
                val[i, geom_idx] = geoms[i]
            if (not oneway[i]) and (not motorway[i]):
                tmp_val = val[[i], :]
                tmp_val[0, geom_idx] = geoms[i]
                if append_val is None:
                    append_val = tmp_val
                else:
                    append_val = np.concatenate((append_val, tmp_val), axis=0)
            else:
                tmp_val = val[[i], :]
                tmp_val[0, geom_idx] = geoms[i]
                if no_exist_val is None:
                    no_exist_val = tmp_val
                else:
                    no_exist_val = np.concatenate((no_exist_val, tmp_val), axis=0)
        if append_val is not None:
            val = np.concatenate((val, append_val), axis=0)
        self.edges = gpd.GeoDataFrame(val, columns=self.edges.columns)
        self.no_exist_edges = gpd.GeoDataFrame(no_exist_val, columns=self.edges.columns)
        # remove duplicated
        self.edges = self.edges[~self.edges["geometry"].duplicated()]
        all_edges = pd.concat([self.edges["geometry"], self.no_exist_edges["geometry"]], axis=0)
        idx = ~all_edges.duplicated().values[len(self.edges):]
        self.no_exist_edges = self.no_exist_edges[idx]

    @staticmethod
    def get_lane(lane):
        try:
            return int(lane)
        except:
            return 1

    @staticmethod
    def get_maxspeed(highway):
        highway_dict = {
            "motorway": 80,
            "motorway_link": 40,
            "trunk": 50,
            "trunk_link": 40,
            "primary": 50,
            "primary_link": 40,
            "secondary": 50,
            "secondary_link": 40,
            "tertiary": 40,
            "tertiary_link": 40,
            "unclassified": 40,
            "road": 30,
            "residential": 30,
            "living_street": 20,
            "service": 10
        }
        if type(highway) == str:
            if highway in highway_dict:
                return highway_dict[highway]
            else:
                print(highway)
                return 20
        else:
            speed = [NX2GDF.get_maxspeed(h) for h in highway]
            return np.mean(speed)
