import osmnx as ox
import geopandas as gpd

class NX2GDF:
    def __init__(self, G):
        # G: networkx multi di graph
        # nodes: geopandas dataframe geometry:shapely.geometry.point.Point
        # edges: geopandas dataframe geometry:shapely.geometry.linestring.LineString
        self.G = G
        self.nodes, self.edges = ox.graph_to_gdfs(G)

    def save(self, node_file, edge_file):
        self.nodes.to_file(node_file)
        self.edges.to_file(edge_file)
    
    def load(self, node_file, edge_file):
        self.nodes = gpd.read_file(node_file)
        self.edges = gpd.read_file(edge_file)
        self.G = ox.graph_from_gdfs(self.nodes, self.edges)

    def _add_geometry_to_edge(self):
        self.edges["geometry"] = self.edges.apply(lambda x: self.G.edges[x["u"], x["v"], x["key"]]["geometry"], axis=1)