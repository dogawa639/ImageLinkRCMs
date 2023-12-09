import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from PIL import Image


__all__ = ["get_vertex_from_polygon", "tile2latlon", "MapSegmentation"]


def get_vertex_from_polygon(geo_file):
    # geo_file: geojson file
    # return: [(lon, lat)]
    gdf = gpd.read_file(geo_file)
    geom = gdf["geometry"].values[0]
    if type(geom) is not shapely.geometry.polygon.Polygon:
        raise Exception("geom should be Polygon")
    return [(x[0], x[1]) for x in geom.exterior.coords]


def tile2latlon(x, y, z):
    # 座標変換 (タイル座標->タイル北西緯度経度)
    lon = (x / 2 ** z) * 360 - 180
    lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y / 2 ** z))))
    return lon, lat


class MapSegmentation:
    # 地図の色からクラスのone-hot vectorを作成する
    def __init__(self, files, max_class_num=10):
        self.files = files
        self.max_class_num = max_class_num  # containing other class
        self._load_color_dict() # self.color_dict, self.pixel_count  key: (r,g,b)
        self._sort_color()  # self.color_list  sorted bu color pixel count, [0]: majour class
        self.class_num = len(self.color_dict)

    def convert_file(self, file, np_file=None):
        # return: (class_num, H, W) ndarray
        input_image = Image.open(file)
        input_image = input_image.convert("RGB")  # (H, W, 3)
        input_image = np.array(input_image)
        class_num = min(self.class_num, self.max_class_num)
        one_hot = np.zeros((class_num, input_image.shape[0], input_image.shape[1]), dtype=np.uint8)# (C, H, W)
        color_dict_small = {k: v for i, (k, v) in enumerate(self.color_dict.items()) if i < class_num - 1}
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                color = tuple(input_image[i,j,:])
                if color in color_dict_small:
                    one_hot[self.color_dict[color], i, j] = 1.0
                else:  # last element is other class
                    one_hot[-1, i, j] = 1.0
        if np_file is not None:
            np.save(np_file, one_hot)
        return one_hot

    # visualization
    def write_colormap(self, file=None):
        class_num = min(self.class_num, self.max_class_num)

        fig = plt.figure(dpi=15, figsize=(1, class_num))
        for i in range(class_num - 1):
            ax = fig.add_subplot(class_num, 1, i+1)
            ax.imshow(np.array([[self.color_list[i]]], dtype=np.uint8))
            ax.axis("off")
        ax = fig.add_subplot(class_num, 1, class_num)
        ax.imshow(np.zeros((1, 1, 3), dtype=np.uint8))
        ax.axis("off")
        if file is not None:
            plt.savefig(file)
        plt.show()

    def write_hist(self, file=None):
        class_num = min(self.class_num, self.max_class_num)
        fig, ax = plt.subplots()
        ax.bar(range(self.class_num), self.pixel_count.values())
        ax.plot([class_num, class_num], [0, max(self.pixel_count.values())], color="red")
        ax.set_xlabel("class")
        ax.set_ylabel("pixel count")
        ax.set_title("Histogram of pixel count")
        ax.set_yscale("log")
        if file is not None:
            plt.savefig(file)
        plt.show()

    # inside function
    def _load_color_dict(self):
        color_dict = {}  # key: (r,g,b), value: class_num
        pixel_count = {}  # key: (r,g,b), value: num_pixel
        for file in self.files:
            input_image = Image.open(file)
            input_image = input_image.convert("RGB")
            input_image = np.array(input_image)
            for i in range(input_image.shape[0]):
                for j in range(input_image.shape[1]):
                    color = tuple(input_image[i, j, :])
                    if color == (0, 0, 0) or color == (255, 255, 255):  # black and white
                        continue
                    if color not in color_dict:
                        color_dict[color] = len(color_dict)
                        pixel_count[color] = 1
                    else:
                        pixel_count[color] += 1

        self.color_dict = color_dict
        self.pixel_count = pixel_count

    def _sort_color(self):
        pixel_count_sorted = sorted(self.pixel_count.items(), key=lambda x: x[1], reverse=True)
        self.color_list = [x[0] for x in pixel_count_sorted]
        self.color_dict = {k: i for i, k in enumerate(self.color_list)}
        self.pixel_count = {k: self.pixel_count[k] for k in self.color_list}


class GISSegmentation(MapSegmentation):
    # segmentation of GIS data
    def __init__(self, files, prop, utm_num, bbox=None, max_class_num=10):
        # bbox: [min_lon, min_lat, max_lon, max_lat]
        self.files = files
        self.prop = prop
        self.utm_num = utm_num
        self.bbox = bbox
        self.max_class_num = max_class_num  # containing other class

        self.mask = None
        if bbox is not None:
            min_lon, min_lat, max_lon, max_lat = bbox
            polygon = shapely.geometry.Polygon([(min_lon, min_lat), (min_lon, max_lat), (max_lon, max_lat), (max_lon, min_lat)])
            self.mask = gpd.GeoSeries({"geometry": [polygon]}, crs="EPSG:4326")



        super().__init__(files, max_class_num)


    @staticmethod
    def gis2png(file, prop, out_file, mask=None, bbox=None):
        # bbox: [min_lon, min_lat, max_lon, max_lat]
        gdf = gpd.read_file(file, mask=mask, bbox=bbox)

        ## ここから
        gdf = gdf[gdf[prop] == 1]
        gdf = gdf.to_crs("EPSG:3857")
        gdf.plot()
        plt.savefig(out_file)
        plt.close()
