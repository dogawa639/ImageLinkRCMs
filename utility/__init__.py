import json
import pickle

import numpy as np

import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from PIL import Image

__all__ = ["load_json", "dump_json", "load_pickle", "dump_pickle", "Coord", "MapSegmentation"]


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def dump_json(data, file):
    with open(file, "e") as f:
        json.dump(data, f, indent=4)


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


class Coord:
    # 座標変換用
    def __init__(self, utm_num):
        # utm_num: 平面直交座標系番号
        self.utm_num = utm_num
        self.wgs84 = pyproj.CRS("epsg:4326")
        self.utm = pyproj.CRS(f"epsg:{utm_num + 2442}")
        self.project_to_utm = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True).transform
        self.project_from_utm = pyproj.Transformer.from_crs(self.utm, self.wgs84, always_xy=True).transform

    def to_utm(self, lon, lat):
        utm_point = transform(self.project_to_utm, Point(lon, lat))
        return (utm_point.x, utm_point.y)

    def from_utm(self, x, y):
        lon, lat = transform(self.project_from_utm, Point(x, y))
        return (lon, lat)
    

class MapSegmentation:
    # 地図の色からクラスのone-hot vectorを作成する
    def __init__(self, files):
        self.files = files
        self.color_dict = self._load_color_dict() # key: (r,g,b), value: class_num
        self.class_num = len(self.color_dict)

        self.color_list = [None] * self.class_num
        for k,v in self.color_dict.items():
            self.color_list[v] = k

    def _load_color_dict(self):
        color_dict = {}
        for file in self.files:
            input_image = Image.open(file)
            input_image = input_image.convert("RGB")
            input_image = np.array(input_image)
            for i in range(input_image.shape[0]):
                for j in range(input_image.shape[1]):
                    color = tuple(input_image[i,j,:])
                    if color not in color_dict:
                        color_dict[color] = len(color_dict)
        return color_dict

    def convert_file(self, file, np_file=None):
        input_image = Image.open(file)
        input_image = input_image.convert("RGB")
        input_image = np.array(input_image)
        one_hot = np.zeros((self.class_num, input_image.shape[0], input_image.shape[1]), dtype=np.uint8)# (C, H, W)
        for i in range(input_image.shape[0]):
            for j in range(input_image.shape[1]):
                color = tuple(input_image[i,j,:])
                if color in self.color_dict:
                    one_hot[self.color_dict[color], i, j] = 1.0
        if np_file is not None:
            np.save(np_file, one_hot)
        return one_hot


