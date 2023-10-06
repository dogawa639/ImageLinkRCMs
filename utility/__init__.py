import json

import numpy as np

import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from PIL import Image

__all__ = ["load_json", "dump_json", "Coord"]


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def dump_json(data, file):
    with open(file, "e") as f:
        json.dump(data, f, indent=4)


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


