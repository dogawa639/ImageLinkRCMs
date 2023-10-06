from PIL import Image
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from utility import load_json, dump_json, Coord

# 画像データの読み込み，座標付与
class ImageData:
    def __init__(self, data_file):
        # data_file: json ["name": {"path": path, "z":z, "x": [x_min, x_max], "y": [y_min, y_max], "utm_num": utm_num}]
        self.data_list = load_json(data_file)
        self._get_xy()

    def _get_xy(self):
        # 座標取得
        for i in range(len(self.data_list)):
            coord = Coord(self.data_list[i]["utm_num"])
            x_min, y_max = ImageData.tile2latlon(self.data_list[i]["x"][0], self.data_list[i]["y"][0], self.data_list[i]["z"])
            x_max, y_min = ImageData.tile2latlon(self.data_list[i]["x"][1], self.data_list[i]["y"][1], self.data_list[i]["z"])
            x_min, y_min = coord.to_utm(x_min, y_min)
            x_max, y_max = coord.to_utm(x_max, y_max)
            self.data_list[i]["x_coord"] = [x_min, x_max]
            self.data_list[i]["y_coord"] = [y_min, y_max]

    @staticmethod
    def tile2latlon(x, y, z):
        # 座標変換 (タイル座標->タイル北西緯度経度)
        lon = (x / 2 ** z) * 360 - 180
        lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y / 2 ** z))))
        return lon, lat
