import os
import json
import pickle
import shutil

import numpy as np
import pandas as pd

import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from PIL import Image

__all__ = ["load_json", "dump_json", "load_pickle", "dump_pickle", "heron", "heron_vertex", "write_2d_ndarray", "write_1d_array", "load_2d_ndarray", "load_1d_array", "read_csv", "Coord"]


def load_json(file):
    with open(file) as f:
        data = json.load(f)
    return data


def dump_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def load_pickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def heron(a, b, c):
    # a, b, c: length of triangle
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def heron_vertex(v1, v2, v3):
    # v1, v2, v3: (x, y)
    a = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    b = np.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    c = np.sqrt((v3[0] - v1[0])**2 + (v3[1] - v1[1])**2)
    return heron(a, b, c)


def write_2d_ndarray(file, array):
    if type(array) != np.ndarray:
        raise Exception("array should be numpy.ndarray")
    if len(array.shape) != 2:
        raise Exception("array should be 2d")
    with open(file, "w") as f:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f.write(str(array[i,j]) + " ")
            f.write("\n")


def write_1d_array(file, array):
    if len(array.shape) != 1:
        raise Exception("array should be 1d")
    with open(file, "w") as f:
        for i in range(array.shape[0]):
            f.write(str(array[i]) + " ")
        f.write("\n")


def load_2d_ndarray(file):
    with open(file) as f:
        lines = f.readlines()
    array = np.zeros((len(lines), len(lines[0].split())))
    for i, line in enumerate(lines):
        array[i,:] = np.array([float(x) for x in line.split()])
    return array


def load_1d_array(file):
    with open(file) as f:
        line = f.readline()
    array = np.array([float(x) for x in line.split()])
    return array


def read_csv(file, encodings=["utf-8","shift-jis", "cp932"]):
    # save npz file in _csv folder
    f_name, d_name = os.path.split(file)
    save_dir = file.replace(".csv", "_csv")
    # npz file exists
    if os.path.exists(save_dir):
        if os.path.getmtime(save_dir) > os.path.getmtime(file):  # npz file is newer than csv file
            nval = np.load(os.path.join(save_dir, "val.npz"), allow_pickle=True)
            return pd.DataFrame({k: nval[k] for k in nval.files})
        else:
            shutil.rmtree(save_dir)
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            # save npz file
            os.makedirs(save_dir)
            kwargs = {str(col): df[col] for col in df.columns}
            np.savez(os.path.join(save_dir, "val.npz"), **kwargs)
            return df
        except:
            pass
    raise Exception("CSV file not loaded.")


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
        wgs_point = transform(self.project_from_utm, Point(x, y))
        return (wgs_point.x, wgs_point.y)


class KalmanFilter:
    def __init__(self, f, p, h):
        # f: 状態遷移行列
        # p: 誤差共分散行列
        # h: 観測行列

        # x: 状態ベクトル dim_x
        # z: 観測ベクトル dim_z
        self.f = f  # (dim_x, dim_x)
        self.p = p  # (dim_x, dim_x)
        self.h = h  # (dim_z, dim_x)


    





