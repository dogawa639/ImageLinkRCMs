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

__all__ = ["load_json", "dump_json", "load_pickle", "dump_pickle", "heron", "heron_vertex", "mutual_information", "write_2d_ndarray", "write_1d_array", "load_2d_ndarray", "load_1d_array", "read_csv", "Coord", "KalmanFilter", "Hungarian"]


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
    if s * (s - a) * (s - b) * (s - c) >= 0:
        return np.sqrt(s * (s - a) * (s - b) * (s - c))
    else:
        print("Invalid value for Heron's formula.", a, b, c)
        return 0


def heron_vertex(v1, v2, v3):
    # v1, v2, v3: (x, y)
    a = np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
    b = np.sqrt((v2[0] - v3[0])**2 + (v2[1] - v3[1])**2)
    c = np.sqrt((v3[0] - v1[0])**2 + (v3[1] - v1[1])**2)
    return heron(a, b, c)


def mutual_information(x, y):
    # x, y: 1d array
    # return: float, mutual information
    if len(x) != len(y):
        raise Exception("x, y should have the same length")
    cov = np.cov(x, y)
    if cov[0, 0] == 0.0 or cov[1, 1] == 0.0:
        return 0.0
    return -0.5 * np.log(1. - cov[0, 1] ** 2 / (cov[0, 0] * cov[1, 1]))


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
    print("file path: ",file)
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
    def __init__(self, f, h, q_fn, r_fn, g=None):
        # f: 状態遷移行列
        # h: 観測行列
        # q_fn: プロセスノイズの共分散行列を返す関数
        # r_fn: 観測ノイズの共分散行列を返す関数
        # g: 入力行列

        # x: 状態ベクトル (dim_x, num_obj)
        # z: 観測ベクトル (dim_z, num_obj)
        self.dim_x = f.shape[0]
        self.dim_z = h.shape[0]
        self.dim_u = g.shape[1] if g is not None else 0
        self.f = f  # (dim_x, dim_x)
        self.h = h  # (dim_z, dim_x)
        self.q_fn = q_fn  # (dim_x, dim_x), x->cov
        self.r_fn = r_fn  # (dim_z, dim_z), z->cov
        self.g = g  # (dim_x, dim_u)

        if self.f.shape != (self.dim_x, self.dim_x):
            raise Exception("f shape error")
        if self.h.shape != (self.dim_z, self.dim_x):
            raise Exception("h shape error")
        if self.g is not None and self.g.shape != (self.dim_x, self.dim_u):
            raise Exception("g shape error")

        self.xnn = None  # (num_obj, dim_x)
        self.pnn = None  # (num_obj, dim_x, dim_x)
        self.x_nnm1 = None  # (num_obj, dim_x)
        self.p_nnm1 = None  # (num_obj, dim_x, dim_x)
        self.active_idx = None  # (num_obj), bool

    def __len__(self):
        return 0 if self.xnn is None else self.xnn.shape[0]

    def add_obj(self, z):
        # initilize x, p
        # z: (num_obj, dim_z)
        z = np.array(z)
        num_obj = z.shape[0]
        x00s = np.zeros((num_obj, self.dim_x), dtype=np.float32)
        p00s = np.zeros((num_obj, self.dim_x, self.dim_x), dtype=np.float32)
        for i in range(num_obj):
            r = self.r_fn(z[i])  # (dim_z, dim_z)
            hphr_inv = np.linalg.pinv(np.dot(self.h, self.h.T) + r)
            k = np.dot(self.h.T, hphr_inv)  # (dim_x, dim_z)
            ikh = np.eye(self.dim_x) - np.dot(k, self.h)
            x00 = np.dot(k, z[i])  # (dim_x)
            p00 = np.dot(ikh, ikh.T) + np.linalg.multi_dot([k, r, k.T])  # (dim_x, dim_x)
            x00s[i] = x00
            p00s[i] = p00
        initial_idx = 0 if self.xnn is None else self.xnn.shape[0]
        active = np.ones(num_obj, dtype=bool)
        self.xnn = np.concatenate([self.xnn, x00s], axis=0) if self.xnn is not None else x00s
        self.pnn = np.concatenate([self.pnn, p00s], axis=0) if self.pnn is not None else p00s
        self.x_nnm1 = np.concatenate([self.x_nnm1, np.zeros_like(x00s)], axis=0) if self.x_nnm1 is not None else np.zeros_like(x00s)
        self.p_nnm1 = np.concatenate([self.p_nnm1, np.zeros_like(p00s)], axis=0) if self.p_nnm1 is not None else np.zeros_like(p00s)
        self.active_idx = np.concatenate([self.active_idx, active], axis=0) if self.active_idx is not None else active
        return list(range(initial_idx, initial_idx + num_obj))

    def predict(self, u=None):
        # update x_{n,n-1}, p_{n,n-1}
        # u: (num_obj, dim_u)
        if self.xnn is None:
            return
        num_obj = self.xnn.shape[0]
        if u is not None and u.shape[0] != num_obj:
            raise Exception("u shape error")

        for i in range(num_obj):
            if not self.active_idx[i]:
                continue
            self.x_nnm1[i] = np.dot(self.f, self.xnn[i]) + (np.dot(self.g, u[i]) if u is not None else 0)
            self.p_nnm1[i] = np.linalg.multi_dot([self.f, self.pnn[i], self.f.T]) + self.q_fn(self.x_nnm1[i])

    def correct(self, z):
        # update x_{n,n}, p_{n,n}
        # z: (num_obj, dim_z)
        z = np.array(z)
        if self.xnn is None:
            return
        num_obj = self.xnn.shape[0]
        if z.shape[0] != num_obj:
            raise Exception("z shape error")

        for i in range(num_obj):
            if not self.active_idx[i]:
                continue
            r = self.r_fn(z[i])
            hphr_inv = np.linalg.pinv(np.linalg.multi_dot([self.h, self.p_nnm1[i], self.h.T]) + r)
            k = np.linalg.multi_dot([self.p_nnm1[i], self.h.T, hphr_inv])
            ikh = np.eye(self.dim_x) - np.dot(k, self.h)
            dz = z[i] - np.dot(self.h, self.x_nnm1[i])
            mahalanobis = np.linalg.multi_dot([dz.T, hphr_inv, dz])
            self.xnn[i] = self.x_nnm1[i] + (np.dot(k, dz) if mahalanobis < 3.0 else 0)
            self.pnn[i] = np.linalg.multi_dot([ikh, self.p_nnm1[i], ikh.T]) + np.linalg.multi_dot([k, r, k.T])


class Hungarian:
    def __init__(self):
        self.n = None
        self.m = None
        self.row_starred = None
        self.col_starred = None
        self.row_covered = None
        self.col_covered = None
        self.optim = None
        self.done = None
        pass

    def compute(self, mat):
        mat = np.array(mat)
        self.n, self.m = mat.shape
        self.done = False

        mat = self._step1(mat)
        self._step2(mat)
        while not self.done:
            self._step3()
            min_val = self._step4(mat)
            if min_val == 0:
                break
            mat = self._step6(mat, min_val)
        return self.optim

    def _step1(self, mat):
        row_min = mat.min(axis=1, keepdims=True)
        mat = mat - row_min
        return mat

    def _step2(self, mat):
        # star zero
        self.row_starred = np.zeros(self.n, dtype=bool)
        self.col_starred = np.zeros(self.m, dtype=bool)
        self.optim = []
        for i in range(self.n):
            for j in range(self.m):
                if mat[i, j] == 0 and not self.row_starred[i] and not self.col_starred[j]:
                    self.row_starred[i] = True
                    self.col_starred[j] = True
                    self.optim.append((i, j))
                    break
        self.done = self.row_starred.sum() == self.n or self.col_starred.sum() == self.m
        return self.done

    def _step3(self):
        # cover columns with star zero
        self.row_covered = np.zeros(self.n, dtype=bool)
        self.col_covered = self.col_starred.copy()

    def _step4(self, mat):
        # cover row with prime zero while uncovering column
        mat_covered = mat.copy()  # covered -> -1
        mat_covered[:, self.col_covered] = -1
        while (mat_covered == 0).sum() > 0:
            for i in range(self.n):
                for j in range(self.m):
                    if mat_covered[i, j] == 0:
                        if not self.row_starred[i]:
                            self._step5(mat, i, j)  # change starred zero
                        else:
                            # uncover column j
                            self.row_covered[i] = True
                            self.col_covered[j] = False
                            mat_covered[~self.row_covered, j] = mat[~self.row_covered, j]
                            mat_covered[i, :] = -1
        self.done = self.row_starred.sum() == self.n or self.col_starred.sum() == self.m
        min_val = 0 if self.done else mat[np.ix_(~self.row_covered, ~self.col_covered)].min()
        return min_val

    def _step5(self, mat, i, j):
        # no starred zero in row i
        # change starred zero in col j
        # z1: starred zero in col j
        z2 = (-1, -1)
        self.optim.append((i, j))
        self.row_starred[i] = True
        self.col_starred[j] = True
        while z2 is not None:
            recal = False
            z2 = None
            for i2 in range(self.n):
                if i == i2:
                    continue
                if (i2, j) in self.optim:
                    z1 = (i2, j)
                    # z2: primed zero in row i2
                    for j2 in range(self.m):
                        if mat[i2, j2] == 0 and (i2, j2) not in self.optim:
                            z2 = (i2, j2)
                            self.optim.remove(z1)
                            self.optim.append(z2)

                            self.row_starred = np.zeros(self.n, dtype=bool)
                            self.col_starred = np.zeros(self.m, dtype=bool)
                            for z in self.optim:
                                self.row_starred[z[0]] = True
                                self.col_starred[z[1]] = True
                            i = i2
                            j = j2
                            recal = True
                            break
                    if recal:
                        break
        self._step3()

    def _step6(self, mat, min_val):
        mat[self.row_covered, :] += min_val
        mat[:, ~self.col_covered] -= min_val
        return mat







    





