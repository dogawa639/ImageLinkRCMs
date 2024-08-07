import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shapely
import geopandas as gpd

import scipy
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import csgraph
__all__ = ["Cell", "MeshNetwork"]


class Cell:
    def __init__(self, x, y, x_size, y_size, prop_dim):
        self.x = x
        self.y = y
        self.x_size = x_size
        self.y_size = y_size
        self.prop_dim = prop_dim
        self.prop = np.zeros(prop_dim, dtype=np.float32)  # each dim corresponds to person, building, vehicle, etc.

    def distance_to(self, cell):
        return self.get_distance(self, cell)

    def center_distance_to(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        if type(points) == tuple:
            return np.sqrt((self.x - points[0]) ** 2 + (self.y - points[1]) ** 2)
        elif type(points) == np.ndarray:
            return np.sqrt((self.x - points[:, 0]) ** 2 + (self.y - points[:, 1]) ** 2)  # (num_points)

    def contains(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        if type(points) == tuple:
            return (self.x_min <= points[0]) & (points[0] < self.x_max) & (self.y_min <= points[1]) & (points[1] < self.y_max)
        elif type(points) == np.ndarray:
            return (self.x_min <= points[:, 0]) & (points[:, 0] < self.x_max) & (self.y_min <= points[:, 1]) & (points[:, 1] < self.y_max)  # (num_points)

    def add_props(self, prop_dim):
        self.prop[prop_dim] += 1

    def remove_props(self, prop_dim):
        if type(prop_dim) == int:
            if self.prop[prop_dim] > 0:
                self.prop[prop_dim] -= 1
            else:
                raise ValueError("prop_dim must be positive or 0")
            self.prop[prop_dim] -= 1
        elif (self.prop[prop_dim] > 0).all():
            self.prop[prop_dim] -= 1
        else:
            raise ValueError("prop_dim must be positive or 0")

    def get_geometry(self):
        return shapely.geometry.box(self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def center(self):
        return (self.x, self.y)

    @property
    def x_min(self):
        return self.x - self.x_size / 2

    @property
    def x_max(self):
        return self.x + self.x_size / 2

    @property
    def y_min(self):
        return self.y - self.y_size / 2

    @property
    def y_max(self):
        return self.y + self.y_size / 2

    @staticmethod
    def get_distance(cell1, cell2):
        return np.sqrt((cell1.x - cell2.x) ** 2 + (cell1.y - cell2.y) ** 2)


class MeshNetwork:
    def __init__(self, coords, w_dim, h_dim, prop_dim):
        # coords: left, bottom, right, top
        self.coords = coords
        self.w_dim = w_dim
        self.h_dim = h_dim
        self.prop_dim = prop_dim

        self.x_size = (coords[2] - coords[0]) / w_dim
        self.y_size = (coords[3] - coords[1]) / h_dim

        self.center_coords = np.array([[[coords[0] + self.x_size * (i + 0.5), coords[1] + self.y_size * (h_dim - j - 0.5)]
                                       for i in range(w_dim)] for j in range(h_dim)])  # (h_dim, w_dim, 2)

        self.cells = [[Cell(*self.center_coords[i, j, :], self.x_size, self.y_size, prop_dim)
                       for j in range(w_dim)] for i in range(h_dim)]

    def contains(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        if type(points) == tuple:
            return (self.coords[0] <= points[0]) & (points[0] < self.coords[2]) & (self.coords[1] <= points[1]) & (points[1] < self.coords[3])
        else:
            return (self.coords[0] <= points[:, 0]) & (points[:, 0] < self.coords[2]) & (self.coords[1] <= points[:, 1]) & (points[:, 1] < self.coords[3])  # (num_points)

    def get_idx(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        # idx(0, 0): top left
        # return (y_idx, x_idx) or np.array([[y_idx1, x_idx1], [y_idx2, x_idx2], ...])
        contained = self.contains(points)
        if type(points) == tuple:
            if not contained:
                return None
            x_idx = int((points[0] - self.coords[0]) / self.x_size)
            y_idx = self.h_dim - int((points[1] - self.coords[1]) / self.y_size) - 1
            return (y_idx, x_idx)
        else:
            x_idx = ((points[:, 0] - self.coords[0]) / self.x_size).astype(int)
            y_idx = (self.h_dim - (points[:, 1] - self.coords[1]) / self.y_size - 1).astype(int)
            return np.array([[y_idx[i], x_idx[i]] if contained[i] else None for i in
                    range(len(points))])  # (num_points, 2)

    def get_cell(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        contained = self.contains(points)
        if type(points) == tuple:
            if not contained:
                return None
            x_idx = int((points[0] - self.coords[0]) / self.x_size)
            y_idx = self.h_dim - int((points[1] - self.coords[1]) / self.y_size) - 1
            return self.cells[y_idx][x_idx]
        else:
            x_idx = ((points[:, 0] - self.coords[0]) / self.x_size).astype(int)
            y_idx = (self.h_dim - (points[:, 1] - self.coords[1]) / self.y_size - 1).astype(int)
            return [self.cells[y_idx[i]][x_idx[i]] if contained[i] else None for i in range(len(points))]  # (num_points)

    def distance_from(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        if type(points) == tuple:
            return np.sqrt((self.center_coords[:, :, 0] - points[0]) ** 2 + (self.center_coords[:, :, 1] - points[1]) ** 2)  # (h_dim, w_dim)
        else:
            return np.sqrt((np.expand_dims(self.center_coords[:, :, 0], 0) - np.expand_dims(points[:, 0], (1, 2))) ** 2 + (np.expand_dims(self.center_coords[:, :, 1], 0) - np.expand_dims(points[:, 1], (1, 2))) ** 2)  # (num_points, h_dim, w_dim)

    def angle_from(self, points):
        # point: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        if type(points) == tuple:
            center_coords = self.center_coords.reshape(-1, 2)  # (h_dim * w_dim, 2)
            angles = self.get_angle(points, center_coords).reshape(self.h_dim, self.w_dim)  # (h_dim,  w_dim)
            return angles
        elif type(points) == np.ndarray:
            center_coords = self.center_coords.reshape(-1, 2)
            angles = np.array([self.get_angle(tuple(point), center_coords).reshape(self.h_dim, self.w_dim) for point in points])   # (num_points, h_dim,  w_dim)
            return angles

    def load_prop(self, npy_path):
        prop_array = np.load(npy_path)
        if prop_array.shape[0] > self.prop_dim:
            prop_array = prop_array[:self.prop_dim]
        elif prop_array.shape[0] < self.prop_dim:
            prop_array = np.concatenate([prop_array, np.zeros((self.prop_dim - prop_array.shape[0], self.h_dim, self.w_dim))], axis=0)
        for i in range(self.h_dim):
            for j in range(self.w_dim):
                self.cells[i][j].prop = prop_array[:, i, j]

    def add_prop(self, points, prop_dims):
        # points: np.array([[x1, y1], [x2, y2], ...])
        # prop_dims: np.array([prop_dim1, prop_dim2, ...])
        if len(points) != len(prop_dims):
            raise ValueError("points and prop_dims must have the same length")
        cells = self.get_cell(points)
        for i, cell in enumerate(cells):
            if cell is None:
                continue
            cell.add_props(prop_dims[i])

    def move_props(self, from_points, to_points, prop_dims):
        # from_points: np.array([[x1, y1], [x2, y2], ...])
        # to_points: np.array([[x1, y1], [x2, y2], ...])
        # prop_dims: np.array([prop_dim1, prop_dim2, ...])
        if len(from_points) != len(to_points) or len(from_points) != len(prop_dims):
            raise ValueError("points and prop_dims must have the same length")
        from_cells = self.get_cell(from_points)
        to_cells = self.get_cell(to_points)
        for i, (from_cell, to_cell) in enumerate(zip(from_cells, to_cells)):
            if from_cell is None or to_cell is None:
                continue
            from_cell.remove_props(prop_dims[i])
            to_cell.add_props(prop_dims[i])

    def remove_props(self, points, prop_dims):
        # points: np.array([[x1, y1], [x2, y2], ...])
        # prop_dims: np.array([prop_dim1, prop_dim2, ...])
        if len(points) != len(prop_dims):
            raise ValueError("points and prop_dims must have the same length")
        cells = self.get_cell(points)
        for i, cell in enumerate(cells):
            if cell is None:
                continue
            cell.remove_props(prop_dims[i])

    def get_prop_array(self, min_x_idx=None, min_y_idx=None, max_x_idx=None, max_y_idx=None):
        # min_x_idx, min_y_idx, max_x_idx, max_y_idx: int
        # return: (prop_dim, max_y_idx - min_y_idx, max_x_idx - min_x_idx)
        # max_idx is NOT contained.
        if min_x_idx is None:
            min_x_idx = 0
        if min_y_idx is None:
            min_y_idx = 0
        if max_x_idx is None:
            max_x_idx = self.w_dim
        if max_y_idx is None:
            max_y_idx = self.h_dim

        min_x_idx = min(max(min_x_idx, 0), self.w_dim)
        min_y_idx = min(max(min_y_idx, 0), self.h_dim)
        max_x_idx = max(min(max_x_idx, self.w_dim), 0)
        max_y_idx = max(min(max_y_idx, self.h_dim), 0)

        return np.array([[self.cells[y_idx][x_idx].prop for x_idx in range(min_x_idx, max_x_idx)] for y_idx in range(min_y_idx, max_y_idx)]).transpose((2, 0, 1))  # (prop_dim, max_y_idx - min_y_idx, max_x_idx - min_x_idx)

    def get_surroundings_prop(self, point, d_x, d_y=None):
        # point: (x, y)
        # d: int
        # return: (prop_dim, d * 2 + 1, d * 2 + 1)
        if d_y is None:
            d_y = d_x
        idx = self.get_idx(point)
        if idx is None:
            return None
        return self.get_prop_array(idx[1] - d_x, idx[0] - d_y, idx[1] + d_x + 1, idx[0] + d_y + 1)

    def get_surroundings_idxs(self, point, d_x, d_y=None):
        # point: (x, y)
        # d: int
        # return: (d * 2 + 1, d * 2 + 1, 2)
        if d_y is None:
            d_y = d_x
        idx = self.get_idx(point)
        if idx is None:
            return None
        min_x = min(max(idx[1] - d_x, 0), self.w_dim)
        min_y = min(max(idx[0] - d_y, 0), self.h_dim)
        max_x = max(min(idx[1] + d_x + 1, self.w_dim), 0)
        max_y = max(min(idx[0] + d_y + 1, self.h_dim), 0)

        return (min_x, min_y, max_x, max_y)

    def get_center_points(self, idxs):
        # idxs: np.array(num_points, 2) or tuple
        # return: (num_points, 2) or tuple
        if type(idxs) == tuple:
            return self.cells[idxs[0]][idxs[1]].center
        else:
            return np.array([self.cells[idxs[i, 0]][idxs[i, 1]].center for i in range(len(idxs))])

    def clear_prop(self, prop_dim):
        for i in range(self.h_dim):
            for j in range(self.w_dim):
                self.cells[i][j].prop[prop_dim] = 0

    # visualize
    def show_props(self):
        prop_array = self.get_prop_array(0, 0, self.w_dim, self.h_dim)
        fig = plt.figure(figsize=(6.4, 2.4 * self.prop_dim))
        for i in range(self.prop_dim):
            ax = fig.add_subplot(self.prop_dim, 1, i + 1)
            ax.set_title("Prop {}".format(i))
            ax.set_aspect("equal")
            ax.imshow(prop_array[i, :, :])
        plt.show()

    def show_grid(self, ax=None, vals=None, *args, **kwargs):
        show_fig = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            show_fig = True
        if vals is None:
            vals = np.full((self.h_dim, self.w_dim, 3), 255)
        vals = np.flipud(vals)
        im = ax.imshow(vals, *args, **kwargs)
        # grid
        ax.set_xticks(np.arange(-0.5, self.w_dim, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.h_dim, 1), minor=True)
        ax.grid(which='minor', color="black", linestyle='-', linewidth=0.5)
        # ticks labels
        ax.set_xticks(np.arange(0, self.w_dim, 5))
        ax.set_yticks(np.arange(0, self.h_dim, 5))
        ax.set_xticklabels(self.center_coords[0, ::5, 0], rotation=30)
        ax.set_yticklabels(self.center_coords[::-5, 0, 1])
        # remove minor ticks
        ax.tick_params(which="minor", bottom=False, left=False)

        if show_fig:
            plt.show()
        return im

    def show_path(self, path_idxs, vals=None):
        # path_idxs: np.array(num_points, 2) (y_idx, x_idx)
        # vals: np.array((h_dim, w_dim))
        fig = plt.figure()
        if vals is None:
            vals = np.full((self.h_dim, self.w_dim, 3), 255)

        vals = np.flipud(vals)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(vals)
        # grid
        ax.set_xticks(np.arange(-0.5, self.w_dim, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.h_dim, 1), minor=True)
        ax.grid(which='minor', color="black", linestyle='-', linewidth=0.5)
        # ticks labels
        ax.set_xticks(np.arange(0, self.w_dim, 5))
        ax.set_yticks(np.arange(0, self.h_dim, 5))
        ax.set_xticklabels(self.center_coords[0, ::5, 0], rotation=30)
        ax.set_yticklabels(self.center_coords[::-5, 0, 1])
        # remove minor ticks
        ax.tick_params(which="minor", bottom=False, left=False)

        x = path_idxs[:-1, 1]
        y = self.h_dim - path_idxs[:-1, 0] - 1
        u = path_idxs[1:, 1] - path_idxs[:-1, 1]
        v = - (path_idxs[1:, 0] - path_idxs[:-1, 0])
        ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color="red")
        plt.show()

    def write_geo_file(self, utf_num, file_path):
        # utf_num: int
        geometries = []
        properties = []
        for i in range(self.h_dim):
            for j in range(self.w_dim):
                geom = self.cells[i][j].get_geometry()
                prop = self.cells[i][j].prop
                geometries.append(geom)
                properties.append(prop.tolist())
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=f"EPSG:{2442+utf_num}")
        gdf.to_file(file_path)

    @staticmethod
    def get_angle(start_point, end_points):
        # start_point: (x, y)
        # end_points: (x, y) or np.array([[x1, y1], [x2, y2], ...])
        # angle: 0-360
        if type(end_points) == tuple:
            dx = end_points[0] - start_point[0]
            dy = end_points[1] - start_point[1]
        elif type(end_points) == np.ndarray:
            dx = end_points[:, 0] - start_point[0]
            dy = end_points[:, 1] - start_point[1]
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
    def interpolate(idx1, idx2):
        # idx: (y_idx, x_idx)
        # return: np.array(num_points, 2)
        # includes idx1 and idx2
        dy = idx2[0] - idx1[0]
        dx = idx2[1] - idx1[1]
        idx1 = [*idx1]; idx2 = [*idx2]
        path = [[*idx1]]
        while dx != 0 or dy != 0:
            if abs(dx) > abs(dy):
                idx1[1] += np.sign(dx)
            elif abs(dx) < abs(dy):
                idx1[0] += np.sign(dy)
            else:
                idx1[0] += np.sign(dy)
                idx1[1] += np.sign(dx)
            dx = idx2[1] - idx1[1]
            dy = idx2[0] - idx1[0]
            path.append([*idx1])
        return np.array(path)



