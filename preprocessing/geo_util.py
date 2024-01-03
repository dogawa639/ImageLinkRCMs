import os
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from PIL import Image
from utility import Coord


__all__ = ["get_vertex_from_polygon", "tile2lonlat", "MapSegmentation", "GISSegmentation"]


def get_vertex_from_polygon(geo_file):
    # geo_file: geojson file
    # return: [(lon, lat)]
    gdf = gpd.read_file(geo_file)
    geom = gdf["geometry"].values[0]
    if type(geom) is not shapely.geometry.polygon.Polygon:
        raise Exception("geom should be Polygon")
    return [(x[0], x[1]) for x in geom.exterior.coords]


def tile2lonlat(x, y, z):
    # 座標変換 (タイル座標->タイル北西緯度経度)
    lon = (x / 2 ** z) * 360 - 180
    lat = np.rad2deg(np.arctan(np.sinh(np.pi * (1 - 2 * y / 2 ** z))))
    return lon, lat


class MapSegmentation:
    # 地図の色からクラスのone-hot vectorを作成する
    def __init__(self, files, max_class_num=10):
        self.files = files
        self.max_class_num = max_class_num  # containing other class
        self._load_color_dict() # self.color_dict, self.pixel_count  key: color_id
        self._sort_color()  # self.color_list  sorted bu color pixel count, [0]: majour class
        self.class_num = len(self.color_dict) + 1  # 1: other (class_num = 0)

    def convert_file(self, file, png_file=None, np_file=None, vis_file=None):
        # return: (class_num, H, W) ndarray
        input_image = Image.open(file)
        input_image = input_image.convert("RGB")  # (H, W, 3)
        input_image = np.array(input_image)
        input_image = MapSegmentation.get_color_id(input_image[:, :, 0], input_image[:, :, 1],
                                                   input_image[:, :, 2])  # (H, W)
        h, w = input_image.shape
        input_image = input_image.reshape(-1)

        class_num = min(self.class_num, self.max_class_num)
        if class_num > 255:
            raise ValueError("class_num should be less than 255")
        color_dict_small = {k: v for i, (k, v) in enumerate(self.color_dict.items()) if i < class_num - 1}  # key: color_id, value: class_num(>=1)

        def get_idx(cid):
            if cid in color_dict_small:
                return color_dict_small[cid]
            else:
                return 0  # other class
        idxs = np.vectorize(get_idx)(input_image).reshape(h, w)

        png_data = np.zeros((h, w), dtype=np.uint8)   # (H, W)
        one_hot = np.zeros((class_num, h, w), dtype=np.uint8)  # (C, H, W)
        for i in range(h):
            for j in range(w):
                png_data[i, j] = idxs[i, j]
                one_hot[idxs[i, j], i, j] = 1.0
        if png_file is not None:
            img = Image.fromarray(png_data)
            img.save(png_file)
        if np_file is not None:
            np.save(np_file, one_hot)
        if vis_file is not None:
            vis_data = png_data.astype(float)
            vis_data = vis_data / (class_num - 1) * 255
            vis_data = vis_data.astype(np.uint8)
            img = Image.fromarray(vis_data)
            img.save(vis_file)
        return png_data, one_hot

    # visualization
    def write_colormap(self, file=None):
        class_num = min(self.class_num, self.max_class_num)

        fig = plt.figure(dpi=15, figsize=(2, class_num))
        for i in range(class_num - 1):
            rgb = [i + 1] * 3
            ax = fig.add_subplot(class_num, 2, i*2+1)
            ax.imshow(np.array([[rgb]], dtype=np.uint8))
            ax.axis("off")
            rgb = [int((i + 1) / class_num * 255.)] * 3
            ax = fig.add_subplot(class_num, 2, i * 2 + 2)
            ax.imshow(np.array([[rgb]], dtype=np.uint8))
            ax.axis("off")
        ax = fig.add_subplot(class_num, 2, class_num*2-1)
        ax.imshow(np.zeros((1, 1, 3), dtype=np.uint8))
        ax.axis("off")
        ax = fig.add_subplot(class_num, 2, class_num*2)
        ax.imshow(np.zeros((1, 1, 3), dtype=np.uint8))
        ax.axis("off")
        if file is not None:
            plt.savefig(file)
        plt.show()

    def write_hist(self, file=None):
        # write histogram of pixel count for all colors (other class is not defined)
        class_num = min(self.class_num, self.max_class_num)
        fig, ax = plt.subplots()
        ax.bar(range(self.class_num - 1), self.pixel_count.values())
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
            input_image = MapSegmentation.get_color_id(input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2])  # (H, W)
            h, w = input_image.shape
            input_image = input_image.reshape((h * w))  # (H, W) -> (H * W)
            colors, counts = np.unique(input_image, return_counts=True)
            white_id = MapSegmentation.get_color_id(0, 0, 0)
            black_id = MapSegmentation.get_color_id(255, 255, 255)
            for i, color in enumerate(colors):
                if color == white_id or color == black_id:  # black and white
                    continue
                if color in color_dict:
                    pixel_count[color] += counts[i]
                else:
                    color_dict[color] = len(color_dict) + 1
                    pixel_count[color] = counts[i]

        self.color_dict = color_dict  # key: color_id, value: class_num
        self.pixel_count = pixel_count

    def _sort_color(self):
        pixel_count_sorted = sorted(self.pixel_count.items(), key=lambda x: x[1], reverse=True)
        self.color_list = [x[0] for x in pixel_count_sorted]
        self.color_dict = {k: i+1 for i, k in enumerate(self.color_list)}
        self.pixel_count = {k: self.pixel_count[k] for k in self.color_list}

    @staticmethod
    def get_color_id(r, g, b):
        # rgb: (r, g, b)
        return np.array(r, dtype=int) * (256 ** 2) + np.array(g, dtype=int) * 256 + np.array(b, dtype=int)


class GISSegmentation(MapSegmentation):
    # segmentation of GIS model
    def __init__(self, file, prop, utm_num, z, xs, ys, resolution=None, bbox=None, max_class_num=10):
        # bbox: [min_lon, min_lat, max_lon, max_lat]
        self.file = file
        self.prop = prop
        self.utm_num = utm_num
        self.coord = Coord(utm_num)
        self.xs = xs
        self.ys = ys
        self.resolution = resolution
        self.bbox = bbox
        self.max_class_num = max_class_num  # containing other class

        lon_min, lat_max = tile2lonlat(xs[0], ys[0], z)
        lon_max, lat_min = tile2lonlat(xs[-1], ys[-1], z)
        x_min, y_max = self.coord.to_utm(lon_min, lat_max)
        x_max, y_min = self.coord.to_utm(lon_max, lat_min)
        self.x_coords = (x_min, x_max)
        self.y_coords = (y_min, y_max)

        self.mask = None
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
        polygon = shapely.geometry.Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
        self.mask = gpd.GeoSeries([polygon], crs="EPSG:4326")

        self.raster_file = os.path.splitext(file)[0] + ".tif"
        self.prop2cid = self.gis2tif(file, prop, self.raster_file, utm_num, tif_range=(x_min, y_min, x_max, y_max), mask=self.mask, resolution=resolution)

        super().__init__([self.raster_file], max_class_num)
        self.write_color_info()

    def write_color_info(self):
        gdf = gpd.read_file(self.file)
        df = pd.DataFrame(gdf[[self.prop]])
        df = df[~df.duplicated()]

        class_num = min(self.class_num, self.max_class_num)
        color_dict_small = {k: v for i, (k, v) in enumerate(self.color_dict.items()) if i < class_num - 1}
        color_ids = np.vectorize(lambda x: MapSegmentation.get_color_id(*([self.prop2cid[x]] * 3)))(df[self.prop].values)
        class_nums = np.vectorize(lambda x: self.color_dict[x] if x in color_dict_small else class_num - 1)(color_ids)
        df["class_num"] = class_nums.astype(int)
        df.sort_values("class_num", inplace=True)
        df.to_csv(os.path.splitext(self.file)[0] + "_color.csv", index=False)

    @staticmethod
    def gis2tif(file, prop, out_file, utm_num, tif_range=None, mask=None, bbox=None, resolution=None):
        # tif_range [min_lon, min_lat, max_lon, max_lat]
        # bbox: [min_lon, min_lat, max_lon, max_lat]
        gdf = gpd.read_file(file, mask=mask, bbox=bbox, crs="EPSG:4326")
        gdf = gdf.to_crs(f"EPSG:{utm_num + 2442}")
        prop_unique = gdf[prop].unique()
        prop2cid = {p: i + 1 for i, p in enumerate(prop_unique)}

        if tif_range is None:
            tif_range = gdf["geometry"].total_bounds
        if resolution is None:
            resolution = 0.5
        dx = tif_range[2] - tif_range[0]
        dy = tif_range[3] - tif_range[1]
        shape = int(dx / resolution), int(dy / resolution)
        transformation = rasterio.transform.from_bounds(*tif_range, *shape)
        rasterized = rasterize(
            [(gdf.loc[i, "geometry"], prop2cid[gdf.loc[i, prop]]) for i in gdf.index],
            out_shape=shape[::-1],
            transform=transformation,
            fill=0,
            all_touched=True,
            dtype=rasterio.uint8
        )
        with rasterio.open(
            out_file,
            "w",
            crs=rasterio.CRS.from_string(f"EPSG:{utm_num + 2442}"),
            driver="GTiff",
            dtype=rasterio.uint8,
            count=1,
            width=shape[0],
            height=shape[1],
            transform=transformation
        ) as dst:
            dst.write(rasterized, indexes=1)
        return prop2cid
