from PIL import Image
import cv2

import geopandas as gpd
import shapely
import rasterio
from rasterio.features import rasterize
from rasterio.transform import GroundControlPoint
from rasterio.mask import mask as rasterio_mask
from rasterio.enums import Resampling

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import tensor
from torch.utils.data import DataLoader
import time
import shutil
import os

from utility import load_json, dump_json, Coord
from preprocessing.geo_util import GISSegmentation
from preprocessing.dataset import PatchDataset
from preprocessing.geo_util import tile2lonlat
from preprocessing.dataset import ImageDatasetBase
__all__ = ["CompressedImageData", "LinkImageData", "SatelliteImageData", "OneHotImageData"]


class CompressedImageData:
    def __init__(self, link_image_data_list, merge_type="sum"):
        self.link_image_data_list = link_image_data_list
        self.link_num = link_image_data_list[0].link_num
        self.lids = link_image_data_list[0].lids
        self.link_num = len(self.lids)

        self.merge_type = merge_type  # sum or concat

    def __len__(self):
        return self.link_num

    def load_compressed(self, idx):
        # idx: int
        # compressed: npy
        # return list of compressed(None or tensor)
        compressed = None
        for link_image_data in self.link_image_data_list:
            compressed_tmp = link_image_data.load_compressed(idx)
            if compressed_tmp is None:
                continue
            if self.merge_type == "concat":
                compressed_tmp = np.expand_dims(compressed_tmp, 0)
            if compressed is None:
                compressed = compressed_tmp
            elif self.merge_type == "sum":
                compressed += compressed_tmp
            elif self.merge_type == "concat":
                compressed = np.concatenate((compressed, compressed_tmp), axis=0)
        return compressed


class LinkImageData:
    # load link images from data_dir
    # data_dir - name - link_id.png
    image_dataset_base = ImageDatasetBase(crop=False, affine=False, transform_coincide=False, flip=False)
    def __init__(self, data_file, nw_data):
        self.data_list = load_json(data_file)
        self.nw_data = nw_data
        self.lids = nw_data.lids
        self.link_num = len(self.lids)

    def __len__(self):
        return len(self.data_list)

    def load_image(self, idx):
        # idx: int
        # image: (3, H, W) RGB or (H, W) gray
        # return: list of image(None or tensor (3, H, W) RGB or (H, W) gray)
        images = []
        for data in self.data_list:
            if "data_dir" in data:
                path = os.path.join(data["data_dir"], f"{self.lids[idx]}.png")
                image = None
                if os.path.exists(path):
                    image = self.image_dataset_base.preprocess(Image.open(path))
                images.append(image)
            else:
                print(f"No data dir is set for {data}.")
        return images

    def load_compressed(self, idx):
        # idx: int
        # compressed: npy
        # return list of compressed(None or tensor)
        compressed = None
        for data in self.data_list:
            if "data_dir" in data:
                path = os.path.join(data["data_dir"], f"{self.lids[idx]}.npy")
                if os.path.exists(path):
                    tmp_compressed = np.load(path).astype(np.float32)
                    if compressed is None:
                        compressed = tmp_compressed
                    else:
                        compressed = compressed + tmp_compressed
        return compressed

    def compress_images(self, encoder, device="cpu"):
        # no backward process
        encoder = encoder.to(device)
        for data in self.data_list:
            if "data_dir" in data:
                tmp_dir = data["data_dir"]
                for i, lid in enumerate(self.lids):
                    path = os.path.join(tmp_dir, f"{lid}.png")
                    if os.path.exists(path):
                        encoder.eval()
                        image = self.image_dataset_base.preprocess(Image.open(path))
                        image = image.unsqueeze(0).to(device)
                        compressed = encoder(image).detach().cpu().numpy().flatten()
                        np.save(os.path.join(tmp_dir, f"{lid}.npy"), compressed)


# 画像データの読み込み，座標付与
class SatelliteImageData:
    # currently only the same utm_num for all model is allowed.
    required_prop = ["name", "path", "z", "x", "y", "utm_num"]
    def __init__(self, data_file, resolution=None, output_data_file=None, output_mask_file=None):
        # data_file: json [{"name": name, "path": path, "z":z, "x": [x_min, x_max], "y": [y_min, y_max], "utm_num": utm_num}]
        # (x_max and y_max are not included in model.)
        # initial transformation: coordinate calculation (coords, bounds), resolution calculation (resolution)
        # visualization: gtif
        # note: Image.open(path) returns RGB (H, W, 3) or gray (H, W)
        #       cv2.imread(path) returns BGR (H, W, 3) or gray (H, W)
        self.data_list = load_json(data_file)
        self.resolution = resolution  # m / pixel
        self.output_data_file = output_data_file
        self.output_mask_file = output_mask_file
        utm_num = None
        for data in self.data_list:
            for prop in SatelliteImageData.required_prop:
                if prop not in data:
                    print(f"property {prop} is not in the model")
                    return
            if utm_num is not None and utm_num != data["utm_num"]:
                print("utm_num should be the same for all model.")
                return

        self._get_xy()  # coords [(x, y)] (north west, south west, south east, north east), bounds [x or y] (west, south, east, north)
        self._get_tif()  # replace path with gtif
        self._clip_shared()  # replace path with clipped gtif, update (coords, bounds)
        self._set_resolution()  # resolution
    
    def __len__(self):
        return len(self.data_list)

    def set_voronoi(self, nw_data):
        # network: Network in network.py
        self.nw_data = nw_data
        center_points = [nw_data.edges_all[v[0]].center for v in nw_data.undir_edges.values()]
        nodes = [(v.x, v.y) for v in nw_data.nodes.values()]

        # cv2 subdiv for each image
        for data in self.data_list:
            with rasterio.open(data["path"]) as src:
                profile = src.profile
            transformation = profile["transform"]
            image = np.array(Image.open(data["path"]))  # RGB (H, W, 3) or gray (H, W)
            h, w = image.shape[:2]

            center_idxs = [rasterio.transform.rowcol(transformation, x, y) for x,y in center_points]  # row, col

            subdiv = cv2.Subdiv2D((0, 0, w, h))
            for i in range(len(center_points)):
                subdiv.insert((center_idxs[i][1], center_idxs[i][0]))
            voronoi, _ = subdiv.getVoronoiFacetList([])

            # voronoi image
            imgv = np.zeros((h, w, 3), dtype=np.uint8)  # RGB
            undir_ids = list(nw_data.undir_edges.keys())
            for i, p in enumerate(f.astype(int) for f in voronoi):
                color = SatelliteImageData.get_rbg_from_number(undir_ids[i])  # rgb
                color = (color[2], color[1], color[0])  # bgr
                cv2.fillConvexPoly(imgv, p, color)
            imgv = cv2.cvtColor(imgv, cv2.COLOR_BGR2RGB)
            basename, ext = os.path.splitext(data["path"])
            voronoi_path = basename + "_voronoi.tif"
            data["voronoi_path"] = voronoi_path
            profile["count"] = 3
            with rasterio.open(voronoi_path, "w", **profile) as dst:
                dst.write(imgv[:, :, 0], indexes=1)
                dst.write(imgv[:, :, 1], indexes=2)
                dst.write(imgv[:, :, 2], indexes=3)

            # mask of convex hull
            node_idxs = [rasterio.transform.rowcol(transformation, x, y) for x,y in nodes]

            node_points = [[node_idxs[i][1], node_idxs[i][0]] for i in range(len(nodes)) if node_idxs[i][0] >= 0 and node_idxs[i][0] < image.shape[0] and node_idxs[i][1] >= 0 and node_idxs[i][1] < image.shape[1]]

            convex_hull = cv2.convexHull(np.array(node_points))
            convex_hull_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(convex_hull_mask, convex_hull, 255)
            basename, ext = os.path.splitext(data["path"])
            convex_hull_path = basename + "_convex_hull.tif"
            data["convex_hull_path"] = convex_hull_path
            profile["count"] = 1
            with rasterio.open(convex_hull_path, "w", **profile) as dst:
                dst.write(convex_hull_mask, indexes=1)
        self._output_data_file()

    def load_images(self):
        # for i in range(len(image_data)):
        #    image = image_data.load_images()
        data_idx = 0
        while True:
            if data_idx >= len(self.data_list):
                break
            data = self.data_list[data_idx]
            image = np.array(Image.open(data["path"]), dtype=np.uint8)  # RGB (H, W, 3) or gray (H, W)
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)  # (3, H, w)
            yield image
            data_idx += 1

    # call after set_voronoi
    def load_link_patches(self, patch_size=256, min_ratio=0.8):
        # 直前に呼び出したset_voronoiに対応する．
        # for i in range(len(image_data)):
        #    patches = image_data.load_link_patches()
        # min_ratio: minimum ratio of non-nodata pixels
        data_idx = 0
        while True:
            if data_idx >= len(self.data_list):
                break
            data = self.data_list[data_idx]
            if "voronoi_path" not in data or "convex_hull_path" not in data:
                print("Set voironoi first!")
                break

            voronoi = np.array(Image.open(data["voronoi_path"]))  # RGB (H, W, 3)
            voronoi_lids = SatelliteImageData.get_number_from_rgb(voronoi[:,:,0].astype(int), voronoi[:,:,1].astype(int), voronoi[:,:,2].astype(int))  # (H, W)
            del voronoi

            convex_hull_mask = np.array(Image.open(data["convex_hull_path"]))  # gray (H, W)
            voronoi_lids = voronoi_lids * (convex_hull_mask > 0) + (convex_hull_mask == 0) * -1  # gray (H, W)
            del convex_hull_mask

            # link patches
            patches = []
            image = np.array(Image.open(data["path"]))  # RGB (H, W, 3) or gray (H, W)
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)  # RGB (3, H, W)
            else:
                image = np.expand_dims(image, 0)  # (1, H, W)
            unmasked_size = int(patch_size * (min_ratio ** 0.5))
            for edge in self.nw_data.edges_all.values():
                patch = np.zeros((image.shape[0], patch_size, patch_size), dtype=np.uint8)
                target_mask = (voronoi_lids == edge.undir_id)
                x_idxs = np.where(target_mask.sum(axis=0) > 0)[0].tolist()
                y_idxs = np.where(target_mask.sum(axis=1) > 0)[0].tolist()

                if len(x_idxs) < unmasked_size:
                    unmask_num = unmasked_size - len(x_idxs)
                    min_x = x_idxs[0] - unmask_num // 2
                    x_idxs = list(range(min_x, x_idxs[0])) + x_idxs + list(range(x_idxs[-1] + 1, min_x + unmasked_size))
                if len(y_idxs) < unmasked_size:
                    unmask_num = unmasked_size - len(y_idxs)
                    min_y = y_idxs[0] - unmask_num // 2
                    y_idxs = list(range(min_y, y_idxs[0])) + y_idxs + list(range(y_idxs[-1] + 1, min_y + unmasked_size))
                print(f"patch shape link {edge.id} ({len(x_idxs)}, {len(y_idxs)})")

                if len(x_idxs) > 0 and len(y_idxs) > 0:
                    cropped_image = image[np.ix_(list(range(image.shape[0])), y_idxs, x_idxs)]
                    if cropped_image.shape[1] > patch_size:
                        pad = (cropped_image.shape[1] - patch_size) // 2
                        cropped_image = cropped_image[:, pad:pad+patch_size, :]
                    if cropped_image.shape[2] > patch_size:
                        pad = (cropped_image.shape[2] - patch_size) // 2
                        cropped_image = cropped_image[:, :, pad:pad+patch_size]
                    pad = (patch_size - np.array(cropped_image.shape)[1:]) // 2
                    patch[:, pad[0]:pad[0]+cropped_image.shape[1], pad[1]:pad[1]+cropped_image.shape[2]] = cropped_image
                if patch.shape[0] == 1:
                    patch = np.squeeze(patch, 0)
                patches.append(patch)
            del image
            print(f"patch for data {data_idx} is loaded.")

            yield patches
            data_idx += 1

    def compress_patches(self, comp_fn, patch_size=256, device="cpu"):
        # callable: function to compress patches (BS, C, H, W)->(BS, mid_dim)
        # for i in range(len(image_data)):
        #    patches = image_data.load_link_patches()
        for i, patches in enumerate(self.load_link_patches(patch_size=patch_size)):
            if len(patches[0].shape) != 4:
                raise ValueError("This function does not support gray scale patches.")
            dataset = PatchDataset(patches)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
            cnt = 0
            for j, batch in enumerate(dataloader):
                batch = batch.to(device)
                compressed_tmp = comp_fn(batch)
                bs_tmp = len(compressed_tmp)
                if j == 0:
                    mid_dim = compressed_tmp.shape[1]
                    compressed = np.zeros((len(patches), mid_dim), dtype=np.float32)
                compressed[cnt:cnt+bs_tmp] = compressed_tmp.detach.cpu().numpy()
                cnt += bs_tmp
            basename, ext = os.path.splitext(self.data_list[i]["path"])
            path = basename + "_compressed.npy"
            self.data_list[i]["compressed_path"] = path
            np.save(path, compressed)
        self._output_data_file()

    def load_compressed_patches(self, i):
        # i: int
        # for i in range(len(image_data)):
        #    patches = image_data.load_link_patches()
        if "compressed_path" not in self.data_list[i]:
            print("Set compressed patches first!")
            return None
        compressed = np.load(self.data_list[i]["compressed_path"])  # (link_num, mid_dim)
        return compressed

    def load_link_masks(self):
        # 直前に呼び出したset_voronoiに対応する．
        # for i in range(len(image_data)):
        #    mask = image_data.load_link_masks()
        data_idx = 0
        while True:
            if data_idx >= len(self.data_list):
                break
            data = self.data_list[data_idx]
            if "voronoi_path" not in data or "convex_hull_path" not in data:
                print("Set voironoi first!")
                break

            voronoi = np.array(Image.open(data["voronoi_path"]))  # RGB (H, W, 3)
            voronoi_lids = SatelliteImageData.get_number_from_rgb(voronoi[:,:,0], voronoi[:,:,1], voronoi[:,:,2]).astype(int)  # (H, W)
            del voronoi

            convex_hull_mask = np.array(Image.open(data["convex_hull_path"]))  # gray (H, W)
            voronoi_lids = voronoi_lids * (convex_hull_mask > 0) + (convex_hull_mask == 0) * -1
            del convex_hull_mask

            yield voronoi_lids
            data_idx += 1

    def set_datafolder(self, data_dir, patch_size=256, min_ratio=0.8):
        # data_dir: path
        # data_dir - name - link_id.png
        # min_ratio: minimum ratio that the non-nodata pixels
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for i, patches in enumerate(self.load_link_patches(patch_size=patch_size, min_ratio=min_ratio)):
            tmp_dir = os.path.join(data_dir, self.data_list[i]["name"])
            self.data_list[i]["data_dir"] = tmp_dir

            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)  # [np.array((3, H, W), dtype=np.uin8)] RGB
            for j, patch in enumerate(patches):  # (3, H, W) RGB or (H, W) gray
                path = os.path.join(tmp_dir, f"{self.nw_data.lids_all[j]}.png")
                if len(patch.shape) == 3:
                    patch = patch.transpose(1, 2, 0)  # (H, W, 3) RGB
                patch = Image.fromarray(patch)
                patch.save(path)
        self._output_data_file()

    # internal functions
    def _get_xy(self):
        # 座標取得
        for i in range(len(self.data_list)):
            coord = Coord(self.data_list[i]["utm_num"])
            x_0, y_0 = tile2lonlat(self.data_list[i]["x"][0], self.data_list[i]["y"][0], self.data_list[i]["z"])
            x_1, y_1 = tile2lonlat(self.data_list[i]["x"][0], self.data_list[i]["y"][1], self.data_list[i]["z"])
            x_2, y_2 = tile2lonlat(self.data_list[i]["x"][1], self.data_list[i]["y"][1], self.data_list[i]["z"])
            x_3, y_3 = tile2lonlat(self.data_list[i]["x"][1], self.data_list[i]["y"][0], self.data_list[i]["z"])
            x_0, y_0 = coord.to_utm(x_0, y_0)
            x_1, y_1 = coord.to_utm(x_1, y_1)
            x_2, y_2 = coord.to_utm(x_2, y_2)
            x_3, y_3 = coord.to_utm(x_3, y_3)
            self.data_list[i]["coords"] = [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)]

            coords = np.array([(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)])
            bounds = [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]  # west, south, east, north
            self.data_list[i]["bounds"] = bounds
        self._output_data_file()

    def _get_tif(self):
        for i, data in enumerate(self.data_list):
            image = Image.open(self.data_list[i]["path"])
            w, h = image.size
            image = np.array(image)
            gcps = [
                GroundControlPoint(row=0.0, col=0.0, x=self.data_list[i]["coords"][0][0],
                                   y=self.data_list[i]["coords"][0][1]),
                GroundControlPoint(row=h, col=0.0, x=self.data_list[i]["coords"][1][0],
                                   y=self.data_list[i]["coords"][1][1]),
                GroundControlPoint(row=h, col=w, x=self.data_list[i]["coords"][2][0],
                                   y=self.data_list[i]["coords"][2][1]),
                GroundControlPoint(row=0.0, col=w, x=self.data_list[i]["coords"][3][0],
                                   y=self.data_list[i]["coords"][3][1])
            ]
            transformation = rasterio.transform.from_gcps(gcps)
            basename, ext = os.path.splitext(self.data_list[i]["path"])
            new_path = basename + "_transformed.tif"
            utm_num = self.data_list[i]["utm_num"]
            with rasterio.open(
                    new_path,
                    "w",
                    crs=rasterio.CRS.from_string(f"EPSG:{utm_num + 2442}"),
                    driver="GTiff",
                    dtype=rasterio.uint8,
                    count=3,
                    width=w,
                    height=h,
                    transform=transformation
            ) as dst:
                dst.write(image[:, :, 0], indexes=1)
                dst.write(image[:, :, 1], indexes=2)
                dst.write(image[:, :, 2], indexes=3)
            self.data_list[i]["path"] = new_path
            coords, bounds = SatelliteImageData.get_gtif_coord(new_path)
            self.data_list[i]["coords"] = coords
            self.data_list[i]["bounds"] = bounds
        self._output_data_file()

    def _clip_shared(self):
        # bounds: [west, south, east, north]
        bounds = None
        for data in self.data_list:
            # get shared bounds of all model
            if bounds is None:
                bounds = list(data["bounds"])
            else:
                bounds = [max(bounds[0], data["bounds"][0]), max(bounds[1], data["bounds"][1]), min(bounds[2], data["bounds"][2]), min(bounds[3], data["bounds"][3])]
        utm_num = self.data_list[0]["utm_num"]
        coords = [(bounds[0], bounds[3]), (bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3])]
        self.mask = shapely.geometry.Polygon(coords)  # utm coords
        if self.output_mask_file is not None:
            mask = gpd.GeoDataFrame({"geometry": self.mask}, crs=f"EPSG:{utm_num + 2442}")
            mask.to_file(self.output_mask_file)

        for i, data in enumerate(self.data_list):
            basename, ext = os.path.splitext(data["path"])
            new_path = basename + "_clipped.tif"
            with rasterio.open(data["path"]) as src:
                masked, transformation = rasterio_mask(src, [self.mask], crop=True)  # mask: (C, H, W)
                c, h, w = masked.shape
            with rasterio.open(
                    new_path,
                    "w",
                    crs=rasterio.CRS.from_string(f"EPSG:{utm_num + 2442}"),
                    driver="GTiff",
                    dtype=rasterio.uint8,
                    count=c,
                    width=w,
                    height=h,
                    transform=transformation
            ) as dst:
                for j in range(c):
                    dst.write(masked[j, :, :], indexes=j+1)
            coords, bounds = SatelliteImageData.get_gtif_coord(new_path)
            self.data_list[i]["path"] = new_path
            self.data_list[i]["coords"] = coords
            self.data_list[i]["bounds"] = bounds
        self._output_data_file()

    def _set_resolution(self):
        # overwrite the tif
        if self.resolution is None:  # set the highest resolution in the model
            for i in range(len(self.data_list)):
                img = Image.open(self.data_list[i]["path"])
                w, h = img.size
                dx = (self.data_list[i]["coords"][3][0] - self.data_list[i]["coords"][0][0] + self.data_list[i]["coords"][2][0] - self.data_list[i]["coords"][1][0]) / 2
                dy = (self.data_list[i]["coords"][0][1] - self.data_list[i]["coords"][1][1] + self.data_list[i]["coords"][3][1] - self.data_list[i]["coords"][2][1]) / 2
                res = max(dx / w, dy / h)
                if self.resolution is None:
                    self.resolution = res  # m/pixel
                else:
                    self.resolution = max(res, self.resolution)
        for i, data in enumerate(self.data_list):
            w = int((data["bounds"][2] - data["bounds"][0]) / self.resolution)
            h = int((data["bounds"][3] - data["bounds"][1]) / self.resolution)
            with rasterio.open(data["path"]) as src:
                upsampled = src.read(
                    out_shape=(src.count, h, w),
                    resampling=Resampling.bilinear
                )
                transformation = src.transform * src.transform.scale(src.width / w, src.height / h)
            c = upsampled.shape[0]
            utm_num = data["utm_num"]
            with rasterio.open(
                    data["path"],
                    "w",
                    crs=rasterio.CRS.from_string(f"EPSG:{utm_num + 2442}"),
                    driver="GTiff",
                    dtype=rasterio.uint8,
                    count=c,
                    width=w,
                    height=h,
                    transform=transformation
            ) as dst:
                for j in range(c):
                    dst.write(upsampled[j, :, :], indexes=j+1)
            self.data_list[i]["resolution"] = self.resolution
            coords, bounds = SatelliteImageData.get_gtif_coord(data["path"])
            self.data_list[i]["coords"] = coords
            self.data_list[i]["bounds"] = bounds

        self._output_data_file()

    def _output_data_file(self):
        if self.output_data_file is not None:
            dump_json(self.data_list, self.output_data_file)
    
    @staticmethod
    def get_rbg_from_number(num):
        """
        get the rgb value from the number
        """
        num = int(num)
        r = num // 65536
        g = (num - r * 65536) // 256
        b = num - r * 65536 - g * 256
        return r, g, b

    @staticmethod
    def get_number_from_rgb(r, g, b):
        """
        get the number from the rgb value
        """
        return r * 65536 + g * 256 + b
    
    @staticmethod
    def get_masked(image, mask, lid):
        # image: (C, H, W)
        # mask: (H, W) with lid at each pixel
        # lid: int
        mask = mask == lid
        x_idxs = np.where(mask.sum(axis=0) > 0)
        y_idxs = np.where(mask.sum(axis=1) > 0)
        return (image * mask)[np.ix_(list(range(image.shape[0])), x_idxs, y_idxs)]

    @staticmethod
    def to_gtif(image_path, x_min, x_max, y_min, y_max, out_path, utm_num):
        img = Image.open(image_path)
        w, h = img.size
        img = np.array(img, dtype=np.uint8)  # (H, W, 3) RGB or gray (H, W)
        c = img.shape[2] if len(img.shape) == 3 else 1
        tif_range = x_min, y_min, x_max, y_max
        transformation = rasterio.transform.from_bounds(*tif_range, w, h)
        with rasterio.open(
                out_path,
                "w",
                crs=rasterio.CRS.from_string(f"EPSG:{utm_num + 2442}"),
                driver="GTiff",
                dtype=rasterio.uint8,
                count=c,
                width=w,
                height=h,
                transform=transformation
        ) as dst:
            if c == 1:
                dst.write(img, indexes=1)
            else:
                for i in range(c):
                    dst.write(img[:, :, i], indexes=i+1)

    @staticmethod
    def get_gtif_coord(image_path):
        with rasterio.open(image_path) as src:
            bounds = src.bounds  # left, bottom, right, top
            bounds = list(bounds)
            coords = [(bounds[0], bounds[3]), (bounds[0], bounds[1]), (bounds[2], bounds[1]), (bounds[2], bounds[3])]   # north west, south west, south east, north east
        return coords, bounds

class OneHotImageData(SatelliteImageData):
    required_prop = ["name", "geo_path", "z", "x", "y", "utm_num"]
    def __init__(self, data_file, max_class_num=10, resolution=None, output_data_file=None, output_mask_file=None):
        # data_file: json [{"name": name, "geo_path": geo_path, "prop", "z":z, "x": [x_min, x_max], "y": [y_min, y_max], "utm_num": utm_num}]
        # (x_max and y_max are not included in model.)
        # path: png or geo file
        # initial transformation: coordinate calculation (coords, bounds), resolution calculation (resolution)
        # note: Image.open(path) returns RGB (H, W, 3)
        #       cv2.imread(path) returns BGR (H, W, 3)
        self.data_list = load_json(data_file)
        self.max_class_num = max_class_num
        self.resolution = resolution
        self.output_data_file = output_data_file
        self.output_mask_file = output_mask_file
        for data in self.data_list:
            for prop in OneHotImageData.required_prop:
                if prop not in data:
                    print(f"property {prop} is not in the model")
                    return

        self._get_xy()  # coords, bounds
        self._transform2image()  # tif, onehot
        self._clip_shared()  # replace path with clipped gtif, update (coords, bounds)
        self._set_resolution()  # resolution
        self._convert2onehot()

    def set_datafolder(self, data_dir, patch_size=256):
        # data_dir: path
        # data_dir - name - link_id.png
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for i, patches in enumerate(self.load_link_patches(patch_size=patch_size)):
            tmp_dir = os.path.join(data_dir, self.data_list[i]["name"])

            self.data_list[i]["data_dir"] = tmp_dir
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)  # [np.array((3, H, W), dtype=np.uin8)] RGB
            self.gis_segs[i].write_color_info()
            self.gis_segs[i].write_colormap(os.path.join(tmp_dir, "colormap.png"))
            self.gis_segs[i].write_hist(os.path.join(tmp_dir, "class_hist.png"))

            for j, patch in enumerate(patches):  # (3, H, W) RGB or (H, W) gray
                path_org = os.path.join(tmp_dir, f"{self.nw_data.lids_all[j]}_org.png")  # color is not sorted by pixel count
                if len(patch.shape) == 3:
                    patch = patch.transpose(1, 2, 0)  # (H, W, 3) RGB
                patch = Image.fromarray(patch)
                patch.save(path_org)

                path = os.path.join(tmp_dir, f"{self.nw_data.lids_all[j]}.png")
                path_onehot = None#os.path.join(tmp_dir, f"{self.nw_data.lids_all[j]}_onehot.npy")
                path_vis = os.path.join(tmp_dir, f"{self.nw_data.lids_all[j]}_vis.png")
                self.gis_segs[i].convert_file(path_org, png_file=path, np_file=path_onehot, vis_file=path_vis)

        self._output_data_file()

    def write_link_prop(self, data_dir, output_path):
        # columns: [LinkID, ...]
        df = pd.DataFrame({"LinkID": [lid for lid in self.nw_data.edges_all.keys()]})
        for i, data in enumerate(self.data_list):
            class_num = min(self.gis_segs[i].class_num, self.max_class_num)
            data_name = self.data_list[i]["name"]
            tmp_dir = os.path.join(data_dir, data_name)
            # name of the png files are link_ids
            # self.nw_data are registered in set_voronoi
            tmp_val = None
            for j, lid in enumerate(self.nw_data.lids_all):
                tmp_file = os.path.join(tmp_dir, f"{lid}.png")
                if os.path.exists(tmp_file):
                    ratio = OneHotImageData.get_prop_ratio(tmp_file, class_num)  # (C)
                    if tmp_val is None:
                        tmp_val = np.zeros((len(self.nw_data.lids_all), len(ratio)), dtype=np.float32)
                    tmp_val[j, :] = ratio
            if tmp_val is not None:
                df.loc[:, [f"{data_name}_{j}" for j in range(tmp_val.shape[1])]] = tmp_val
        df.to_csv(output_path, index=False)

    # inside functions
    def _transform2image(self):
        self.gis_segs = []
        for i, data in enumerate(self.data_list):
            gis_seg = GISSegmentation(data["geo_path"], data["prop"], data["utm_num"], data["z"], data["x"], data["y"], max_class_num=self.max_class_num, resolution=self.resolution)
            self.gis_segs.append(gis_seg)
            tif_path = gis_seg.raster_file
            self.data_list[i]["path"] = tif_path
            coords, bounds = SatelliteImageData.get_gtif_coord(tif_path)
            self.data_list[i]["coords"] = coords
            self.data_list[i]["bounds"] = bounds

        self._output_data_file()

    def _convert2onehot(self):
        for i, data in enumerate(self.data_list):
            tif_path = data["path"]
            onehot_path = os.path.splitext(data["path"])[0] + "_onehot.npy"
            self.gis_segs[i].convert_file(tif_path, np_file=onehot_path)
            self.data_list[i]["onehot"] = onehot_path

        self._output_data_file()

    @staticmethod
    def get_prop_ratio(onehot_path, class_num):
        _, ext = os.path.splitext(onehot_path)
        if ext == ".npy":
            onehot = np.load(onehot_path)  # (C, H, W)
            class_sum = onehot.sum((1, 2))
            total = class_sum.sum()
        elif ext == ".png":
            onehot = np.array(Image.open(onehot_path))
            class_sum = np.array([(onehot == i).sum() for i in range(class_num)])
            total = class_sum.sum()
        else:
            raise ValueError(f"extension {ext} is not supported.")
        return class_sum / (total + (total == 0))


