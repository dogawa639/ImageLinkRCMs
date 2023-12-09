from PIL import Image
import cv2
from osgeo import gdal
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os

from utility import load_json, dump_json, Coord
from preprocessing.dataset import PatchDataset
from preprocessing.geo_util import tile2latlon
__all__ = ["SatelliteImageData"]


# 画像データの読み込み，座標付与
class SatelliteImageData:
    required_prop = ["name", "path", "z", "x", "y", "utm_num"]
    def __init__(self, data_file):
        # data_file: json [{"name": name, "path": path, "z":z, "x": [x_min, x_max], "y": [y_min, y_max], "utm_num": utm_num}]
        # initial transformation: coordinate calculation (x_coord, y_coord), resolution calculation (resolution)
        # visualization: gtif
        # note: Image.open(path) returns RGB (H, W, 3)
        #       cv2.imread(path) returns BGR (H, W, 3)
        self.data_list = load_json(data_file)
        for data in self.data_list:
            for prop in SatelliteImageData.required_prop:
                if prop not in data:
                    print(f"property {prop} is not in the data")
                    return

        self._get_xy()
        self._get_resolution()
    
    def __len__(self):
        return len(self.data_list)

    def set_voronoi(self, nw_data):
        # network: Network in network.py
        self.nw_data = nw_data
        center_points = [nw_data.edges[v[0]].center for v in nw_data.undir_edges.values()]
        nodes = [(v.x, v.y) for v in nw_data.nodes.values()]

        # cv2 subdiv for each image
        for data in self.data_list:
            image = np.array(Image.open(data["path"]))  # RGB (H, W, 3)

            center_idxs_x = [int((x - data["x_coord"][0]) / (data["x_coord"][1] - data["x_coord"][0]) * image.shape[1]) for x,_ in center_points]
            center_idxs_y = [int((y - data["y_coord"][0]) / (data["y_coord"][1] - data["y_coord"][0]) * image.shape[0]) for _,y in center_points]

            subdiv = cv2.Subdiv2D((0, 0, image.shape[1], image.shape[0]))
            for i in range(len(center_points)):
                if center_idxs_x[i] < 0 or center_idxs_x[i] >= image.shape[1] or center_idxs_y[i] < 0 or center_idxs_y[i] >= image.shape[0]:
                    continue
                subdiv.insert((center_idxs_x[i], image.shape[0] - center_idxs_y[i]))
            voronoi, _ = subdiv.getVoronoiFacetList([])

            # voronoi image
            imgv = np.zeros(image.shape, dtype=np.uint8)
            undir_ids = list(nw_data.undir_edges.keys())
            for i, p in enumerate(f.astype(int) for f in voronoi):
                color = SatelliteImageData.get_rbg_from_number(undir_ids[i])  # rgb
                color = (color[2], color[1], color[0])  # bgr
                cv2.fillConvexPoly(imgv, p, color)
            voronoi_path = data["path"].replace(".png", "_voronoi.png")
            data["voronoi_path"] = voronoi_path
            cv2.imwrite(voronoi_path, imgv)

            # mask of convex hull
            node_idxs_x = [int((x - data["x_coord"][0]) / (data["x_coord"][1] - data["x_coord"][0]) * image.shape[1]) for x,_ in nodes]
            node_idxs_y = [int((y - data["y_coord"][0]) / (data["y_coord"][1] - data["y_coord"][0]) * image.shape[0]) for _,y in nodes]

            node_points = [[node_idxs_x[i], image.shape[0] - node_idxs_y[i]] for i in range(len(nodes)) if node_idxs_x[i] >= 0 and node_idxs_x[i] < image.shape[1] and node_idxs_y[i] >= 0 and node_idxs_y[i] < image.shape[0]]

            convex_hull = cv2.convexHull(np.array(node_points))
            convex_hull_mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.fillConvexPoly(convex_hull_mask, convex_hull, (255, 255, 255))
            convex_hull_mask = cv2.cvtColor(convex_hull_mask, cv2.COLOR_BGR2GRAY)
            convex_hull_path = data["path"].replace(".png", "_convex_hull.png")
            data["convex_hull_path"] = convex_hull_path
            cv2.imwrite(convex_hull_path, convex_hull_mask)

    def load_images(self):
        # for i in range(len(image_data)):
        #    image = image_data.load_images()
        data_idx = 0
        while True:
            if data_idx >= len(self.data_list):
                break
            data = self.data_list[data_idx]
            image = np.array(Image.open(data["path"]), dtype=np.uint8).transpose(0, 2)  # RGB (3, H, W)
            yield image
            data_idx += 1

    # call after set_voronoi
    def load_link_patches(self, patch_size=256):
        # 直前に呼び出したset_voronoiに対応する．
        # for i in range(len(image_data)):
        #    patches = image_data.load_link_patches()
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
            voronoi_lids = voronoi_lids * (convex_hull_mask > 0) + (convex_hull_mask == 0) * -1
            del convex_hull_mask

            # link patches
            patches = []
            image = np.array(Image.open(data["path"]))
            image = image.transpose(2, 0, 1)  # RGB (3, H, W)
            for edge in self.nw_data.edges.values():
                patch = np.zeros((3, patch_size, patch_size), dtype=np.uint8)
                target_mask = (voronoi_lids == edge.undir_id)
                x_idxs = np.where(target_mask.sum(axis=0) > 0)[0]
                y_idxs = np.where(target_mask.sum(axis=1) > 0)[0]
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
            path = self.data_list[i]["path"].replace(".png", "_compressed.npy")
            self.data_list[i]["compressed_path"] = path
            np.save(path, compressed)

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

    def set_datafolder(self, data_dir, patch_size=256):
        # data_dir: path
        # data_dir - name - link_idx.png
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        for i, patches in enumerate(self.load_link_patches(patch_size=patch_size)):
            tmp_dir = os.path.join(data_dir, self.data_list[i]["name"])
            self.data_list[i]["data_dir"] = tmp_dir
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)  # [np.array((3, H, W), dtype=np.uin8)] RGB
            for j, patch in enumerate(patches):
                path = os.path.join(tmp_dir, f"{j}.png")
                patch = patch.transpose(1, 2, 0)  # (H, W, 3) RGB
                patch = Image.fromarray(patch)
                patch.save(path)

    # visualization
    def write_gtif(self):
        for i in range(len(self.data_list)):
            gtif_path = self.data_list[i]["path"].replace(".png", ".tif")
            x_min, x_max = self.data_list[i]["x_coord"]
            y_min, y_max = self.data_list[i]["y_coord"]
            SatelliteImageData.to_gtif(self.data_list[i]["path"], x_min, x_max, y_min, y_max, gtif_path, self.data_list[i]["utm_num"])

    # internal functions
    def _get_xy(self):
        # 座標取得
        for i in range(len(self.data_list)):
            coord = Coord(self.data_list[i]["utm_num"])
            x_min, y_max = tile2latlon(self.data_list[i]["x"][0], self.data_list[i]["y"][0], self.data_list[i]["z"])
            x_max, y_min = tile2latlon(self.data_list[i]["x"][1], self.data_list[i]["y"][1], self.data_list[i]["z"])
            x_min, y_min = coord.to_utm(x_min, y_min)
            x_max, y_max = coord.to_utm(x_max, y_max)
            self.data_list[i]["x_coord"] = [x_min, x_max]
            self.data_list[i]["y_coord"] = [y_min, y_max]

    def _get_resolution(self):
        for i in range(len(self.data_list)):
            img = Image.open(self.data_list[i]["path"])
            w, h = img.size
            dx = self.data_list[i]["x_coord"][1] - self.data_list[i]["x_coord"][0]
            dy = self.data_list[i]["y_coord"][1] - self.data_list[i]["y_coord"][0]
            self.data_list[i]["resolution"] = [dx / w, dy / h]  # m/pixel

    @property
    def range(self):
        whole_range = [np.inf, -np.inf, np.inf, -np.inf]  # [x_min, x_max, y_min, y_max]
        for i in range(len(self.data_list)):
            if whole_range[0] > self.data_list[i]["x_coord"][0]:
                whole_range[0] = self.data_list[i]["x_coord"][0]
            if whole_range[1] < self.data_list[i]["x_coord"][1]:
                whole_range[1] = self.data_list[i]["x_coord"][1]
            if whole_range[2] > self.data_list[i]["y_coord"][0]:
                whole_range[2] = self.data_list[i]["y_coord"][0]
            if whole_range[3] < self.data_list[i]["y_coord"][1]:
                whole_range[3] = self.data_list[i]["y_coord"][1]
        return whole_range
    
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
        gdal.Translate(out_path, image_path, format="GTiff",
                       outputSRS=f"EPSG:{2442+utm_num}", bandList=list(range(1, 4)), outputBounds=[x_min, y_max, x_max, y_min])





