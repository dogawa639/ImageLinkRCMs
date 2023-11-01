from PIL import Image
import cv2
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from utility import load_json, dump_json, Coord

__all__ = ["ImageData"]

# 画像データの読み込み，座標付与
class ImageData:
    def __init__(self, data_file):
        # data_file: json [{"name": name, "path": path, "z":z, "x": [x_min, x_max], "y": [y_min, y_max], "utm_num": utm_num}]
        self.data_list = load_json(data_file)
        self._get_xy()
    
    def __len__(self):
        return len(self.data_list)

    def set_voronoi(self, network):
        # network: Network in network_processing.py
        self.network = network
        center_points = [v.cetner for v in network.edges.values()]
        nodes = [(v.x, v.y) for v in network.nodes.values()]

        # cv2 subdiv for each image
        for data in self.data_list:
            image = np.array(cv2.imread(data["path"]))  # BGR

            center_idxs_x = [int((x - data["x_coord"][0]) / (data["x_coord"][1] - data["x_coord"][0]) * image.shape[1]) for x,_ in center_points]
            center_idxs_y = [int((y - data["y_coord"][0]) / (data["y_coord"][1] - data["y_coord"][0]) * image.shape[0]) for _,y in center_points]

            subdiv = cv2.subdiv2D((0, 0, image.shape[1], image.shape[0]))
            for i in range(len(center_points)):
                if center_idxs_x[i] < 0 or center_idxs_x[i] >= image.shape[1] or center_idxs_y[i] < 0 or center_idxs_y[i] >= image.shape[0]:
                    continue
                subdiv.insert((center_idxs_x[i], image.shape[0] - center_idxs_y[i]))
            voronoi, _ = subdiv.getVoronoiFacetList([])

            # voronoi image
            imgv = np.zeros(image.shape, dtype=np.uint8)
            lids = list(network.edges.keys())
            for i, p in enumerate(f.astype(int) for f in voronoi):
                color = ImageData.get_rbg_from_number(lids[i])
                cv2.fillConvexPoly(imgv, p, color)
            voronoi_path = data["path"].replace(".png", "_voronoi.png")
            data["voronoi_path"] = voronoi_path
            cv2.imwrite(voronoi_path, imgv)

            # mask of convex hull
            node_idxs_x = [int((x - data["x_coord"][0]) / (data["x_coord"][1] - data["x_coord"][0]) * image.shape[1]) for x,_ in nodes]
            node_idxs_y = [int((y - data["y_coord"][0]) / (data["y_coord"][1] - data["y_coord"][0]) * image.shape[0]) for _,y in nodes]

            node_points = [[node_idxs_x[i], node_idxs_y[i]] for i in range(len(nodes)) if node_idxs_x[i] >= 0 and node_idxs_x[i] < image.shape[1] and node_idxs_y[i] >= 0 and node_idxs_y[i] < image.shape[0]]

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
        
    def load_link_patches(self):
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
            voronoi_lids = ImageData.get_number_from_rgb(voronoi[:,:,0], voronoi[:,:,1], voronoi[:,:,2]).astype(int)  # (H, W)
            del voronoi

            convex_hull_mask = np.array(Image.open(data["convex_hull_path"]))  # gray (H, W)
            voronoi_lids = voronoi_lids * (convex_hull_mask > 0) + (convex_hull_mask == 0) * -1
            del convex_hull_mask

            # link patches
            patches = []
            image = np.array(Image.open(data["path"])).transpose(0, 2)  # RGB (3, H, W)
            lids = list(self.network.edges.keys())
            for lid in lids:
                target_mask = (voronoi_lids == lid)
                x_idxs = np.where(target_mask.sum(axis=0) > 0)
                y_idxs = np.where(target_mask.sum(axis=1) > 0)
                patches.append(image[np.ix_(list(range(image.shape[0])), x_idxs, y_idxs)])
            del image

            yield patches
            data_idx += 1

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
            voronoi_lids = ImageData.get_number_from_rgb(voronoi[:,:,0], voronoi[:,:,1], voronoi[:,:,2]).astype(int)  # (H, W)
            del voronoi

            convex_hull_mask = np.array(Image.open(data["convex_hull_path"]))  # gray (H, W)
            voronoi_lids = voronoi_lids * (convex_hull_mask > 0) + (convex_hull_mask == 0) * -1
            del convex_hull_mask

            yield voronoi_lids
            data_idx += 1

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
    

# test
if __name__ == "__main__":
    from preprocessing.network_processing import *
    device = "mps"
    node_path = '/Users/dogawa/Desktop/bus/estimation/data/node.csv'
    link_path = '/Users/dogawa/Desktop/bus/estimation/data/link.csv'
    link_prop_path = '/Users/dogawa/Desktop/bus/estimation/data/link_attr_min.csv'

    image_data = ImageData("../data/image_data.json")
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    image_data.set_voronoi(nw_data)
    for patches in image_data.load_link_patches():
        print(patches[0].shape)
        break


