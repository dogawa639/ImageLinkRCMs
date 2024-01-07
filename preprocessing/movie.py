import torch
import torchvision
from torch import tensor
from torchvision.utils import draw_bounding_boxes
from torchvision.models import detection
from ultralytics import YOLO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import json
import datetime
import cv2
from utility import *

__all__ = ["YoLoV5Detect", "ByteTrack"]


class YoLoV5Detect:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
    #model = YOLO('yolov8n.pt')

    def __init__(self, start_time=None, img_points=None, xy_points=None, batch_size=1, device="cpu"):
        # img_points, xy_points: ndarray(4, 2)
        # img_points: in [0,1]
        self.start_time = start_time
        self.img_points = img_points
        self.xy_points = xy_points
        self.model = self.model.to(device)
        self.batch_size = batch_size
        self.device = device

        self.fps = 25.0

    def detect(self, movie_path, idxs, out_csv_dir=None, out_mov_path=None, colors=None, threshold=0.1):
        # idx  0: "person", 1: bicycle, 2: car, 3: motorbike, 5: bus, 7: truck
        cap = cv2.VideoCapture(movie_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        cmap = plt.get_cmap("hsv")
        colors = [tuple((np.array(cmap(i/len(idxs))[2::-1]) * 255).astype(int)) for i in range(len(idxs))] if colors is None else colors # [(r, g, b)]
        if out_mov_path is not None:
            out = cv2.VideoWriter(out_mov_path, fourcc, fps, (width, height))

        if cap.isOpened():
            x = []
            objs = [[[] for _ in range(frame_cnt)] for _ in range(len(idxs))]  # [[(x1, y1, x2, y2, conf)]]
            expansion = tensor([[width, height, width, height]])
            for i in range(frame_cnt):
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.uint8)
                    x.append(frame)
                    if i % self.batch_size == self.batch_size - 1 or i == frame_cnt - 1:
                        self.model.eval()
                        try:
                            results = self.model(x)
                        except RuntimeError:
                            print(f"RuntimeError {i}")
                            break
                        xyxyn = [xyxy.detach().cpu() for xyxy in results.xyxyn]  # (batch_size, detect_num, 6) 6: (x1, y1, x2, y2, conf, class)
                        bs = len(xyxyn)
                        for j in range(bs):
                            for k, idx in enumerate(idxs):
                                xyxynj = xyxyn[j]
                                xyxynj = xyxynj[(xyxynj[:, 2] > xyxynj[:, 0]) & (xyxynj[:, 3] > xyxynj[:, 1])]

                                target_idx = (xyxynj[:, 5] == idx) & (xyxynj[:, 4] > threshold)
                                objs[k][i - bs + j + 1] = xyxynj[target_idx, 0:5]
                                if out_mov_path is not None:
                                    with_bb = draw_bounding_boxes(tensor(x[j].transpose(2, 0, 1)),
                                                                  xyxynj[target_idx, 0:4] * expansion,
                                                                  colors=colors[k],
                                                                  width=3)
                                    out.write(with_bb.detach().cpu().numpy().transpose(1, 2, 0))  # (h, w, c)
                        x = []
                else:
                    break
        cap.release()
        if out_mov_path is not None:
            out.release()
        arrays = [np.array([[i]+objs[k][i][j].tolist() for i in range(len(objs[k])) for j in range(len(objs[k][i])) if objs[k][i][j][4] > threshold]) for k in range(len(objs))]
        objs = [pd.DataFrame(array, columns=["frame_num", "x1", "y1", "x2", "y2", "conf"]) for array in arrays]
        for i in range(len(objs)):
            # foot position calculation
            objs[i]["foot_x"] = objs[i][["x1", "x2"]].mean(axis=1)
            objs[i]["foot_y"] = objs[i]["y2"]
        if out_csv_dir is not None:
            for i, idx in enumerate(idxs):
                objs[i].to_csv(os.path.join(out_csv_dir, f"yolov_out_{idx}.csv"), index=False)

    def mapping2xy(self, obj_df, df_path=None, out_mov_path=None, resolution=0.5, img_points=None, xy_points=None):
        # obj_df : ["frame_num", "x1", "y1", "x2", "y2", "conf", "foot_x", "foot_y"]
        # img_points, xy_points: ndarray(4, 2)
        # img_points: in [0,1]
        # resolution: m per pixel
        if self.img_points is None:
            self.img_points = img_points
        if self.xy_points is None:
            self.xy_points = xy_points
        if self.img_points is None or self.xy_points is None:
            raise ValueError("img_points and xy_points must be set.")
        if type(obj_df) is str:
            obj_df = pd.read_csv(obj_df)
        frame_num = int(obj_df["frame_num"].max()) + 1
        m = cv2.getPerspectiveTransform(self.img_points, self.xy_points)

        img_pos = obj_df[["foot_x", "foot_y"]].values.T  # (2, *)
        img_pos = np.concatenate([img_pos, np.ones((1, img_pos.shape[1]), dtype=img_pos.dtype)], axis=0)  # (3, *)
        # 2d position calculation
        xy_pos = np.matmul(m, img_pos)
        xy_pos = xy_pos[:2, :] / xy_pos[[2], :]  # (2, *)
        min_x, max_x = xy_pos[0, :].min(), xy_pos[0, :].max()
        min_y, max_y = xy_pos[1, :].min(), xy_pos[1, :].max()
        dx, dy = max_x - min_x, max_y - min_y
        size = (int(dx / resolution) + 1, int(dy / resolution) + 1)

        obj_df["2d_x"] = xy_pos[0, :]
        obj_df["2d_y"] = xy_pos[1, :]
        # top view img position calculation
        xy_img_pos = (xy_pos - np.array([[min_x], [min_y]])) / resolution  # (2, *)
        obj_df["2d_img_x"] = xy_img_pos[0, :]
        obj_df["2d_img_y"] = size[1] - xy_img_pos[1, :]
        # time calculation
        if self.start_time is not None:
            obj_df["time"] = [datetime.timedelta(seconds=obj_df["frame_num"][i] / self.fps) + self.start_time for i in range(len(obj_df))]

        if df_path is not None:
            obj_df.to_csv(df_path, index=False)  # ["frame_num", "x1", "y1", "x2", "y2", "conf", "foot_x", "foot_y", "2d_x", "2d_y", "2d_img_x", "2d_img_y", "time"]

        if out_mov_path is not None:
            val = obj_df[["frame_num", "2d_img_x", "2d_img_y"]].values
            pts = [[] for _ in range(frame_num)]  # [[(2d_img_x, 2d_img_y)]]
            for i in range(len(val)):
                f_num = int(val[i, 0])
                xy_img_x = val[i, 1]
                xy_img_y = val[i, 2]
                pts[f_num].append((xy_img_x, xy_img_y))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_mov_path, fourcc, self.fps, size)
            for i in range(frame_num):
                img = np.zeros((*size[::-1], 3), dtype=np.uint8)
                for x, y in pts[i]:
                    cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=2)
                out.write(img)
            out.release()
        return min_x, min_y

    def write_detection_result_movie(self, movie_path, obj_dfs, out_mov_path, colors=None):
        if type(obj_dfs[0]) is str:
            obj_dfs = [pd.read_csv(obj_df) for obj_df in obj_dfs]
        for obj_df in obj_dfs:
            if "2d_img_x" not in obj_df.columns or "2d_img_y" not in obj_df.columns:
                raise ValueError("calculate mapping2xy in advance")
        cmap = plt.get_cmap("hsv")
        colors = [tuple((np.array(cmap(i / len(obj_dfs))[2::-1]) * 255).astype(float)) for i in
                  range(len(obj_dfs))] if colors is None else colors  # [(r, g, b)]

        cap = cv2.VideoCapture(movie_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        x_min, y_min, x_max, y_max = np.infty, np.infty, -np.infty, -np.infty
        for obj_df in obj_dfs:
            x_min = min(x_min, obj_df["2d_x"].min())
            y_min = min(y_min, obj_df["2d_y"].min())
            x_max = max(x_max, obj_df["2d_x"].max())
            y_max = max(y_max, obj_df["2d_y"].max())
        dx = x_max - x_min
        dy = y_max - y_min
        size = (width + int(dx * 20) + 1, max(height, int(dy * 20) + 1))  # left: original perspective, right: top view

        bbs = [[[] for _ in range(frame_cnt)] for _ in range(len(obj_dfs))]  # [[[(x1, y1, x2, y2)]]]
        pts = [[[] for _ in range(frame_cnt)] for _ in range(len(obj_dfs))]  # [[[(2d_img_x, 2d_img_y)]]]
        for j, obj_df in enumerate(obj_dfs):
            val = obj_df[["frame_num", "x1", "y1", "x2", "y2", "2d_x", "2d_y"]].values
            for i in range(len(val)):
                f_num = int(val[i, 0])
                bb = val[i, 1:5].tolist()
                xy_img_x = (val[i, 5] - x_min) * 20
                xy_img_y = (val[i, 6] - y_min) * 20
                bbs[j][f_num].append(bb)
                pts[j][f_num].append((xy_img_x, xy_img_y))

        expansion = tensor([[width, height, width, height]])
        if not cap.isOpened():
            print("invalid path of original movie.")
            return
        out = cv2.VideoWriter(out_mov_path, fourcc, fps, size)
        for i in range(frame_cnt):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)
            else:
                continue
            img = np.ones((*size[::-1], 3), dtype=np.uint8) * 255
            img[:min(height, int(dy * 20) + 1), width:, :] = 0
            img[:height, :width, :] = frame
            for j in range(len(obj_dfs)):
                color = tuple(np.array(colors[j], dtype=np.uint8))
                if len(bbs[j][i]) > 0:
                    img = draw_bounding_boxes(tensor(img.transpose(2, 0, 1)),
                                              tensor(bbs[j][i]) * expansion,
                                              colors=color,
                                              width=3)
                    img = img.detach().cpu().numpy().transpose(1, 2, 0)
                for x, y in pts[j][i]:
                    cv2.drawMarker(img, (width + int(x), int(y)), colors[j], markerType=cv2.MARKER_CROSS, markerSize=32, thickness=6)
            out.write(img)
        cap.release()
        out.release()

    @staticmethod
    def get_clip(mov_path, png_path, frame_num):
        cap = cv2.VideoCapture(mov_path)
        for i in range(frame_num + 1):
            ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(png_path, frame)
        cap.release()


class ByteTrack:
    def __init__(self, dt, detect_thresh=0.1, iou_thresh=0.1, time_lost=1./ 25.):
        # dt: sec
        self.dt = dt
        self.detect_thresh = detect_thresh
        self.iou_thresh = iou_thresh
        self.time_lost = time_lost

        # Kalman Filter
        # state : [x, y, aspect, height, vx, vy, v_aspect, v_height]
        f = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                      [0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])
        h = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0]])

        def q_fn(x):  # process noise
            return np.diag(np.square([0.05 * x[3], 0.05 * x[3], 0.01, 0.08 * x[3], 0.05 * x[3], 0.05 * x[3], 0.1, 0.05 * x[3]]))

        def r_fn(z):  # observation noise
            return np.diag(np.square([0.5 * z[3], 0.5 * z[3], 0.2, 0.8 * z[3]]))

        self.kalman_filter = KalmanFilter(f, h, q_fn, r_fn)
        self.hungarian = Hungarian()


    def track(self, obj_df, out_path=None):
        # obj_df : ["frame_num", "x1", "y1", "x2", "y2", "conf", "foot_x", "foot_y"]
        obj_df = obj_df if type(obj_df) is pd.DataFrame else pd.read_csv(obj_df)
        aspect = (obj_df["x2"] - obj_df["x1"]) / (obj_df["y2"] - obj_df["y1"])
        obj_df = obj_df[(aspect < 1.0) & (aspect > 0.1)]

        frame_num = int(obj_df["frame_num"].max()) + 1
        unused_count = []

        traj_total = []  # [["frame_num", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "p_foot_x", "p_foot_y", "id"]]
        for i in range(frame_num):
            unused_count = [count + 1 for count in unused_count]
            target = obj_df.loc[obj_df["frame_num"] == i, :]
            if len(target) == 0:
                continue
            self.kalman_filter.predict()

            # divide into 2 sets
            target_high = target.loc[target["conf"] > self.detect_thresh, :]
            target_low = target.loc[target["conf"] <= self.detect_thresh, :]

            z = np.zeros((len(self.kalman_filter), 4), dtype=np.float32)  # z for kalman filter update
            z_unmatched = []  # unmatched high confidence set
            matched_traj = []
            matched_obj = []
            matched_traj_idx = [False] * len(self.kalman_filter)  # matched trajectory index
            # first matching: high confidence set - trajectory
            if len(target_high) > 0:
                matched = []
                coords_high = target_high[["x1", "y1", "x2", "y2"]].values
                if len(self.kalman_filter) > 0 and sum(self.kalman_filter.active_idx) > 0:
                    states = self.kalman_filter.xnn[self.kalman_filter.active_idx, :]
                    ious = np.array([[self.iou(state, coord) for coord in coords_high] for state in states])  # (traj_num, obj_num)
                    matched = self.hungarian.compute(1 - ious)

                for j in range(len(matched)):
                    if ious[*matched[j]] >= self.iou_thresh:
                        traj_idx = np.arange(len(self.kalman_filter))[self.kalman_filter.active_idx][matched[j][0]]
                        matched_traj.append(traj_idx)
                        matched_obj.append(matched[j][1])

                        unused_count[traj_idx] = 0
                        matched_traj_idx[traj_idx] = True
                for j in range(len(coords_high)):
                    if j in matched_obj:
                        z[matched_traj[matched_obj.index(j)]] = ByteTrack.get_state(coords_high[j])
                    else:
                        z_unmatched.append(ByteTrack.get_state(coords_high[j]))

            # second matching: low confidence set - trajectory
            if len(target_low) > 0 and (len(self.kalman_filter) > 0 and sum(self.kalman_filter.active_idx) > 0):
                coords_low = target_low[["x1", "y1", "x2", "y2"]].values
                active_idx = self.kalman_filter.active_idx.copy()
                active_idx[matched_traj] = False
                states = self.kalman_filter.xnn[active_idx, :]  # traj_remain
                ious = np.array([[self.iou(state, coord) for coord in coords_low] for state in states])  # (traj_num, obj_num)
                matched = self.hungarian.compute(1 - ious)

                matched_traj = []
                matched_obj = []
                for j in range(len(matched)):
                    if ious[*matched[j]] >= self.iou_thresh:
                        traj_idx = np.arange(len(self.kalman_filter))[active_idx][matched[j][0]]
                        matched_traj.append(traj_idx)
                        matched_obj.append(matched[j][1])

                        unused_count[traj_idx] = 0
                        matched_traj_idx[traj_idx] = True
                for j in range(len(coords_low)):
                    if j in matched_obj:
                        z[matched_traj[matched_obj.index(j)]] = ByteTrack.get_state(coords_low[j])

            if len(self.kalman_filter) > 0:
                tmp_active_idx = self.kalman_filter.active_idx.copy()
                self.kalman_filter.active_idx = np.array(matched_traj_idx)
                self.kalman_filter.correct(z)
                self.kalman_filter.active_idx = tmp_active_idx & (np.array(unused_count) < self.time_lost / self.dt)
            self.kalman_filter.add_obj(z_unmatched)
            unused_count += [0] * len(z_unmatched)

            for j in range(len(self.kalman_filter)):
                if self.kalman_filter.active_idx[j]:
                    x, y, aspect, height, vx, vy, v_aspect, v_height = self.kalman_filter.xnn[j, :]
                    px, py, paspect, pheight, pvx, pvy, pv_aspect, pv_height = np.diag(self.kalman_filter.pnn[j, :])
                    width = aspect * height
                    x1, y1, x2, y2 = x - width / 2, y - height, x + width / 2, y
                    if height < 0 or aspect < 0 or min([x1, y1, x2, y2]) < 0 or max([x1, y1, x2, y2]) > 1:
                        continue
                    traj_total.append([i, x1, y1, x2, y2, x, y, px, py, j+1])

        traj_total = pd.DataFrame(traj_total, columns=["frame_num", "x1", "y1", "x2", "y2", "foot_x", "foot_y", "p_foot_x", "p_foot_y", "id"])
        if out_path is not None:
            traj_total.to_csv(out_path, index=False)
        return traj_total

    @staticmethod
    def iou(state, coords):
        # state : [foot_x, foot_y, aspect, height, ...]
        # coords : [x1, y1, x2, y2]
        width = state[2] * state[3]
        area1 = width * state[3]
        x1, y1, x2, y2 = coords
        area2 = (x2 - x1) * (y2 - y1)

        width_shared = max(0, min(x2, state[0] + width / 2) - max(x1, state[0] - width / 2))
        height_shared = max(0, min(y2, state[1] + state[3]) - max(y1, state[1] - state[3]))
        area_shared = width_shared * height_shared
        return area_shared / max(1e-9, area1 + area2 - area_shared)

    @staticmethod
    def get_state(coords):
        # coords : [x1, y1, x2, y2]
        width = (coords[2] - coords[0])
        height = (coords[3] - coords[1])
        return np.array([(coords[0] + coords[2]) / 2, coords[3], width / height, height])




