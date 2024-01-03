import torch
import torchvision
from torch import tensor
from torchvision.utils import draw_bounding_boxes
from torchvision.models import detection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import datetime
import cv2


class YoLoV5Detect:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def __init__(self, start_time=None, img_points=None, xy_points=None, batch_size=32, device="cpu"):
        # img_points, xy_points: ndarray(4, 2)
        # img_points: in [0,1]
        self.start_time = start_time
        self.img_points = img_points
        self.xy_points = xy_points
        self.model = self.model.to(device)
        self.batch_size = batch_size
        self.device = device

        self.fps = 25.0

    def detect(self, movie_path, idx, df_path=None, out_mov_path=None, threshold=0.5):
        # idx  0: "person"
        cap = cv2.VideoCapture(movie_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        if out_mov_path is not None:
            out = cv2.VideoWriter(out_mov_path, fourcc, fps, (width, height))

        if cap.isOpened():
            x = []
            objs = [[] for _ in range(frame_cnt)]
            expansion = tensor([[width, height, width, height]]).to(self.device)
            for i in range(frame_cnt):
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.uint8)
                    x.append(frame)
                    if i % self.batch_size == self.batch_size - 1 or i == frame_cnt - 1:
                        self.model.eval()
                        results = self.model(x)
                        xyxyn = results.xyxyn  # (batch_size, detect_num, 6) 6: (x1, y1, x2, y2, conf, class)
                        bs = len(xyxyn)
                        for j in range(bs):
                            target_idx = (xyxyn[j][:, 5] == idx) & (xyxyn[j][:, 4] > threshold)
                            objs[i - bs + j + 1] = xyxyn[j][target_idx, 0:5]
                            if out_mov_path is not None:
                                with_bb = draw_bounding_boxes(tensor(x[j].transpose(2, 0, 1)),
                                                              xyxyn[j][target_idx, 0:4] * expansion,
                                                              colors=(255, 0, 0),
                                                              width=3)
                                out.write(with_bb.detach().cpu().numpy().transpose(1, 2, 0))  # (h, w, c)
                        x = []
                else:
                    break
        cap.release()
        if out_mov_path is not None:
            out.release()
        array = np.array([[i]+objs[i][j].tolist() for i in range(len(objs)) for j in range(len(objs[i])) if objs[i][j][4] > threshold])
        objs = pd.DataFrame(array, columns=["frame_num", "x1", "y1", "x2", "y2", "conf"])
        if df_path is not None:
            objs.to_csv(df_path, index=False)

    def mapping2xy(self, obj_df, df_path=None, out_mov_path=None, resolution=0.5, img_points=None, xy_points=None):
        # obj_df : ["frame_num", "x1", "y1", "x2", "y2", "conf"]
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

        # foot position calculation
        obj_df["foot_x"] = obj_df[["x1", "x2"]].mean(axis=1)
        obj_df["foot_y"] = obj_df["y2"]
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

    def write_detection_result_movie(self, movie_path, obj_df, out_mov_path):
        if type(obj_df) is str:
            obj_df = pd.read_csv(obj_df)
        if "2d_img_x" not in obj_df.columns or "2d_img_y" not in obj_df.columns:
            raise ValueError("calculate mapping2xy in advance")
        cap = cv2.VideoCapture(movie_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        dx = obj_df["2d_img_x"].max() - obj_df["2d_img_x"].min()
        dy = obj_df["2d_img_y"].max() - obj_df["2d_img_y"].min()
        size = (width + int(dx * 5) + 1, max(height, int(dy * 5) + 1))  # left: original perspective, right: top view

        val = obj_df[["frame_num", "x1", "y1", "x2", "y2", "2d_img_x", "2d_img_y"]].values
        out = cv2.VideoWriter(out_mov_path, fourcc, fps, size)
        bbs = [[] for _ in range(frame_cnt)]
        pts = [[] for _ in range(frame_cnt)]  # [[(2d_img_x, 2d_img_y)]]
        for i in range(len(val)):
            f_num = int(val[i, 0])
            bb = val[i, 1:5].tolist()
            xy_img_x = val[i, 5] * 5
            xy_img_y = val[i, 6] * 5
            bbs[f_num].append(bb)
            pts[f_num].append((xy_img_x, xy_img_y))
        expansion = tensor([[width, height, width, height]])
        if not cap.isOpened():
            print("invalid path of original movie.")
            return
        for i in range(frame_cnt):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.uint8)
            else:
                continue
            img = np.zeros((*size[::-1], 3), dtype=np.uint8)
            img[:height, :width, :] = frame
            if len(bbs[i]) > 0:
                img = draw_bounding_boxes(tensor(img.transpose(2, 0, 1)),
                                          tensor(bbs[i]) * expansion,
                                          colors=(255, 0, 0),
                                          width=3)
                img = img.detach().cpu().numpy().transpose(1, 2, 0)
            for x, y in pts[i]:
                cv2.drawMarker(img, (width + int(x), int(y)), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
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


# test
if __name__ == "__main__":
    start_time = datetime.datetime.strptime("2022-09-28 22:41:29", "%Y-%m-%d %H:%M:%S")
    img_points = np.array([[700,520],[1530,670],[730,930],[1560,410]], dtype=np.float32) / np.array([[1920, 1080]], dtype=np.float32)
    xy_points = np.array([[-69266.30906479518,93476.6843737138], [-69260.76750148823,93480.99447847917],[-69260.67954013556,93476.06864445977],[-69268.68402048285,93492.07760501398]], dtype=np.float32)
    detection = YoLoV5Detect(start_time=start_time, img_points=img_points, xy_points=xy_points, batch_size=32, device="mps")
    detection.mapping2xy("/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/peope.csv", "/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/peope_2d.csv", "/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/peope_topview.mp4", resolution=0.1)
    #detection.get_clip("/Users/dogawa/Desktop/bus/カメラ調査/調査地点①/220928/20220928-224129MA.mp4", "/Users/dogawa/PycharmProjects/ImageLinkRCM/model/mov/cap_1.png", 0)
    #detection.detect("/Users/dogawa/Desktop/bus/カメラ調査/調査地点①/220928/20220928-224129MA.mp4", 0)
    detection = YoLoV5Detect()
    detection.write_detection_result_movie("/Users/dogawa/Desktop/bus/カメラ調査/調査地点①/220928/20220928-224129MA.mp4", "/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/peope_2d.csv", "/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/people_2d_total.mp4")