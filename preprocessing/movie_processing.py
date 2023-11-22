import torch
import torchvision
from torch import tensor
from torchvision.utils import draw_bounding_boxes
from torchvision.models import detection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import cv2


class YoLoV5Detect:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def __init__(self, batch_size=32, device="cpu"):
        self.model = self.model.to(device)
        self.batch_size = batch_size
        self.device = device

    def detect(self, movie_path, df_path, idx, out_mov_path=None, threshold=0.5):
        # idx  0: "person"
        cap = cv2.VideoCapture(movie_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if out_mov_path is not None:
            out = cv2.VideoWriter(out_mov_path, fourcc, fps, (width, height))

        if cap.isOpened():
            x = []
            objs = [[] for _ in range(frame_cnt)]
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
                                with_bb = draw_bounding_boxes(tensor(x[j].transpose(2, 0, 1)), xyxyn[j][target_idx, 0:4])
                                out.write(with_bb.detach().cpu().numpy().transpose(1, 2, 0))
                        x = []
                else:
                    break
        cap.release()
        if out_mov_path is not None:
            out.release()
        array = np.array([[i]+objs[i][j].tolist() for i in range(len(objs)) for j in range(len(objs[i])) if objs[i][j][4] > threshold])
        objs = pd.DataFrame(array, columns=["frame_num", "x1", "y1", "x2", "y2", "conf"])
        objs.to_csv(df_path, index=False)

    def mapping2xy(self, img_points, xy_points):


# test
if __name__ == "__main__":
    detection = YoLoV5Detect()
    detection.detect("/Users/dogawa/Desktop/bus/カメラ調査/調査地点①/220928/20220928-224129MA.mp4", "/Users/dogawa/PycharmProjects/GANs/debug/data/peope.csv", 0, out_mov_path="/Users/dogawa/PycharmProjects/GANs/debug/data/people_bb.mp4")