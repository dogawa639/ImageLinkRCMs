import datetime
import time
import os

import numpy as np
import matplotlib.pyplot as plt

from utility import *


class Logger:
    def __init__(self, json_file, conf_file, fig_file=None, figsize=(6.4, 4.8)):
        self.json_file = json_file
        self.conf_file = conf_file
        self.fig_file = fig_file

        self.start_time = datetime.datetime.now()
        self.data = {
                "start_time": str(self.start_time),
                "config": conf_file
        }

        self.fig = plt.figure(tight_layout=True, figsize=figsize)
        self.closed = False

    def close(self):
        self.end_time = datetime.datetime.now()
        self.data["end_time"] = str(self.end_time)
        self.data["duration"] = str(self.end_time - self.start_time)

        dump_json(self.data, self.json_file)
        self.show_fig()
        input("Logger has been closed. Press Enter to continue...")
        self.closed = True

    def add_log(self, key, value):
        # value: float,int or 1d list of float,int
        if self.closed:
            raise Exception("Logger has already been closed")
        if type(value) == np.ndarray:
            value = value.astype(float).tolist()
        if type(value) == np.float32 or type(value) == np.float64 or type(value) == np.int32 or type(value) == np.int64:
            value = float(value)
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]

        dump_json(self.data, self.json_file)

        v_list = []
        label_list = []
        for k, v in self.data.items():
            if type(v) != list:
                continue
            try:
                if type(v[0]) == float or type(v[0]) == int or type(v[0]) == np.float64 or type(
                        v[0]) == np.int64 or type(v[0]) == np.float32 or type(v[0]) == np.int32:
                    v_list.append(v)
                    label_list.append(str(k))
                elif type(v[0]) == list:
                    v_list.append(np.array(v))
                    label_list.append([f"{k}_{i}" for i in range(len(v[0]))])
            except:
                pass

        prop_num = len(v_list)
        if prop_num == 0:
            return

        self.fig.clf()
        for i in range(prop_num):
            ax = self.fig.add_subplot(prop_num, 1, i + 1)
            if len(v_list[i]) == 1:
                if type(v_list[i]) is np.ndarray:
                    ax.scatter([0] * len(v_list[i][0]), v_list[i][0], label=label_list[i])
                else:
                    ax.scatter([0], v_list[i], label=label_list[i])
            else:
                ax.plot(v_list[i], label=label_list[i])
            ax.legend()
            ax.set_ylabel(f"{label_list[i]}")
        self.show_fig()

    def add_prop(self, key, value):
        self.data[key] = value
        dump_json(self.data, self.json_file)

    def save_fig(self, out_path):
        v_list = []
        label_list = []
        for k, v in self.data.items():
            if type(v) != list:
                continue
            try:
                if type(v[0]) == float or type(v[0]) == int or type(v[0]) == np.float64 or type(
                        v[0]) == np.int64 or type(v[0]) == np.float32 or type(v[0]) == np.int32:
                    v_list.append(v)
                    label_list.append(str(k))
                elif type(v[0]) == list:
                    v_list.append(np.array(v))
                    label_list.append([f"{k}_{i}" for i in range(len(v[0]))])
            except:
                pass

        prop_num = len(v_list)
        if prop_num == 0:
            return
        fig = plt.figure()
        for i in range(prop_num):
            ax = fig.add_subplot(prop_num, 1, i+1)
            if len(v_list[i]) == 1:
                if type(v_list[i]) is np.ndarray:
                    ax.scatter([0] * len(v_list[i][0]), v_list[i][0], label=label_list[i])
                else:
                    ax.scatter([0], v_list[i], label=label_list[i])
            else:
                ax.plot(v_list[i], label=label_list[i])
            ax.legend()
            ax.set_ylabel(f"{label_list[i]}")
        plt.savefig(out_path)
        self.show_fig()

    def show_fig(self):
        if self.fig_file is None:
            plt.pause(0.01)
        else:
            plt.figure(self.fig.number)
            plt.savefig(self.fig_file)



