import datetime
import time
import os

import matplotlib.pyplot as plt

from utility import *

class Logger:
    def __init__(self, json_file, conf_file):
        self.json_file = json_file
        self.conf_file = conf_file

        self.start_time = datetime.datetime.now()
        self.data = {
                "start_time": str(self.start_time),
                "config": conf_file
        }
        self.fig = plt.figure()
        plt.tight_layout()

    def __del__(self):
        self.end_time = datetime.datetime.now()
        self.data["end_time"] = str(self.end_time)
        self.data["duration"] = str(self.end_time - self.start_time)

        dump_json(self.data, self.json_file)
        plt.show()

    def add_log(self, key, value):
        # value: float or 1d list
        if key in self.data:
            self.data[key].append(value)
        else:
            self.data[key] = [value]

        self.fig.clf()
        cnt = 1
        for k, v in self.data.items():
            if type(v) != list:
                continue
            plt.add_subplot(len(self.data)-2, cnt, 1)
            if type(v[0]) == float:
                plt.plot(v, label=k)
            elif type(v[0]) == list:
                plt.plot(v, label=[f"{k}_{i}" for i in range(len(v[0]))])
            plt.legend()
            cnt += 1
        plt.pause(0.01)


