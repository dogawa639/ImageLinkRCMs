import datetime
import time
import os

import numpy as np
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

        self.closed = False

    def close(self):
        self.end_time = datetime.datetime.now()
        self.data["end_time"] = str(self.end_time)
        self.data["duration"] = str(self.end_time - self.start_time)

        dump_json(self.data, self.json_file)
        plt.show()

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

        plt.close("all")
        fig = plt.figure()
        plt.tight_layout()
        cnt = 1
        for k, v in self.data.items():
            if type(v) != list:
                continue
            ax = fig.add_subplot(len(self.data)-2, 1, cnt)
            try:
                if type(v[0]) == float or type(v[0]) == int or type(v[0]) == np.float64 or type(v[0]) == np.int64 or type(v[0]) == np.float32 or type(v[0]) == np.int32:
                    ax.plot(v, label=k)
                elif type(v[0]) == list:
                    ax.plot(np.array(v), label=[f"{k}_{i}" for i in range(len(v[0]))])
            except:
                pass
            ax.legend()
            ax.set_ylabel(f"{k}")
            cnt += 1
        plt.pause(0.01)


# test
if __name__ == "__main__":
    logger = Logger("/Users/dogawa/PycharmProjects/GANs/debug/test.json", "")
    for i in range(10):
        logger.add_log("test", i)
        time.sleep(1)
    logger.close()

