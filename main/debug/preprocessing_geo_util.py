if __name__ == "__main__":
    import configparser
    import os
    import json

    from preprocessing.geo_util import *
    from utility import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    image_data_path = read_data["image_data_path"]

    image_data_list = load_json(image_data_path)
    #print(image_data_list[1]["name"])
    #map_path = image_data_list[1]["path"]
    map_path = "/Users/dogawa/Desktop/bus/aviational/data/ohsaka_avi_lum.png"

    map_seg = MapSegmentation([map_path], max_class_num=3)
    one_hot = map_seg.convert_file(map_path)
    print(one_hot.shape, one_hot.dtype)
    map_seg.write_colormap()
    map_seg.write_hist()



