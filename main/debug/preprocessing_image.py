if __name__ == "__main__":
    import configparser
    import os
    import json
    from preprocessing.image import *
    from preprocessing.network import *
    import numpy as np

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_general = config["GENERAL"]
    device = read_general["device"]

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]
    image_data_path = read_data["image_data_path"]
    image_data_dir = read_data["satellite_image_datadir"]

    image_data = SatelliteImageData(image_data_path)
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    image_data.set_voronoi(nw_data)
    image_data.write_gtif()
    image_data.set_datafolder(image_data_dir)
