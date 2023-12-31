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
    onehot_data_path = read_data["onehot_data_path"]
    onehot_data_dir = read_data["onehot_image_datadir"]

    SATELLITE = True
    ONEHOT = True
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    if SATELLITE:
        image_data = SatelliteImageData(image_data_path, resolution=0.5, output_data_file=os.path.join(image_data_dir, "satellite_image.json"))
        image_data.set_voronoi(nw_data)
        image_data.set_datafolder(image_data_dir)
    if ONEHOT:
        image_data = OneHotImageData(onehot_data_path, resolution=0.5, output_data_file=os.path.join(onehot_data_dir, "onehot_image.json"))
        image_data.set_voronoi(nw_data)
        image_data.set_datafolder(onehot_data_dir)
        image_data.write_link_prop(onehot_data_dir, os.path.join(onehot_data_dir, "link_prop.csv"))

