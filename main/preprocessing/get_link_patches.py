if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import json
    import configparser
    from preprocessing.network import *
    from preprocessing.image import *
    from utility import *

    CONFIG = "../../config/config_all.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]  # csv
    link_path = read_data["link_path"]  # csv
    link_prop_path = read_data["link_prop_path"]  # csv
    image_data_path = read_data["image_data_path"]  # json
    image_data_dir = read_data["satellite_image_datadir"]  # dir
    onehot_data_path = read_data["onehot_data_path"]  # json
    onehot_data_dir = read_data["onehot_image_datadir"]  # dir

    SATELLITE = False
    ONEHOT = True
    LINKPROP = True
    nw_data = NetworkCNN(node_path, link_path)

    if SATELLITE:
        output_data_path = os.path.splitext(image_data_path)[0] + "_processed.json"
        image_data = SatelliteImageData(image_data_path, output_data_file=output_data_path)
        image_data.set_voronoi(nw_data)
        image_data.set_datafolder(image_data_dir, patch_size=128)
    if ONEHOT:
        output_data_path = os.path.splitext(onehot_data_path)[0] + "_processed.json"
        image_data = OneHotImageData(onehot_data_path, output_data_file=output_data_path)
        image_data.set_voronoi(nw_data)
        image_data.set_datafolder(onehot_data_dir, patch_size=128)
        if LINKPROP:  # overwrite link_prop data with one-hot data
            print(f"Write link prop in {link_prop_path}.")
            image_data.write_link_prop(onehot_data_dir, link_prop_path)
