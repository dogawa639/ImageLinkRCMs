if __name__ == "__main__":
    import configparser
    import os
    import json
    from preprocessing.dataset import *
    from utility import *
    import numpy as np

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    image_data_path = read_data["image_data_path"]  # [{"name", "path", "z", "x", "y", "utm_num"}]
    image_data_dir = read_data["satellite_image_datadir"]

    image_data = load_json(image_data_path)
    base_dirs = [os.path.join(image_data_dir, x["name"]) for x in image_data]
    ximage_dataset = XImageDataset(base_dirs, corresponds=False, expansion=2, input_shape=(256, 256), crop=True, affine=True, transform_coincide=True, flip=True)

    ximage_dataset.show_samples(num_samples=3)
