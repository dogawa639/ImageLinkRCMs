if __name__ == "__main__":
    import configparser
    import os
    import json
    from preprocessing.dataset import *
    from utility_kalmanfilter import *
    import pandas as pd
    import numpy as np

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    image_data_dir = read_data["satellite_image_datadir"]
    link_prop_path = os.path.join(image_data_dir, "link_prop.csv")
    image_data_path = read_data["image_data_path"]  # [{"name", "path", "z", "x", "y", "utm_num"}]
    image_data_dir = read_data["satellite_image_datadir"]
    onehot_data_path = read_data["onehot_data_path"]
    onehot_data_dir = read_data["onehot_image_datadir"]
    streetview_data_path = read_data["streetview_data_path"]

    XIMAGE = True
    XYIMAGE = True  # image classification
    XHIMAGE = True
    STIMAGE = True
    STXIMAGE = True

    image_data = load_json(image_data_path)
    base_dirs_x = [os.path.join(image_data_dir, x["name"]) for x in image_data]
    one_hot_data = load_json(onehot_data_path)
    base_dirs_h = [os.path.join(onehot_data_dir, x["name"]) for x in one_hot_data]

    link_prop = read_csv(link_prop_path)
    lids = link_prop["LinkID"].values
    print(f"Number of links: {len(lids)}")
    link_prop.drop("LinkID", axis=1, inplace=True)
    columns = ["Path"] + link_prop.columns.tolist()
    y_df = None
    for x in image_data:
        path_tmp = [os.path.join(image_data_dir, x["name"], f"{lid}.png") for lid in lids]
        y_tmp = pd.concat([pd.DataFrame({"Path": path_tmp}), link_prop], axis=1)
        if y_df is None:
            y_df = y_tmp
        else:
            y_df = pd.concat([y_df, y_tmp], axis=0)
    print(f"Length of Y: {len(y_df)}")

    kwargs = {"corresponds": False,
              "expansion": 2,
              "crop":True,
              "affine": True,
              "transform_coincide": True,
              "flip": True}
    if XIMAGE:
        ximage_dataset = XImageDataset(base_dirs_x, **kwargs)
        print(len(ximage_dataset))
        d1, d2 = ximage_dataset.split_into((0.8, 0.2))
        print(len(d1), len(d2))
        ximage_dataset.show_samples(num_samples=3)

    if XYIMAGE:
        xyimage_dataset = XYImageDataset(base_dirs_x, y_df, **kwargs)
        print(len(xyimage_dataset))
        d1, d2 = xyimage_dataset.split_into((0.8, 0.2))
        print(len(d1), len(d2))
        xyimage_dataset.show_samples(num_samples=3)

    if XHIMAGE:
        del kwargs["corresponds"]
        xhimage_dataset = XHImageDataset(base_dirs_x, base_dirs_h, **kwargs)
        print(len(xhimage_dataset))
        d1, d2 = xhimage_dataset.split_into((0.8, 0.2))
        print(len(d1), len(d2))
        xhimage_dataset.show_samples(num_samples=3)
        d1.show_samples(num_samples=3)

    if STIMAGE:
        stimage_dataset = StreetViewDataset(streetview_data_path, link_prop_path, **kwargs)
        print(len(stimage_dataset))
        d1, d2 = stimage_dataset.split_into((0.8, 0.2))
        print(len(d1), len(d2))
        stimage_dataset.show_samples(num_samples=3)
        d1.show_samples(num_samples=3)

    if STXIMAGE:
        stximage_dataset = StreetViewXDataset(streetview_data_path, base_dirs_x, link_prop_path, **kwargs)
        print(len(stximage_dataset))
        d1, d2 = stximage_dataset.split_into((0.8, 0.2))
        print(len(d1), len(d2))
        stximage_dataset.show_samples(num_samples=3)
        d1.show_samples(num_samples=3)