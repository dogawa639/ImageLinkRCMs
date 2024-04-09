if __name__ == "__main__":
    import configparser
    import json
    import datetime
    import os

    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt

    from learning.generator import *
    from learning.discriminator import *
    from learning.util import get_models
    from preprocessing.mesh import *
    from preprocessing.mesh_trajectory import *
    from preprocessing.dataset import *
    from preprocessing.geo_util import *
    from preprocessing.image import *
    from preprocessing.mesh_trajectory import *
    from learning.mesh_airl_static import *
    from utility import *

    CONFIG = "../../config/config_mesh_static.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    device = read_general["device"]
    # geographic
    read_geo = config["GEOGRAPHIC"]
    utm_num = int(read_geo["utm_num"])  # int
    bb_coords = json.loads(read_geo["bb_coords"])  # list(float)
    mask_path = read_geo["mask_path"]
    # data
    read_data = config["DATA"]
    pp_path = json.loads(read_data["pp_path"])
    image_data_path = read_data["image_data_path"]
    onehot_data_path = read_data["onehot_data_path"]
    mesh_image_dir = read_data["mesh_image_dir"]

    # train setting
    read_train = config["TRAIN"]
    bs = int(read_train["bs"])  # int
    epoch = int(read_train["epoch"])  # int
    lr_g = float(read_train["lr_g"])  # float
    lr_d = float(read_train["lr_d"])  # float
    lr_f0 = float(read_train["lr_f0"])  # float
    lr_e = float(read_train["lr_e"])  # float
    shuffle = bool(read_train["shuffle"])  # bool
    train_ratio = float(read_train["train_ratio"])  # float
    d_epoch = int(read_train["d_epoch"])  # int

    # model setting
    read_model_general = config["MODELGENERAL"]
    use_f0 = True if read_model_general["use_f0"] == "true" else False  # bool
    gamma = float(read_model_general["gamma"])  # float
    ext_coeff = float(read_model_general["ext_coeff"])  # float
    hinge_loss = bool(read_model_general["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model_general["hinge_thresh"])  # float or None
    sln = False  # bool(read_model_general["sln"])  # bool

    # encoder setting
    read_model_enc = config["ENCODER"]
    model_type_enc = read_model_enc["model_type"]  # cnn or vit
    patch_size = int(read_model_enc["patch_size"])  # int


    # instance creation
    output_channel = len(pp_path)
    use_encoder = image_data_path != "null"
    if os.path.exists(mask_path):
        bb_coords = gpd.read_file(mask_path).to_crs(epsg=2442 + utm_num).total_bounds
    dx, dy = (bb_coords[2] - bb_coords[0]),  (bb_coords[3] - bb_coords[1])

    # get context mutual information for different mesh_dist
    mis = []
    mesh_dists = list(range(10, 500, 10))
    for mesh_dist in mesh_dists:
        w_dim,  h_dim = int(dx / mesh_dist), int(dy / mesh_dist)

        print("Create MeshNetwork object")
        mnw_data = MeshNetwork(bb_coords, w_dim, h_dim, 0)  # prop_dim: prop from one_hot image

        # load mesh base pp data
        print("Load MeshTrajectoryStatic object")
        pp_path_small = [pp_path[i].replace(".csv", "_small.csv") for i in range(len(pp_path))]
        mesh_traj_data = MeshTrajStatic(pp_path, mnw_data, pp_path_small)  # write down the trimmed data into the pp_path_small
        dataset = MeshDatasetStatic(mesh_traj_data, 1)

        mi = dataset.get_context_mi()
        mis.append(mi)

    plt.plot(mesh_dists, mis)
    plt.xlabel("mesh_dist")
    plt.ylabel("context mutual information")
    plt.show()




