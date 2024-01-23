if __name__ == "__main__":
    import configparser
    import json
    import os

    import geopandas as gpd
    import pandas as pd

    from learning.generator import *
    from learning.discriminator import *
    from learning.util import get_models
    from preprocessing.mesh import *
    from preprocessing.mesh_trajectory import *
    from preprocessing.dataset import *
    from learning.mesh_airl import *

    CONFIG = "../../config/config_mesh.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    device = "cpu"#read_general["device"]
    # data
    read_data = config["DATA"]
    traj_dir = read_data["traj_dir"]

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
    read_model = config["MODELSETTING"]
    drop_out = float(read_model["drop_out"])  # float
    sn = bool(read_model["sn"])  # bool
    gamma = float(read_model["gamma"])   # float
    ext_coeff = float(read_model["ext_coeff"])  # float
    hinge_loss = bool(read_model["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model["hinge_thresh"])  # float or None
    mesh_dist = int(read_model["mesh_dist"])  # int
    # save setting
    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")

    TRAIN = True
    TEST = True

    # instance creation
    traj_path = [os.path.join(traj_dir, "220928", "4", "trajectory_train_0.csv"),
                 os.path.join(traj_dir, "220928", "4", "trajectory_train_2.csv")]

    traj_path_test = [os.path.join(traj_dir, "220928", "4", "trajectory_test_0.csv"),
                 os.path.join(traj_dir, "220928", "4", "trajectory_test_2.csv")]

    for i in range(len(traj_path)):
        traj = pd.read_csv(traj_path[i])
        traj_test = pd.read_csv(traj_path_test[i])
        print(f"transition sample num {i}:", len(traj))
        print(f"transition sample num test {i}:", len(traj_test))

    grid = gpd.read_file(os.path.join(traj_dir, "220928", "4", "grid.gpkg"))
    rows = grid["row_index"].max() + 1
    cols = grid["col_index"].max() + 1
    mnw_data = MeshNetwork(grid.total_bounds, cols, rows, len(traj_path) + 5)
    for i in grid.index:
        mnw_data.cells[int(grid.loc[i, "row_index"])][ int(grid.loc[i, "col_index"])].add_props(int(grid.loc[i, "prop_num"] + len(traj_path)-1))

    output_channel = len(traj_path)
    mesh_traj = MeshTraj(traj_path, mnw_data)
    dataset = MeshDataset(mesh_traj, mesh_dist)


    # model_names : [str] [discriminator, generator]
    model_names = ["UNetDis", "UNetGen"]

    kwargs = {
        "nw_data": mnw_data,
        "output_channel": output_channel,
        "drop_out": drop_out,
        "sn": sn,
        "gamma": gamma,
        "ext_coeff": ext_coeff
    }
    generators = []
    discriminators = []
    for _ in range(output_channel):
        discriminator, generator = get_models(model_names, **kwargs)
        generators.append(generator)
        discriminators.append(discriminator)

    airl = MeshAIRL(generators, discriminators, dataset, model_dir,
                 hinge_loss=hinge_loss, hinge_thresh=hinge_thresh, device=device)

    if TRAIN:
        airl.train_models(CONFIG, epoch, bs, lr_g, lr_d, shuffle, train_ratio=train_ratio, max_train_num=10, d_epoch=d_epoch, image_file=image_file)

    if TEST:
        mesh_traj = MeshTraj(traj_path_test, mnw_data)
        dataset = MeshDataset(mesh_traj, mesh_dist)
        airl.load()
        airl.test_models(CONFIG, dataset)