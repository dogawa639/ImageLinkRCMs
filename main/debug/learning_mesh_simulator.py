if __name__ == "__main__":
    import configparser
    import json
    import os
    from learning.generator import *
    from learning.discriminator import *
    from learning.util import get_models
    from preprocessing.mesh import *
    from preprocessing.mesh_trajectory import *
    from preprocessing.dataset import *
    from learning.mesh_airl import *
    from learning.mesh_simulator import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    device = read_general["device"]
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
    gamma = float(read_model["gamma"])  # float
    ext_coeff = float(read_model["ext_coeff"])  # float
    hinge_loss = bool(read_model["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model["hinge_thresh"])  # float or None
    mesh_dist = int(read_model["mesh_dist"])  # int
    # save setting
    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")
    debug_dir = read_save["debug_dir"]

    # instance creation
    traj_path = [os.path.join(traj_dir, "20220928-050000MA", "trajectory_0.csv"),
                 os.path.join(traj_dir, "20220928-050000MA", "trajectory_0.csv")]
    mnw_data = MeshNetwork((-69280, 93470, -69255, 93495), 25, 25, 3)

    output_channel = len(traj_path)
    mesh_traj = MeshTraj(traj_path, mnw_data)
    dataset = MeshDataset(mesh_traj, mesh_dist)

    # model_names : [str] [discriminator, generator]
    model_names = ["UNetGen"]

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
        generator = get_models(model_names, **kwargs)[0]
        generators.append(generator)

    # od_data: [{timestep : [(o_idx_x, o_idx_y, d_idx_x, d_idx_y)]}]. len(od_data) == output_channel
    od_data = [{0: [(0, 0, 24, 24)]}, {1: [(0, 0, 24, 24)]}]
    simulator = MeshSimulator(generators, dataset, device=device)
    simulator.simulate(od_data, 10, out_dir=os.path.join(debug_dir, "data", "simulator"))