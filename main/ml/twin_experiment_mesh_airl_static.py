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

    CONFIG = "../../config/config_mesh_static_noimage.ini"
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
    data_dir = read_data["data_dir"]
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
    h_dim = int(read_model_general["h_dim"])  # int
    w_dim = int(read_model_general["w_dim"])  # int

    mesh_dist = int(read_model_general["mesh_dist"])  # int

    # encoder setting
    read_model_enc = config["ENCODER"]
    model_type_enc = read_model_enc["model_type"]  # cnn or vit
    patch_size = int(read_model_enc["patch_size"])  # int

    # save setting
    read_save = config["SAVE"]
    output_dir = read_save["output_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")

    IMAGE = False
    USESMALL = True
    ADDOUTPUT = True
    SAVEDATA = True
    LOADDATA = True
    TRAIN = True
    TEST = True
    SHOWATTEN = False
    target_case = "20240414111152"  # only used when ADDOUTPUT is False

    # add datetime to output_dir
    if ADDOUTPUT:
        date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        print(f"date_str: {date_str}")
        output_dir = os.path.join(output_dir, date_str)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise Exception("Output directory already exists.")
    else:
        output_dir = os.path.join(output_dir, target_case)

    # instance creation
    pp_path = [pp_path[0]]
    output_channel = len(pp_path)

    use_encoder = image_data_path != "null"
    if os.path.exists(mask_path):
        bb_coords = gpd.read_file(mask_path).to_crs(epsg=2442 + utm_num).total_bounds
    dx, dy = (bb_coords[2] - bb_coords[0]),  (bb_coords[3] - bb_coords[1])
    w_dim,  h_dim = int(dx / mesh_dist), int(dy / mesh_dist)

    print("Create MeshNetwork object")
    print(f"bb_coords: {bb_coords}, w_dim: {w_dim}, h_dim: {h_dim}")
    mnw_data = MeshNetwork(bb_coords, w_dim, h_dim, 3)  # prop_dim: prop from one_hot image

    # main process
    if IMAGE:
        # split image into mesh
        print("Split image into mesh")
        image_data = SatelliteImageData(image_data_path, resolution=0.5,
                                        output_data_file=os.path.join(mesh_image_dir,
                                                                      "satellite_image_processed.json"))
        image_data.split_by_mesh(mnw_data, mesh_image_dir)
        onehot_data = OneHotImageData(onehot_data_path, resolution=0.5,
                                      output_data_file=os.path.join(mesh_image_dir,
                                                                    "onehot_image_processed.json"))
        onehot_data.split_by_mesh(mnw_data, mesh_image_dir, patch_size=128, aggregate=True)

    print("Load MeshNetwork property")
    mnw_data.load_prop(os.path.join(mesh_image_dir, "landuse", "aggregated.npy"))

    # load mesh base pp data
    print("Load MeshTrajectoryStatic object")
    pp_path_small = [pp_path[i].replace(".csv", "_small.csv") for i in range(len(pp_path))]
    if USESMALL:
        mesh_traj_data = MeshTrajStatic(pp_path_small, mnw_data)
    else:
        mesh_traj_data = MeshTrajStatic(pp_path, mnw_data, pp_path_small)  # write down the trimmed data into the pp_path_small
    mesh_traj_data = mesh_traj_data.split_into([0.1])[0]
    dataset = MeshDatasetStatic(mesh_traj_data, 1)
    # create pseudo dataset for twin experiment overwrite original data
    num_d_train = 50
    num_d_test = 20
    params = [0., 0., 0., -1.]
    print(f"Create pseudo dataset train & val num_d: ({num_d_train}) and test num_d: ({num_d_test})")
    print(f"  params: {params}")
    dataset_train = dataset.get_pseudo_dataset(num_d_train, params)
    dataset_test = dataset.get_pseudo_dataset(num_d_test, params)
    if SAVEDATA:
        data_path_train = os.path.join(data_dir, "twin_exp/data_train.pkl")
        data_path_test = os.path.join(data_dir, "twin_exp/data_test.pkl")
        save_class_val(dataset_train, data_path_train)
        save_class_val(dataset_test, data_path_test)
    if LOADDATA:
        data_path_train = os.path.join(data_dir, "twin_exp/data_train.pkl")
        data_path_test = os.path.join(data_dir, "twin_exp/data_test.pkl")
        dataset_train = load_class_val(dataset_train, data_path_train)
        dataset_test = load_class_val(dataset_test, data_path_test)

    # normalize state and context
    print("Normalize state and context")
    params = dataset_train.get_normalization_params()
    dataset_train.normalize(*params)
    dataset_test.normalize(*params)

    # load satellite image data
    print("Load MeshImageData object")
    image_data = MeshImageData(os.path.join(mesh_image_dir, "satellite_image_processed.json"), mnw_data, img_shapes=[(3, patch_size, patch_size)])

    # load models
    # model_names : [str] [discriminator, generator]
    print("Load models")
    model_names = ["UNetDisStatic", "UNetGen"]

    kwargs = {
        "nw_data": mnw_data,
        "output_channel": output_channel,
        "config": config
    }
    generators = []
    discriminators = []
    for _ in range(output_channel):
        discriminator, generator = get_models(model_names, **kwargs)
        generators.append(generator)
        discriminators.append(discriminator)

    encoders = None
    if use_encoder:
        if model_type_enc == 'cnn':
            model_names = ["CNNEnc"]
        elif model_type_enc == 'vit':
            model_names = ["ViTEnc"]
        encoders = []
        for _ in range(output_channel):
            encoders.append(get_models(model_names, **kwargs)[0])

    # create airl object and perform train and test
    print("Create MeshAIRLStatic object")
    airl = MeshAIRLStatic(generators, discriminators, dataset_train, output_dir,
                 image_data=image_data, encoders=encoders, hinge_loss=hinge_loss, hinge_thresh=hinge_thresh, device=device)

    if TRAIN:
        print("Pre training start")
        airl.pretrain_models(CONFIG, 1, bs, lr_g, shuffle, train_ratio=train_ratio)
        print("Training start")
        airl.train_models(CONFIG, epoch, bs, lr_g, lr_d, shuffle, train_ratio=train_ratio, d_epoch=d_epoch, image_file=image_file)

    if TEST:
        print("Testing start")
        #airl.load(model_dir=os.path.join(output_dir, "20240407131846"))
        airl.load()
        airl.test_models(CONFIG, dataset_test)

    if SHOWATTEN:
    #  show attention map
        if not  os.path.exists(os.path.join(output_dir, "atten")):
            os.mkdir(os.path.join(output_dir, "atten"))
        for row in range(h_dim):
            for col in range(w_dim):

                img_tensor = image_data.load_mesh_image(row, col)[0]
                org_img = image_data.load_org_image(row, col)
                attens = airl.show_attention_map(img_tensor, show=False)
                
                output_channel = len(attens)
                if img_tensor.dim() == 3:
                    bs = 1
                else:
                    bs = img_tensor.shape[0]
                fig = plt.figure(figsize=((1 + output_channel) * 4, bs * 4))
                for i in range(bs):
                    plt.subplot(bs, 1 + output_channel, i * (1 + output_channel) + 1)
                    plt.imshow(org_img[i])
                    for j in range(output_channel):
                        plt.subplot(bs, 1 + output_channel, i * (1 + output_channel) + j + 2)
                        plt.imshow(attens[j][i])
                plt.savefig(os.path.join(output_dir, "atten", f"{row}_{col}.png"))
                plt.clf()
                plt.close()
    print("Program ends.")
