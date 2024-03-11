if __name__ == "__main__":
    import configparser
    import json
    import os
    from learning.generator import *
    from learning.discriminator import *
    from learning.encoder import *
    from learning.w_encoder import *
    from learning.util import get_models
    from preprocessing.network import *
    from preprocessing.pp import *
    from preprocessing.image import *
    from preprocessing.dataset import *
    from learning.airl import *

    CONFIG = "../../config/config_airl_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    model_type = read_general["model_type"]  # cnn or gnn
    device = read_general["device"]
    # data
    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    pp_path = json.loads(read_data["pp_path_train"])  # list(str)
    pp_path_test = json.loads(read_data["pp_path_test"])  # list(str)tmp
    image_data_dir = read_data["satellite_image_datadir"]  # str or None
    image_data_path = read_data["image_data_path"] # str or None
    image_data_path = None if image_data_path == "null" else image_data_path
    onehot_data_dir = read_data["onehot_image_datadir"]  # str or None
    onehot_data_path = read_data["onehot_data_path"]  # str or None
    onehot_data_path = None if onehot_data_path == "null" else onehot_data_path

    # train setting
    read_train = config["TRAIN"]
    bs = int(read_train["bs"])  # int
    epoch = 50#int(read_train["epoch"])  # int
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


    # encoder setting
    read_model_enc = config["ENCODER"]
    patch_size_enc = int(read_model_enc["patch_size"])  # int
    vit_patch_size_enc = json.loads(read_model_enc["vit_patch_size"])  # int
    mid_dim_enc = int(read_model_enc["mid_dim"])  # int
    emb_dim_enc = int(read_model_enc["emb_dim"])  # int
    num_source_enc = int(read_model_enc["num_source"])  # int
    num_head_enc = int(read_model_enc["num_head"])  # int
    depth_enc = int(read_model_enc["depth"])  # int
    dropout_enc = float(read_model_enc["dropout"])  # float
    output_atten_enc = True if read_model_enc["output_atten"] == "true" else False  # bool


    # save setting
    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")

    TRAIN = True
    TEST = True
    SHAP = True

    # instance creation
    use_index = (model_type == "cnn")
    use_encoder = (image_data_path is not None)
    output_channel = len(pp_path)

    if model_type == "cnn":
        nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    else:
        nw_data = NetworkGNN(50, node_path, link_path, link_prop_path=link_prop_path)
    pp_list = [PP(ppath, nw_data) for ppath in pp_path]
    for i in range(len(pp_list)):
        print(f"pp list {i}: ", sum([len(v["path"]) for v in pp_list[i].path_dict.values()]))

    if model_type == "cnn":
        datasets = [GridDataset(pp, h_dim=h_dim) for pp in pp_list]
        datasets_test = [GridDataset(PP(ppath, nw_data), h_dim=h_dim) for ppath in pp_path_test]
    else:
        datasets = [PPEmbedDataset(pp, h_dim=h_dim) for pp in pp_list]
        datasets_test = [PPEmbedDataset(PP(ppath, nw_data), h_dim=h_dim) for ppath in pp_path_test]
    image_data = None
    if image_data_path is not None:
        image_data_list = [os.path.join(image_data_dir, "satellite_image_processed.json"), os.path.join(onehot_data_dir, "onehot_image_processed.json")]
        image_data_list = [LinkImageData(image_data, nw_data) for image_data in image_data_list]
        image_data = CompressedImageData(image_data_list)


    # model_names : [str] [discriminator, generator, (f0, w_encoder), (encoder)]
    model_names = ["CNNDis", "CNNGen"] if model_type == "cnn" else ["GNNDis", "GNNGen"]
    if use_f0:
        model_names += ["FNW"]
        model_names += ["CNNWEnc"] if model_type == "cnn" else ["GNNWEnc"]
    if use_encoder:
        model_names += ["CNNEnc"]

    if not use_f0 and not use_encoder:
        discriminator, generator = get_models(model_names, nw_data=nw_data, output_channel=output_channel, config=config)
        f0 = None
        w_encoder = None
        encoder = None
    elif use_f0 and not use_encoder:
        discriminator, generator, f0, w_encoder = get_models(model_names, nw_data=nw_data, output_channel=output_channel, config=config)
        encoder = None
    elif not use_f0 and use_encoder:
        discriminator, generator, encoder = get_models(model_names, nw_data=nw_data, output_channel=output_channel, config=config)
        f0 = None
        w_encoder = None
    else:
        discriminator, generator, f0, w_encoder, encoder = get_models(model_names, nw_data=nw_data, output_channel=output_channel, config=config)

    #image_data = None
    #encoder = None
    ratio = (0.8, 0.2)

    airl = AIRL(generator, discriminator, use_index, datasets, model_dir, image_data=image_data, encoder=encoder, h_dim=h_dim, f0=f0,
                 hinge_loss=hinge_loss, hinge_thresh=hinge_thresh, device=device)

    if TRAIN:
        airl.train_models(CONFIG, epoch, bs, lr_g, lr_d, shuffle, ratio=ratio, max_train_num=10, d_epoch=d_epoch, lr_f0=lr_f0, lr_e=lr_e, image_file=image_file)
    if TEST:
        airl.load()
        airl.test(datasets_test)
    if SHAP:
        airl.load()
        airl.get_shap(datasets_test, 0)
