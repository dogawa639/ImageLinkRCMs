# test
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

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    model_type = read_general["model_type"]  # cnn or gnn
    device = read_general["device"]
    # model
    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    pp_path = json.loads(read_data["pp_path"])  # list(str)
    image_data_dir = read_data["satellite_image_datadir"]  # str or None
    image_data_path = read_data["image_data_path"] # str or None
    image_data_path = None if image_data_path == "null" else image_data_path
    onehot_data_dir = read_data["onehot_image_datadir"]  # str or None
    onehot_data_path = read_data["onehot_data_path"]  # str or None
    onehot_data_path = None if onehot_data_path == "null" else onehot_data_path

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
    use_f0 = bool(read_model["use_f0"])  # bool
    emb_dim = int(read_model["emb_dim"])  # int
    enc_dim = int(read_model["enc_dim"])  # int
    in_emb_dim = json.loads(read_model["in_emb_dim"])  # int or None
    drop_out = float(read_model["drop_out"])  # float
    sn = bool(read_model["sn"])  # bool
    sln = bool(read_model["sln"])  # bool
    h_dim = int(read_model["h_dim"])  # int
    w_dim = int(read_model["w_dim"])  # int
    num_head = int(read_model["num_head"])  # int
    depth = int(read_model["depth"])  # int
    gamma = float(read_model["gamma"])   # float
    max_num = int(read_model["max_num"])  # int
    ext_coeff = float(read_model["ext_coeff"])  # float
    hinge_loss = bool(read_model["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model["hinge_thresh"])  # float or None
    patch_size = int(read_model["patch_size"])  # int
    # save setting
    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    fig_dir = read_save["figure_dir"]
    image_file = os.path.join(fig_dir, "train.png")

    # instance creation
    use_index = (model_type == "cnn")
    use_encoder = (image_data_path is not None)
    output_channel = len(pp_path)

    if model_type == "cnn":
        nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    else:
        nw_data = NetworkGNN(50, node_path, link_path, link_prop_path=link_prop_path)
    pp_list = [PP(ppath, nw_data) for ppath in pp_path]
    if model_type == "cnn":
        datasets = [GridDataset(pp, h_dim=h_dim) for pp in pp_list]
    else:
        datasets = [PPEmbedDataset(pp, h_dim=h_dim) for pp in pp_list]
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

    kwargs = {
        "nw_data": nw_data,
        "output_channel": output_channel,
        "emb_dim": emb_dim,
        "enc_dim": 0,#enc_dim,
        "in_emb_dim": in_emb_dim,
        "drop_out": drop_out,
        "sn": sn,
        "sln": sln,
        "h_dim": h_dim,
        "w_dim": w_dim,
        "num_head": num_head,
        "depth": depth,
        "gamma": gamma,
        "max_num": max_num,
        "ext_coeff": ext_coeff,
        "patch_size": patch_size,
        "num_source": 1
    }

    if not use_f0 and not use_encoder:
        discriminator, generator = get_models(model_names, **kwargs)
        f0 = None
        w_encoder = None
        encoder = None
    elif use_f0 and not use_encoder:
        discriminator, generator, f0, w_encoder = get_models(model_names, **kwargs)
        encoder = None
    elif not use_f0 and use_encoder:
        discriminator, generator, encoder = get_models(model_names, **kwargs)
        f0 = None
        w_encoder = None
    else:
        discriminator, generator, f0, w_encoder, encoder = get_models(model_names, **kwargs)

    image_data = None
    encoder = None

    airl = AIRL(generator, discriminator, use_index, datasets, model_dir, image_data=image_data, encoder=encoder, h_dim=h_dim, emb_dim=emb_dim, f0=f0,
                 hinge_loss=hinge_loss, hinge_thresh=hinge_thresh, patch_size=patch_size, device=device)

    airl.train_models(CONFIG, epoch, bs, lr_g, lr_d, shuffle, train_ratio=train_ratio, max_train_num=10, d_epoch=d_epoch, lr_f0=lr_f0, lr_e=lr_e, image_file=image_file)
