if __name__ == "__main__":
    import configparser
    import json

    import torch
    from torchinfo import summary
    from preprocessing.network import *
    from preprocessing.pp import *
    from preprocessing.image import *
    from preprocessing.dataset import *
    from learning.util import *

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    model_type = read_general["model_type"]  # cnn or gnn
    device = read_general["device"]

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    pp_path = json.loads(read_data["pp_path"])  # list(str)
    image_data_path = read_data["image_data_path"]  # str or None
    image_data_path = None if image_data_path == "null" else image_data_path

    # model setting
    read_model = config["MODELSETTING"]
    use_f0 = bool(read_model["use_f0"])  # bool
    emb_dim = int(read_model["emb_dim"])  # int
    in_emb_dim = json.loads(read_model["in_emb_dim"])  # int or None
    drop_out = float(read_model["drop_out"])  # float
    sn = bool(read_model["sn"])  # bool
    sln = bool(read_model["sln"])  # bool
    h_dim = int(read_model["h_dim"])  # int
    w_dim = int(read_model["w_dim"])  # int
    num_head = int(read_model["num_head"])  # int
    depth = int(read_model["depth"])  # int
    gamma = float(read_model["gamma"])  # float
    max_num = int(read_model["max_num"])  # int
    ext_coeff = float(read_model["ext_coeff"])  # float
    hinge_loss = bool(read_model["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model["hinge_thresh"])  # float or None
    patch_size = int(read_model["patch_size"])  # int

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
    image_data = None
    num_source = 0
    if image_data_path is not None:
        image_data = SatelliteImageData(image_data_path)
        num_source = len(image_data)
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
        "num_source": num_source
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

    print(model_type)
    if model_type == "cnn":
        total_feature_num = nw_data.feature_num + nw_data.context_feature_num
        input_size = (10, total_feature_num, 3, 3)
        input_size2 = (10, 2, 3, 3)
        print("---Discriminator---")
        print(summary(model=discriminator, input_size=[input_size, input_size2, (10, w_dim)]))
        print("---Generator---")
        print(summary(model=generator, input_size=[input_size, (10, w_dim)]))
        if f0 is not None:
            print("---F0---")
            print(summary(model=f0, input_size=(10, h_dim)))
        if w_encoder is not None:
            print("---W_encoder---")
            print(summary(model=w_encoder, input_size=[input_size, (10, 3, 3)]))
        if encoder is not None:
            print("---Encoder---")
            print(summary(model=encoder, input_size=[(10, 1000), (10, w_dim)]))
    elif model_type == "gnn":
        input_size = (5, nw_data.link_num, nw_data.feature_num)
        input_size2 = (5, nw_data.link_num, nw_data.link_num, 2)
        print("---Discriminator---")
        print(summary(model=discriminator, input_size=[input_size, input_size2, (5, w_dim)]))
        print("---Generator---")
        print(summary(model=generator, input_size=[input_size, (5, w_dim)]))
        if f0 is not None:
            print("---F0---")
            print(summary(model=f0, input_size=(5, h_dim)))
        if w_encoder is not None:
            print("---W_encoder---")
            print(summary(model=w_encoder, input_size=[input_size, (5, nw_data.link_num, nw_data.link_num)]))
        if encoder is not None:
            print("---Encoder---")
            print(summary(model=encoder, input_size=[(5, 1000), (5, w_dim)]))



