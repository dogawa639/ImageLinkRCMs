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
    read_model_general = config["MODELGENERAL"]
    use_f0 = bool(read_model_general["use_f0"])  # bool
    h_dim = int(read_model_general["h_dim"])  # int
    w_dim = int(read_model_general["w_dim"])  # int

    # encoder setting
    read_model_enc = config["ENCODER"]
    model_type_enc = read_model_enc["model_type"]  # cnn or vit

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
        if model_type_enc == "cnn":
            model_names += ["CNNEnc"]
        elif model_type_enc == "vit":
            model_names += ["ViTEnc"]

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

        # discriminator setting
        read_model_dis = config["DISCRIMINATOR"]
        emb_dim_dis = int(read_model_dis["emb_dim"])  # int
        enc_dim_dis = int(read_model_dis["enc_dim"])  # int
        in_emb_dim_dis = json.loads(read_model_dis["in_emb_dim"])  # int or None
        num_head_dis = int(read_model_dis["num_head"])  # int
        depth_dis = int(read_model_dis["depth"])  # int
        dropout_dis = float(read_model_dis["dropout"])  # float

        # generator setting
        read_model_gen = config["GENERATOR"]
        emb_dim_gen = int(read_model_gen["emb_dim"])  # int
        enc_dim_gen = int(read_model_gen["enc_dim"])  # int
        in_emb_dim_gen = json.loads(read_model_gen["in_emb_dim"])  # int or None
        num_head_gen = int(read_model_gen["num_head"])  # int
        depth_gen = int(read_model_gen["depth"])  # int
        sn_gen = True if read_model_gen["sn"] == "true" else False  # bool
        dropout_gen = float(read_model_gen["dropout"])  # float
        max_num_gen = int(read_model_gen["max_num"])  # int

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

        # w_encoder setting
        read_model_wenc = config["WENCODER"]
        emb_dim_wenc = int(read_model_wenc["emb_dim"])  # int

    print(model_type)
    if model_type == "cnn":
        total_feature_num = nw_data.feature_num + nw_data.context_feature_num + emb_dim_enc
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
        total_feature_num = nw_data.feature_num + nw_data.context_feature_num + emb_dim_enc
        input_size = (5, nw_data.link_num, total_feature_num)
        input_size2 = (5, 2, nw_data.link_num, nw_data.link_num)
        print("---Discriminator---")
        print(summary(model=discriminator, input_size=[input_size, input_size2, (5, w_dim)]))
        print("---Generator---")
        print(summary(model=generator, input_size=[input_size, (5, w_dim)]))
        if f0 is not None:
            print("---F0---")
            print(summary(model=f0, input_size=(5, h_dim)))
        if w_encoder is not None:
            print("---W_encoder---")
            input_size = (5, nw_data.link_num, nw_data.feature_num)
            print(summary(model=w_encoder, input_size=[input_size, (5, nw_data.link_num, nw_data.link_num)]))
        if encoder is not None:
            print("---Encoder---")
            print(summary(model=encoder, input_size=[(5, mid_dim_enc), (5, w_dim)]))



