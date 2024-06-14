from learning.discriminator import *
from learning.generator import *
from learning.encoder import *
from learning.w_encoder import *

import json

def get_models(model_names, nw_data=None, output_channel=None, config=None):
    # model_names: list of str

    # FNW(self, h_dim, w_dim)
    # CNNDis(self, nw_data, output_channel, gamma=0.9, max_num=40, sln=True, w_dim=10, ext_coeff=1.0)
    # GNNDis(self, nw_data, emb_dim, output_channel, gamma=0.9, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, sn=False, sln=False, w_dim=None, ext_coeff=1.0)
    # CNNGen(self, nw_data, output_channel, max_num=40, sln=True, w_dim=10)
    # GNNGen(self, nw_data, emb_dim, output_channel, enc_dim, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None)
    # CNNEnc(self, patch_size, emb_dim, mid_dim=1000, num_source=1, sln=True, w_dim=10)
    # ViTEnc(self, patch_size, vit_patch_size, emb_dim, mid_dim=1000, num_source=1, sln=True, w_dim=10, depth=6, heads=1, dropout=0.0, output_atten=False)
    # CNNWEnc(self, total_feature, w_dim)
    # GNNWEnc(self, feature_num, emb_dim, w_dim, adj_matrix)


    # model setting
    read_model_general = config["MODELGENERAL"]
    gamma = float(read_model_general["gamma"])  # float
    ext_coeff = float(read_model_general["ext_coeff"])  # float
    sln = bool(read_model_general["sln"])  # bool
    h_dim = int(read_model_general["h_dim"])  # int
    w_dim = int(read_model_general["w_dim"])  # int

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

    models = []
    for model_name in model_names:
        if model_name == "FNW":
            models.append(FF(h_dim, w_dim, bias=True, sn=False))
        elif model_name == "CNNDis":
            models.append(CNNDis(nw_data, output_channel, image_feature_num=emb_dim_enc, sn=False, sln=sln, w_dim=w_dim, ext_coeff=ext_coeff))
        elif model_name == "GNNDis":
            models.append(GNNDis(nw_data, emb_dim_dis, output_channel, image_feature_num=emb_dim_enc, gamma=gamma, in_emb_dim=in_emb_dim_dis, num_head=num_head_dis, dropout=dropout_dis, depth=depth_dis, sn=True, sln=sln, w_dim=w_dim, ext_coeff=ext_coeff))
        elif model_name == "CNNGen":
            models.append(CNNGen(nw_data, output_channel, image_feature_num=emb_dim_enc, max_num=max_num_gen, sln=sln, w_dim=w_dim))
        elif model_name == "GNNGen":
            models.append(GNNGen(nw_data, emb_dim_gen, output_channel, image_feature_num=emb_dim_enc, enc_dim=enc_dim_gen, in_emb_dim=in_emb_dim_gen, num_head=num_head_gen, dropout=dropout_gen, depth=depth_gen, pre_norm=False, sn=sn_gen, sln=sln, w_dim=w_dim))
        elif model_name == "CNNEnc":  # common for cnn and gnn link airl
            models.append(CNNEnc(patch_size_enc, emb_dim_enc, num_source=num_source_enc, sln=sln, w_dim=w_dim))
        elif model_name == "CNNTransEnc":  # common for cnn and gnn link airl
            models.append(CNNTransEnc(patch_size_enc, emb_dim_enc, num_source=num_source_enc, sln=sln, w_dim=w_dim))
        elif model_name == "ViTEnc":
            models.append(ViTEnc((3, patch_size_enc, patch_size_enc), (vit_patch_size_enc, vit_patch_size_enc), emb_dim_enc, mid_dim=mid_dim_enc, num_source=num_source_enc, sln=sln, w_dim=w_dim, depth=depth_enc, heads=num_head_enc, dropout=dropout_enc, output_atten=output_atten_enc))
        elif model_name == "CNNWEnc":
            models.append(CNNWEnc(nw_data.feature_num + nw_data.context_feature_num, w_dim))
        elif model_name == "GNNWEnc":
            adj_matrix = torch.tensor(nw_data.edge_graph).to(torch.float32)
            models.append(GNNWEnc(nw_data.feature_num, emb_dim_wenc, w_dim, adj_matrix))
        elif model_name == "UNetDis":
            models.append(UNetDis(nw_data.prop_dim+emb_dim_enc, 1, output_channel, gamma=gamma, sn=True, ext_coeff=ext_coeff))
        elif model_name == "UNetDisStatic":
            models.append(UNetDisStatic(nw_data.prop_dim+emb_dim_enc, 1, output_channel, gamma=gamma, sn=True, ext_coeff=ext_coeff, depth=depth_dis))
        elif model_name == "UNetGen":
            models.append(UNetGen(nw_data.prop_dim+emb_dim_enc, 1, sn=sn_gen, depth=depth_gen))
        else:
            raise Exception("Unknown model name: {}".format(model_name))

    return models



