from learning.discriminator import *
from learning.generator import *
from learning.encoder import *
from learning.w_encoder import *

def get_models(model_names, nw_data, output_channel, emb_dim, in_emb_dim, drop_out, sn, sln, h_dim, w_dim,
               num_head=3, depth=6, gamma=0.9, max_num=40, ext_coeff=1.0, patch_size=None, num_source=0):
    # model_names: list of str

    # FNW(self, h_dim, w_dim)
    # CNNDis(self, nw_data, output_channel, gamma=0.9, max_num=40, sln=True, w_dim=10, ext_coeff=1.0)
    # GNNDis(self, nw_data, emb_dim, output_channel, gamma=0.9, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, sn=False, sln=False, w_dim=None, ext_coeff=1.0)
    # CNNGen(self, nw_data, output_channel, max_num=40, sln=True, w_dim=10)
    # GNNGen(self, nw_data, emb_dim, output_channel, enc_dim, in_emb_dim=None, num_head=1, dropout=0.0, depth=1, pre_norm=False, sn=False, sln=False, w_dim=None)
    # CNNEnc(self, nw_data, output_channel, max_num=40, sln=True, w_dim=10)
    # CNNWEnc(self, total_feature, w_dim)
    # GNNWEnc(self, feature_num, emb_dim, w_dim, adj_matrix)

    models = []
    for model_name in model_names:
        if model_name == "FNW":
            models.append(FF(h_dim, w_dim, bias=True, sn=False))
        elif model_name == "CNNDis":
            models.append(CNNDis(nw_data, output_channel, gamma=gamma, max_num=max_num, sn=True, sln=sln, w_dim=w_dim, ext_coeff=ext_coeff))
        elif model_name == "GNNDis":
            models.append(GNNDis(nw_data, emb_dim, output_channel, gamma=gamma, in_emb_dim=in_emb_dim, num_head=num_head, dropout=drop_out, depth=depth, sn=True, sln=sln, w_dim=w_dim, ext_coeff=ext_coeff))
        elif model_name == "CNNGen":
            models.append(CNNGen(nw_data, output_channel, max_num=max_num, sln=sln, w_dim=w_dim))
        elif model_name == "GNNGen":
            models.append(GNNGen(nw_data, emb_dim, output_channel, enc_dim=emb_dim, in_emb_dim=in_emb_dim, num_head=num_head, dropout=drop_out, depth=depth, pre_norm=False, sn=sn, sln=sln, w_dim=w_dim))
        elif model_name == "CNNEnc":
            models.append(CNNEnc(patch_size, emb_dim, num_source=num_source, sln=sln, w_dim=w_dim))
        elif model_name == "CNNWEnc":
            models.append(CNNWEnc(nw_data.feature_num, w_dim))
        elif model_name == "GNNWEnc":
            adj_matrix = torch.tensor(nw_data.edge_graph).to(torch.float32)
            models.append(GNNWEnc(nw_data.feature_num, emb_dim, w_dim, adj_matrix))
        else:
            raise Exception("Unknown model name: {}".format(model_name))

    return models



