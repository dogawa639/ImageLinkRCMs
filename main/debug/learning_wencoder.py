if __name__ == "__main__":
    import torch
    from models.general import *
    from learning.w_encoder import *

    device = "mps"
    bs = 2
    feature_num = 3
    emb_dim = 4
    w_dim = 5
    link_num = 6
    adj_matrix = torch.randint(0, 2, (link_num, link_num), device=device).to(torch.float32)

    cnnwenc = CNNWEnc(feature_num, w_dim).to(device)
    inputs = torch.randn((bs, feature_num, 3, 3), device=device)
    g_output = torch.randn((bs, 3, 3), device=device)

    w_cnn = cnnwenc(inputs, g_output)

    gnnwenc = GNNWEnc(feature_num, emb_dim, w_dim, adj_matrix).to(device)
    inputs = torch.randn((bs, link_num, feature_num), device=device)
    g_output = torch.randn((bs, link_num, link_num), device=device)

    w_gnn = gnnwenc(inputs, g_output)

    print(w_cnn.shape, w_gnn.shape)