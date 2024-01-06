if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from models.general import *
    from models.gnn import *

    device = "mps"
    bs = 3
    emb_dim_in = 4
    emb_dim_out = 5
    node_num = 6
    adj_matrix = torch.randint(0, 2, (node_num, node_num)).to(torch.float32)
    w_dim = 7

    gat = GAT(emb_dim_in, emb_dim_out, adj_matrix,
              in_emb_dim=None, num_head=2, dropout=0.1, depth=2, output_atten=True, sn=True, sln=False, w_dim=w_dim,
              atten_fn="dense").to(device)
    x = torch.randn((bs, node_num, emb_dim_in), device=device)
    w = torch.randn((bs, w_dim), device=device)

    out1, attens1 = gat(x, w)
    print(out1.shape, attens1.shape)

    gt = GT(emb_dim_in, emb_dim_out, adj_matrix,
            enc_dim=3, in_emb_dim=None, num_head=2, dropout=0.0, depth=2, pre_norm=True, sn=True, sln=True,
            w_dim=w_dim, output_atten=True).to(device)
    out2, attens2 = gt(x, w)
    print(out2.shape, attens2[0].shape)

    plt.imshow(attens1[0].detach().cpu().numpy())
    plt.colorbar()
    plt.show()
    plt.imshow(attens2[0][0].detach().cpu().numpy())
    plt.colorbar()
    plt.show()

