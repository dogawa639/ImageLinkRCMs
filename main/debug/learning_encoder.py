if __name__ == "__main__":
    import torch
    from learning.encoder import *

    patch_size = (3, 256, 256)
    bs = 10
    emb_dim = 5
    num_source = 1
    w_dim = 6
    device = "mps"
    inputs = torch.randn(bs, 1000).to(device)
    w = torch.randn(bs, w_dim).to(device)

    cnn = CNNEnc(patch_size, emb_dim=emb_dim, num_source=num_source, w_dim=w_dim).to(device)

    out = cnn(inputs, w=w, source_i=0)
    print(out.shape)