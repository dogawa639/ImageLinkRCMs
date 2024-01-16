if __name__ == "__main__":
    import torch
    from models.general import *
    from models.cnn import *

    b, c, h, w = 2, 3, 4, 5
    w_dim = 6
    device = "mps"
    inputs = torch.randn(b, c, h, w).to(device)
    ws = torch.randn(b, 6).to(device)

    cnn3 = CNN3x3((h, w), [c, c*2, c*4], act_fn=lambda x : -softplus(x), residual=True, sn=True, sln=True, w_dim=w_dim).to(device)
    out3 = cnn3(inputs, ws)

    cnn1 = CNN1x1((h, w), [c, c*2, c*4], act_fn=lambda x : -softplus(x), sn=False, sln=True, w_dim=w_dim).to(device)
    out1 = cnn1(inputs, ws)

    print(out3.shape, out1.shape)
