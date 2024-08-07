if __name__ == "__main__":
    import configparser
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torchinfo import summary
    from preprocessing.network import *
    from learning.generator import *


    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_general = config["GENERAL"]
    device = read_general["device"]

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    CNN = False
    GNN = False
    UNet = True

    bs = 10
    output_channel = 2
    emb_dim = 4
    enc_dim = 3
    w_dim = 5
    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    f = nw_data.feature_num
    c = nw_data.context_feature_num
    print(f"f: {f}, c: {c}")

    if CNN:
        gen = CNNGen(nw_data, output_channel, w_dim=w_dim).to(device)

        gen.train()
        inputs = torch.randn(bs, f + c, 3, 3).to(device)
        w = torch.randn(w_dim).to(device)
        gen.set_w(w)
        out = gen(inputs, i=0)
        out2 = gen(inputs, i=1)

        out12 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out12, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("cnn grad")
        for param in gen.parameters():
            print(param.grad)

        w = torch.randn(2 * bs, w_dim).to(device)
        out3 = gen.generate([bs, bs], w=w)
        print(out.shape, out2.shape)

        out_exp = torch.exp(out - out.max())
        out = out_exp / out_exp.sum((1, 2), keepdim=True)
        plt.imshow(out[0].detach().cpu().numpy())
        plt.colorbar()
        plt.show()

    if GNN:
        gen = GNNGen(nw_data, emb_dim, output_channel, enc_dim, in_emb_dim=int(emb_dim / 2), num_head=2, dropout=0.1,
                     depth=2, pre_norm=False, sn=True, sln=True, w_dim=w_dim).to(device)

        gen.train()
        inputs = torch.randn(bs, len(nw_data.lids), nw_data.feature_num).to(device)
        w = torch.randn(w_dim).to(device)
        gen.set_w(w)
        out = gen(inputs, i=0)
        out2 = gen(inputs, i=1)
        print(out.shape, out2.shape)

        out12 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out12, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("gnn grad")
        for param in gen.parameters():
            print(param.grad)

        out = F.softmax(out, dim=2)
        plt.imshow(out[0].detach().cpu().numpy())
        plt.colorbar()
        plt.show()

    if UNet:
        feature_num = 3
        context_num = 1
        d = 16
        gen = UNetGen(feature_num, context_num).to(device)
        input = torch.randn(bs, feature_num + context_num, 2*d+1, 2*d+1).to(device)

        gen.train()
        out = gen(input)
        print(out.shape)
        plt.imshow(out.detach().cpu().numpy()[0, :, :])
        plt.show()

        out2 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out2, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("unet grad")
        for param in gen.parameters():
            print(param.grad)

        print(summary(model=gen.to("cpu"), input_size=(bs, feature_num + context_num, 2 * d + 1, 2 * d + 1)))

