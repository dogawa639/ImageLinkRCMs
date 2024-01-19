if __name__ == "__main__":
    import configparser
    import torch
    from torchinfo import summary
    import matplotlib.pyplot as plt
    from preprocessing.network import *
    from utility import *
    from learning.discriminator import *

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

    bs = 3
    input_channel = 5
    output_channel = 2
    emb_dim = 2
    w_dim = 5
    enc_dim = 3
    w = torch.randn(bs, w_dim).to(device)

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    f = nw_data.feature_num
    c = nw_data.context_feature_num

    if CNN:
        dis = CNNDis(nw_data, output_channel, gamma=0.9, max_num=40, sln=True, w_dim=w_dim, ext_coeff=1.0).to(device)

        dis.train()
        inputs = torch.randn(bs, f+c, 3, 3).to(device)
        pi = torch.randn(bs, output_channel, 3, 3).to(device)
        dis.set_w(w)
        out = dis(inputs, pi, i=0)
        out2 = dis(inputs, pi, i=1)
        print(out.shape, out2.shape)
        out12 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out12, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("cnn grad")
        for param in dis.parameters():
            print(param.grad)

    if GNN:
        dis = GNNDis(nw_data, emb_dim, output_channel,
                     gamma=0.9, in_emb_dim=int(emb_dim/2), num_head=3, dropout=0.0, depth=2, sn=True, sln=True, w_dim=w_dim, ext_coeff=1.0).to(device)
        dis.train()
        inputs = torch.randn(nw_data.link_num, nw_data.feature_num).to(device)
        pi = torch.randn(bs, output_channel, nw_data.link_num, nw_data.link_num).to(device)
        dis.set_w(w)
        out = dis(inputs, pi, i=0)
        print(out.shape)
        out2 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out2, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("gnn grad")
        for param in dis.parameters():
            print(param.grad)

    if UNet:
        # input: (bs, total_feature, 2d+1, 2d+1)
        # positions: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # pis: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # f_val, util, val: (bs, 2d+1, 2d+1)

        feature_num = 3
        context_num = 1
        output_channel = 2
        d = 16
        dis = UNetDis(feature_num, context_num, output_channel).to(device)
        input = torch.randn(bs, feature_num + context_num, 2 * d + 1, 2 * d + 1).to(device)
        positions = torch.randn(bs, 1, output_channel, 2 * d + 1, 2 * d + 1).to(device)
        pis = torch.randn(bs, 1, output_channel, 2 * d + 1, 2 * d + 1).to(device)

        dis.train()
        out = dis(input, positions, pis)
        print(out.shape)
        plt.imshow(out.detach().cpu().numpy()[0, :, :])
        plt.show()

        out2 = torch.randn_like(out)
        loss = torch.norm(torch.max(out - out2, torch.tensor(-50.0, device=device)))
        loss.backward()
        print("unet grad")
        for param in dis.parameters():
            print(param.grad)

        print(summary(model=dis.to("cpu"), input_size=[input.shape, positions.shape, pis.shape]))

