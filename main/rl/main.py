if __name__ == "__main__":
    import os
    import json
    import random
    import configparser
    from preprocessing.network import *
    from preprocessing.pp import *
    from models.general import FF
    from models.gnn import GAT, GT
    import torch
    from torch import tensor
    from torch import nn

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy
    import scipy.optimize
    from sklearn.preprocessing import StandardScaler
    from logger import Logger
    from others.rl import RL

    print(os.getcwd())

    device = "mps"

    CONFIG = "../../config/config_airl_cnn.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    pp_path_train = json.loads(read_data["pp_path_train"])
    pp_path_test = json.loads(read_data["pp_path_test"])

    read_save = config["SAVE"]
    result_dir = read_save["result_dir"]

    nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)

    pp_data = [PP(path, nw_data) for path in pp_path_train]
    pp_data_test = [PP(path, nw_data) for path in pp_path_test]
    seed = 100
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    # x_ped = np.array([-1.16, 0.29, 0.02, -0.51, 0.27], dtype=np.float32)  # 0: length / 100, -2: angle, -1: mix_term
    # x_car = np.array([-0.36, -0.13, -0.04, -0.64, 0.73], dtype=np.float32)  # 0: length / 100, -2: angle, -1: mix_term

    rl_ped = RL(pp_data[0], device=device, max_sample=1100)
    rl_car = RL(pp_data[1], device=device, max_sample=1100)

    rl_ped_test = RL(pp_data_test[0], device=device, max_sample=1100)
    rl_car_test = RL(pp_data_test[1], device=device, max_sample=1100)

    sample_ped = rl_ped.get_sample()
    sample_car = rl_car.get_sample()

    sample_ped_test = rl_ped_test.get_sample()
    sample_car_test = rl_car_test.get_sample()

    sample_ped = {k: v for i, (k, v) in enumerate(sample_ped.items())}
    sample_car = {k: v for i, (k, v) in enumerate(sample_car.items())}

    sample_ped_test = {k: v for i, (k, v) in enumerate(sample_ped_test.items())}
    sample_car_test = {k: v for i, (k, v) in enumerate(sample_car_test.items())}

    print("train sample num:  ped  car")
    print(sum([len(v) for v in sample_ped.values()]), sum([len(v) for v in sample_car.values()]))
    print("test sample num:  ped  car")
    print(sum([len(v) for v in sample_ped_test.values()]), sum([len(v) for v in sample_car_test.values()]))

    z_test = False
    interaction_estimation = True
    v_ml_test = False

    if z_test:
        x = np.array([-0.95709107, -0.86090123, -0.50018837, 0.48622069])
        m = rl_ped.comp_m(x)
        z_dict = rl_ped.comp_z(m, sample_ped)

        k = list(z_dict.keys())[0]
        # z = Mz+dt  ->  z = (I-M)^(-1)dt
        z_pred = z_dict[k]
        v_pred = np.log(z_pred + (z_pred <= 0))
        dt = np.zeros((rl_ped.n, 1), dtype=np.float32)
        dt[k, 0] = 1.0
        m2 = m.copy()
        m2[k, :] = 0.0
        z_true = np.linalg.pinv(np.eye(rl_ped.n, dtype=np.float32) - m2) @ dt
        z_true = z_true.flatten()
        v_true = np.log(z_true + (z_true <= 0))
        u, s, vh = np.linalg.svd(np.eye(rl_ped.n, dtype=np.float32) - m2)
        print(s[0] / s[-1])

        # write_2d_ndarray("/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/m.txt", m)

        fig = plt.figure()
        plt.scatter(z_true, z_pred)
        plt.xlabel("z_true")
        plt.ylabel("z_pred")
        plt.plot([0, 1], [0, 1], color="red", linestyle="dashed")
        plt.show()

        d_node = nw_data.nodes[nw_data.edges[nw_data.lids[k]].end.id]
        xlim = [-69400, -69000]
        ylim = [92850, 93300]
        fig, ax = plt.subplots(dpi=200)
        ax = nw_data.plot_link(nw_data.lids, v_true, title="v_true", ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()
        fig, ax = plt.subplots(dpi=200)
        ax = nw_data.plot_link(nw_data.lids, v_pred, title="v_pred", ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()

        lid2idx = {lid: i for i, lid in enumerate(nw_data.lids)}
        prob = np.zeros(len(nw_data.lids), dtype=np.float32)
        for node in nw_data.nodes.values():
            idxs = []
            zs = []
            v_total = 0.0
            for link in node.downstream:
                idxs.append(lid2idx[link.id])
                zs.append(z_pred[lid2idx[link.id]])
                v_total += z_pred[lid2idx[link.id]]
            if v_total > 0.0:
                prob[idxs] = np.array(zs) / v_total
        fig, ax = plt.subplots()
        ax = nw_data.plot_link(nw_data.lids, prob, title="prob", xlim=xlim, ylim=ylim, ax=ax)
        ax.scatter([d_node.x], [d_node.y], color="red")
        plt.show()

        paths = rl_ped.forward(x, 10, d_node_id=d_node.id)
        nw_data.plot_paths(paths)

    if interaction_estimation:

        def fn_ll_ped(x, rl_ped, sample_ped):
            # x = np.concatenate([x_ped, x_car])
            mix_term_ped = rl_car.comp_m(x[rl_ped.prop_len + 2:-1])
            return -rl_ped.get_ll(x[:rl_ped.prop_len + 2], sample_ped, mix_term=mix_term_ped, use_bias=True)


        def fn_ll_car(x, rl_car, sample_car):
            mix_term_car = rl_ped.comp_m(x[:rl_car.prop_len + 1], use_bias=True)
            return -rl_car.get_ll(x[rl_car.prop_len + 2:], sample_car, mix_term=mix_term_car)


        x = np.zeros((rl_ped.prop_len + 2) + (rl_car.prop_len + 1), dtype=np.float32)
        dx = 100
        x_ped_len = rl_ped.prop_len + 2
        x_car_len = rl_car.prop_len + 1
        log1 = Logger(os.path.join(result_dir, "rl/log.txt"), "")
        tol = 1e-6
        cnt = 0
        while dx > 1e-2:
            pre_x = x.copy()
            x0 = np.zeros(x_ped_len, dtype=np.float32)
            fn_ped = lambda x_ped: fn_ll_ped(np.concatenate((x_ped, x[x_ped_len:])), rl_ped, sample_ped) + 0.001 * np.sum(np.power(x_ped, 2))
            fprime_ped = lambda x: scipy.optimize.approx_fprime(x, fn_ped)
            opt_kwargs_ped = {"method": "BFGS", "jac": fprime_ped, "tol": tol}
            res_ped = scipy.optimize.minimize(fn_ped, x0, **opt_kwargs_ped)
            x[:x_ped_len] = res_ped.x
            log1.add_log("x", x)
            log1.add_log("fn", res_ped.fun)

            x0 = np.zeros(x_car_len, dtype=np.float32)
            fn_car = lambda x_car: fn_ll_car(np.concatenate((x[:x_ped_len], x_car)), rl_car, sample_car) + 0.001 * np.sum(np.power(x_car, 2))
            fprime_car = lambda x: scipy.optimize.approx_fprime(x, fn_car)
            opt_kwargs_car = {"method": "BFGS", "jac": fprime_car, "tol": tol}
            res_car = scipy.optimize.minimize(fn_car, x0, **opt_kwargs_car)
            x[x_ped_len:] = res_car.x
            log1.add_log("x", x)
            log1.add_log("fn", res_car.fun)

            dx = np.sqrt(np.mean(np.power(x - pre_x, 2)))
            log1.add_log("dx", dx)

            cnt += 1
        print("finish estimation.")
        log1.close()

        x_ped = x[:x_ped_len]
        x_car = x[x_ped_len:]

        mix_term_car = rl_ped.comp_m(x_ped[:-1], use_bias=True)
        mix_term_ped = rl_car.comp_m(x_car[:-1])

        rl_ped.write_result(res_ped, sample_ped_test, mix_term=mix_term_ped, use_bias=True,
                            file=os.path.join(result_dir, "rl", "rl_ped_single.txt"))
        rl_car.write_result(res_car, sample_car_test, mix_term=mix_term_car, file=os.path.join(result_dir, "rl", "rl_car_single.txt"))

    if v_ml_test:
        x_ped = [-0.95709107, -0.86090123, -0.50018837, 0.48622069, -0.85128209, -0.77433027]  # x[:x_ped_len]
        x_car = [-0.31941202, -0.33455477, -0.05158337, 0.25108421, -0.01120311]  # x[x_ped_len:]

        M_ped = rl_ped.comp_m(x_ped, mix_term=mix_term_ped, use_bias=True)
        M_car = rl_car.comp_m(x_car, mix_term=mix_term_car)

        z_ped = rl_ped.comp_z(M_ped, sample_ped)
        z_car = rl_car.comp_z(M_car, sample_car)

        feature_dict = dict()
        input_channel = nw_data.feature_num + nw_data.context_feature_num
        # input_channel = nw_data.context_feature_num
        for d_idx in z_ped.keys():
            d_link_id = rl_ped.nw_data.lids[d_idx]
            d_node_id = rl_ped.nw_data.edges[d_link_id].end.id
            all_features = nw_data.get_all_feature_matrix(d_node_id, normalize=True)
            feature_dict[d_idx] = all_features

        value_model1 = GT(input_channel, input_channel,
                          tensor(nw_data.edge_graph > 0, dtype=torch.float32, device=device), output_atten=False,
                          sn=False, depth=3, atten_fn="dense").to(device)
        value_model2 = FF(input_channel, 1, act_fn=lambda x: torch.exp(x)).to(device)
        value_model = nn.Sequential(value_model1, value_model2)
        optimizer = torch.optim.Adam(value_model.parameters(), lr=0.001)
        sc = StandardScaler()
        losses = []
        for epoch in range(100):
            value_model.train()
            tmp = []
            for i, d_idx in enumerate(z_ped.keys()):
                y = tensor(z_ped[d_idx].reshape(-1, 1), device=device)
                x = tensor(feature_dict[d_idx], device=device)
                x = torch.unsqueeze(x, 0)

                y2 = value_model(x).reshape(-1, 1)
                loss = torch.mean(torch.pow(y - y2, 2))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tmp.append(loss.detach().cpu().item())
            losses.append(np.mean(tmp))

        value_model.eval()
        errors = []
        for i, d_idx in enumerate(z_ped.keys()):
            y = tensor(z_ped[d_idx].reshape(-1, 1), device=device)
            x = tensor(feature_dict[d_idx], device=device)
            x = torch.unsqueeze(x, 0)
            y2 = value_model(x).reshape(-1, 1)
            errors.append((y - y2).cpu().detach().numpy().flatten().tolist())
        print(y, y2)
        print(x.shape, y.shape, y2.shape)
        errors = sum(errors, [])
        # write_1d_array("/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/y.txt", np.array(y.detach().cpu().numpy().flatten()))
        # write_1d_array("/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/y2.txt", np.array(y2.detach().cpu().numpy().flatten()))
        # write_1d_array("/Users/dogawa/PycharmProjects/ImageLinkRCM/debug/model/errors.txt", np.array(errors))
        plt.hist(errors, bins=30)
        plt.xlabel("error")
        plt.show()
        plt.hist(y2.detach().cpu().numpy().flatten(), bins=30)
        plt.xlabel("y2")
        plt.show()
        plt.scatter(y.detach().cpu().numpy().flatten(), y2.detach().cpu().numpy().flatten())
        m1, M1 = y.detach().cpu().numpy().flatten().min(), y.detach().cpu().numpy().flatten().max()
        m, M = y2.detach().cpu().numpy().flatten().min(), y2.detach().cpu().numpy().flatten().max()
        m, M = 0.0, 0.1
        plt.xlim([m, M])
        plt.ylim([m, M])
        plt.plot([m, M], [m, M], color="red", linestyle="dashed")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.show()
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
