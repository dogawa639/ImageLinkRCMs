if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    import json
    import configparser
    from preprocessing.dataset import *
    from models.deeplabv3 import deeplabv3_resnet50, ResNet50_Weights
    from utility import *
    from logger import Logger

    CONFIG = "../../config/config_all.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_general = config["GENERAL"]
    device = read_general["device"]

    read_data = config["DATA"]
    node_path = read_data["node_path"]  # csv
    link_path = read_data["link_path"]  # csv
    link_prop_path = read_data["link_prop_path"]  # csv
    onehot_data_path = read_data["onehot_data_path"]  # json
    onehot_data_dir = read_data["onehot_image_datadir"]  # dir

    read_feature = config["FEATURE"]
    num_classes = int(read_feature["max_class_num"])  # class_num (including other class, class_num = 0)

    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    log_dir = read_save["log_dir"]

    TRINING = True
    TESTING = True
    EARLY_STOP = True
    SAVE_MODEL = True

    case_name = "lu2lu_" + ("train" if TRINING else "") + ("test" if TESTING else "")

    image_data = load_json(onehot_data_path)
    base_dirs_x = [os.path.join(onehot_data_dir, x["name"]) for x in image_data]
    one_hot_data = load_json(onehot_data_path)
    base_dirs_h = [os.path.join(onehot_data_dir, x["name"]) for x in one_hot_data]

    l_coeff = 0.001

    kwargs = {"expansion": 1,
              "crop": True,
              "affine": True,
              "transform_coincide": True,
              "flip": True}
    loader_kwargs = {"batch_size": 64,
                     "shuffle": True,
                     "num_workers": 2,
                     "pin_memory": True,
                     "drop_last": True}

    xhimage_dataset = XHImageDataset(base_dirs_x, base_dirs_h, **kwargs)
    train_data, val_data, test_data = xhimage_dataset.split_into((0.2, 0.02, 0.02))
    train_data_num = len(train_data)
    val_data_num = len(val_data)
    test_data_num = len(test_data)
    print(f"train_data_num: {train_data_num}, val_data_num: {val_data_num}, test_data_num: {test_data_num}")

    train_dataloader = DataLoader(train_data, **loader_kwargs)
    val_dataloader = DataLoader(val_data, **loader_kwargs)
    test_dataloader = DataLoader(test_data, **loader_kwargs)

    model = deeplabv3_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum", label_smoothing=1./num_classes/100)

    logger = Logger(os.path.join(log_dir, f"{case_name}.json"), CONFIG)
    model_path = os.path.join(model_dir, "lu2lu.pth")
    img_shape = None  # (h,w)
    # train, val
    if TRINING:
        min_loss = np.inf
        stop_count = 0
        for epoch in range(500):
            t1 = time.perf_counter()
            model.train()
            tmp_loss = 0.0
            tmp_loss_total = 0.0
            # data_tuple_x, data_tuple_h
            # data_tuple[i]: (img_tensor_x, transformed_x, mask_x, idx_x)
            for batch_x, batch_h in train_dataloader:
                batch_x[1] = batch_x[1].to(device)
                batch_h[1] = batch_h[1].to(device)
                batch_x[2] = batch_x[2].to(device)
                optimizer.zero_grad()
                out = model(batch_x[1])["out"]  # (N, C, H, W)
                out = out * batch_x[2].unsqueeze(1)
                loss = loss_fn(out, batch_h[1])
                l_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                for w in model.parameters():
                    l_loss = l_loss + torch.sum(torch.square(w))
                loss_total = loss + l_coeff / 1.0 ** epoch * l_loss
                loss_total.backward()
                optimizer.step()
                tmp_loss += loss.clone().cpu().detach().item()
                tmp_loss_total += loss_total.clone().cpu().detach().item()
            train_loss = tmp_loss / train_data_num
            train_loss_total = tmp_loss_total / train_data_num
            logger.add_log("train_loss", train_loss)
            logger.add_log("train_loss_total", train_loss_total)

            model.eval()
            tmp_loss = 0.0
            tmp_loss_total = 0.0
            # (img_tensor_x, transformed_x, mask_x, idx_x)
            for batch_x, batch_h in val_dataloader:
                batch_x[1] = batch_x[1].to(device)
                batch_h[1] = batch_h[1].to(device)
                batch_x[2] = batch_x[2].to(device)
                out = model(batch_x[1])["out"]  # (N, C, H, W)
                out = out * batch_x[2].unsqueeze(1)
                loss = loss_fn(out, batch_h[1])
                l_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                for w in model.parameters():
                    l_loss = l_loss + torch.sum(torch.square(w))
                loss_total = loss + l_coeff / 1.0 ** epoch * l_loss
                tmp_loss += loss.clone().cpu().detach().item()
                tmp_loss_total += loss_total.clone().cpu().detach().item()
                if img_shape is None:
                    img_shape = batch_x[1].shape[2:]
            val_loss = tmp_loss / val_data_num
            val_loss_total = tmp_loss_total / val_data_num
            logger.add_log("val_loss", val_loss)
            logger.add_log("val_loss_total", val_loss_total)

            if EARLY_STOP:
                if val_loss_total < min_loss:
                    min_loss = val_loss_total
                    if SAVE_MODEL:
                        torch.save(model.cpu().state_dict(), model_path)
                        logger.add_prop("model", model_path)
                        model.to(device)
                    stop_count = 0
                else:
                    stop_count += 1
                if epoch > 50 and stop_count >= 10:
                    print("Early Stopping.")
                    break
            t2 = time.perf_counter()
            print(f"epoch: {epoch}, train_loss: {train_loss / img_shape[0] / img_shape[1]}, val_loss: {val_loss / img_shape[0] / img_shape[1]}, time: {t2 - t1}")
        print(f"min_val_loss: {min_loss}")

    if TESTING:
        # test
        model = deeplabv3_resnet50(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        model.eval()
        tmp_loss = 0.0
        # (img_tensor_x, transformed_x, mask_x, idx_x)
        sample_num = 3
        cnt = 0
        img_shape = None  # (h,w)
        fig1 = plt.figure()
        fig2 = plt.figure()
        for i, (batch_x, batch_h) in enumerate(test_dataloader):
            batch_x[0] = batch_x[0].to(device)
            batch_h[0] = batch_h[0].to(device)
            out = model(batch_x[0])["out"]  # (N, C, H, W)
            loss = loss_fn(out, batch_h[0])
            tmp_loss += loss.clone().cpu().detach().item()
            if img_shape is None:
                img_shape = batch_x[1].shape[2:]
            if cnt < sample_num:
                for j in range(len(batch_x[0])):
                    cnt += 1
                    ax = fig1.add_subplot(sample_num, 1, cnt)
                    mi = batch_x[0][j].cpu().detach().numpy().min()
                    ma = batch_x[0][j].cpu().detach().numpy().max()
                    ax.imshow(((batch_x[0][j].cpu().detach().numpy().transpose(1, 2, 0) - mi) / (ma - mi) * 255).astype(np.uint8))
                    ax.set_title(f"original_{cnt}")
                    ax = fig2.add_subplot(sample_num, 2, cnt * 2 - 1)
                    ax.imshow(out[j].argmax(0).cpu().detach().numpy())
                    ax.set_title(f"predicted_{cnt}")
                    ax = fig2.add_subplot(sample_num, 2, cnt * 2)
                    ax.imshow(batch_h[0][j].cpu().detach().numpy())
                    ax.set_title(f"ground_truth_{cnt}")
                    if cnt >= sample_num:
                        break
        test_loss = tmp_loss / test_data_num
        print(f"test_loss: {test_loss / img_shape[0] / img_shape[1]}")
        logger.add_prop("test_loss", test_loss)

        fig1.savefig(os.path.join(log_dir, f"{case_name}_original.png"))
        fig2.savefig(os.path.join(log_dir, f"{case_name}_predicted.png"))
        plt.show()

    logger.save_fig(os.path.join(log_dir, f"{case_name}.png"))
    logger.close()





