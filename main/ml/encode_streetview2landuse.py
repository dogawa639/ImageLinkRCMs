if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import numpy as np
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import time
    import os
    import json
    import configparser
    from preprocessing.dataset import *
    from models.deeplabv3 import deeplabv3_resnet50, classifier_resnet50, ResNet50_Weights
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
    image_data_path = read_data["image_data_path"]  # json
    image_data_dir = read_data["satellite_image_datadir"]  # dir
    onehot_data_path = read_data["onehot_data_path"]  # json
    onehot_data_dir = read_data["onehot_image_datadir"]  # dir
    streetview_dir = read_data["streetview_dir"]  # dir
    streetview_data_path = read_data["streetview_data_path"]  # json

    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    log_dir = read_save["log_dir"]

    TRINING = True
    TESTING = True
    EARLY_STOP = True
    SAVE_MODEL = True

    case_name = "st2lu_" + ("train" if TRINING else "") + ("test" if TESTING else "")

    image_data = load_json(image_data_path)
    base_dirs_x = [os.path.join(image_data_dir, x["name"]) for x in image_data]
    one_hot_data = load_json(onehot_data_path)
    base_dirs_h = [os.path.join(onehot_data_dir, x["name"]) for x in one_hot_data]

    num_classes = 10  # class_num (including other class, class_num = -1)

    kwargs = {"expansion": 2,
              "crop": True,
              "affine": True,
              "transform_coincide": True,
              "flip": True}
    loader_kwargs = {"batch_size": 4,
                     "shuffle": True,
                     "num_workers": 2,
                     "pin_memory": True}

    stimage_dataset = StreetViewDataset(streetview_data_path, link_prop_path, **kwargs)
    train_data, val_data, test_data = stimage_dataset.split_into((0.03, 0.01, 0.02))
    train_dataloader = DataLoader(train_data, **loader_kwargs)
    val_dataloader = DataLoader(val_data, **loader_kwargs)
    test_dataloader = DataLoader(test_data, **loader_kwargs)

    model = classifier_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes, num_backbones=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def loss_fn(out, y):
        # out: (batch_size, num_classes)
        # y: (batch_size, num_classes), probability
        out = F.softmax(out, dim=1)
        return torch.sum((out - y) ** 2)

    logger = Logger(os.path.join(log_dir, "st2lu.json"), CONFIG)
    model_path = os.path.join(model_dir, "st2lu.pth")
    if TRINING:
        # train, val
        min_loss = np.inf
        stop_count = 0
        for epoch in range(100):
            t1 = time.perf_counter()
            model.train()
            tmp_loss = 0.0
            # data_tuple, y_tensor
            # data_tuple[i]: (img_tensor_x, transformed_x, mask_x, idx_x)
            for data_tuple, y_tensor in train_dataloader:
                transformed_x = [x[1].to(device) for x in data_tuple]
                y_tensor = y_tensor.to(device)
                optimizer.zero_grad()
                out = model(transformed_x)["out"]  # (N, C)
                loss = loss_fn(out, y_tensor)
                l1_loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                for w in model.parameters():
                    l1_loss = l1_loss + torch.sum(torch.abs(w))
                loss_total = loss + 0.0001 / 2.0 ** epoch * l1_loss
                loss_total.backward()
                optimizer.step()
                tmp_loss += loss.clone().cpu().detach().item()
            train_loss = tmp_loss / len(train_dataloader)
            logger.add_log("train_loss", train_loss)

            model.eval()
            tmp_loss = 0.0
            # (img_tensor_x, transformed_x, mask_x, idx_x)
            for data_tuple, y_tensor in val_dataloader:
                transformed_x = [x[1].to(device) for x in data_tuple]
                y_tensor = y_tensor.to(device)
                out = model(transformed_x)["out"]  # (N, C)
                loss = loss_fn(out, y_tensor)
                tmp_loss += loss.clone().cpu().detach().item()
            val_loss = tmp_loss / len(val_dataloader)
            logger.add_log("val_loss", val_loss)

            if EARLY_STOP:
                if val_loss < min_loss:
                    min_loss = val_loss
                    if SAVE_MODEL:
                        torch.save(model.cpu().state_dict(), model_path)
                        logger.add_prop("model", model_path)
                        model.to(device)
                    stop_count = 0
                else:
                    stop_count += 1
                if stop_count >= 5:
                    break
            t2 = time.perf_counter()
            print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, time: {t2 - t1}")
        print(f"min_val_loss: {min_loss}")

    if TESTING:
        # test
        model = classifier_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes, num_backbones=1)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        model.eval()
        tmp_loss = 0.0
        predicted = None
        true_values = None
        # (img_tensor_x, transformed_x, mask_x, idx_x)
        for data_tuple, y_tensor in test_dataloader:
            transformed_x = [x[1].to(device) for x in data_tuple]
            y_tensor = y_tensor.to(device)
            out = model(transformed_x)["out"]  # (N, C)
            loss = loss_fn(out, y_tensor)
            tmp_loss += loss.clone().cpu().detach().item()

            out = F.softmax(out, dim=1)
            if predicted is None:
                predicted = out.clone().cpu().detach().numpy().T
                true_values = y_tensor.clone().cpu().detach().numpy().T
            else:
                predicted = np.concatenate((predicted, out.clone().cpu().detach().numpy().T), axis=1)
                true_values = np.concatenate((true_values, y_tensor.clone().cpu().detach().numpy().T), axis=1)

        fig = plt.figure(figsize=(5, int(num_classes * 3.5)))
        for i in range(num_classes):
            ax = fig.add_subplot(num_classes, 1, i + 1)
            ax.scatter(true_values[i], predicted[i])
            ax.plot([0, max(predicted.max(), true_values.max())], [0, max(predicted.max(), true_values.max())],
                    color="red", linestyle="dashed")
            ax.set_xlabel("true_values")
            ax.set_ylabel("predicted")
            ax.set_aspect("equal")
        fig.savefig(os.path.join(log_dir, f"{case_name}_comparison.png"))
        plt.show()
        test_loss = tmp_loss / len(test_dataloader)
        print(f"test_loss: {test_loss}")
        logger.add_prop("test_loss", test_loss)

    logger.save_fig(os.path.join(log_dir, f"{case_name}.png"))
    logger.close()





