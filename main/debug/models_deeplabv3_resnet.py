if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import os
    import json

    import configparser

    from preprocessing.dataset import *
    from models.deeplabv3 import deeplabv3_resnet50, resnet50, classifier_resnet50, ResNet50_Weights

    from learning.encoder import *
    from utility_kalmanfilter import *
    from logger import Logger

    CONFIG = "../../config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")

    read_general = config["GENERAL"]
    device = read_general["device"]

    read_data = config["DATA"]
    node_path = read_data["node_path"]  # csv
    link_path = read_data["link_path"]  # csv
    image_data_path = read_data["image_data_path"]  # json
    image_data_dir = read_data["satellite_image_datadir"]  # dir
    onehot_data_path = read_data["onehot_data_path"]  # json
    onehot_data_dir = read_data["onehot_image_datadir"]  # dir
    streetview_dir = read_data["streetview_dir"]  # dir
    streetview_data_path = read_data["streetview_data_path"]  # json

    read_save = config["SAVE"]
    debug_dir = read_save["debug_dir"]
    log_dir = read_save["log_dir"]

    link_prop_path = os.path.join(onehot_data_dir, "link_prop.csv")  # csv

    SAT2LU = False  # XHImageDataset
    ST2LU = False  # StreetViewDataset
    SATST2LU = True  # StreetViewXDataset

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

    if SAT2LU:
        xhimage_dataset = XHImageDataset(base_dirs_x, base_dirs_h, **kwargs)
        train_data, val_data, test_data = xhimage_dataset.split_into((0.7, 0.1, 0.2))
        train_dataloader = DataLoader(train_data, **loader_kwargs)
        val_dataloader = DataLoader(val_data, **loader_kwargs)
        test_dataloader = DataLoader(test_data, **loader_kwargs)

        model = deeplabv3_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        logger = Logger(os.path.join(log_dir, "sat2lu.json"), CONFIG)
        for epoch in range(3):
            model.train()
            tmp_loss = 0.0
            # data_tuple_x, data_tuple_h
            # data_tuple[i]: (img_tensor_x, transformed_x, mask_x, idx_x)
            for batch_x, batch_h in train_dataloader:
                batch_x[0] = batch_x[0].to(device)
                batch_h[0] = batch_h[0].to(device)
                optimizer.zero_grad()
                out = model(batch_x[0])["out"]  # (N, C, H, W)
                loss = torch.sum(loss_fn(out, batch_h[0]))
                loss.backward()
                optimizer.step()
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("train_loss", tmp_loss / len(train_dataloader))

            model.eval()
            tmp_loss = 0.0
            # (img_tensor_x, transformed_x, mask_x, idx_x)
            for batch_x, batch_h in val_dataloader:
                batch_x[0] = batch_x[0].to(device)
                batch_h[0] = batch_h[0].to(device)
                out = model(batch_x[0])["out"]  # (N, C, H, W)
                loss = torch.sum(loss_fn(out, batch_h[0]))
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("val_loss", tmp_loss / len(val_dataloader))

        logger.close()

    if ST2LU:
        stimage_dataset = StreetViewDataset(streetview_data_path, link_prop_path, **kwargs)  # link_prop is ratio of classes
        train_data, val_data, test_data = stimage_dataset.split_into((0.7, 0.1, 0.2))
        train_dataloader = DataLoader(train_data, **loader_kwargs)
        val_dataloader = DataLoader(val_data, **loader_kwargs)
        test_dataloader = DataLoader(test_data, **loader_kwargs)

        model = classifier_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes, num_backbones=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        logger = Logger(os.path.join(log_dir, "st2lu.json"), CONFIG)
        for epoch in range(3):
            model.train()
            tmp_loss = 0.0
            # data_tuple, y_tensor
            # data_tuple[i]: (img_tensor_x, transformed_x, mask_x, idx_x)
            for data_tuple, y_tensor in train_dataloader:
                img_tensors = [x[0].to(device) for x in data_tuple]
                y_tensor = y_tensor.to(device)
                optimizer.zero_grad()
                out = model(img_tensors)["out"]  # (N, C, H, W)
                loss = torch.sum(loss_fn(out, y_tensor))
                loss.backward()
                optimizer.step()
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("train_loss", tmp_loss / len(train_dataloader))

            model.eval()
            tmp_loss = 0.0
            # (img_tensor_x, transformed_x, mask_x, idx_x)
            for data_tuple, y_tensor in val_dataloader:
                img_tensors = [x[0].to(device) for x in data_tuple]
                y_tensor = y_tensor.to(device)
                out = model(img_tensors)["out"]  # (N, C, H, W)
                loss = torch.sum(loss_fn(out, y_tensor))
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("val_loss", tmp_loss / len(val_dataloader))

        logger.close()

    if SATST2LU:
        satstimage_dataset = StreetViewXDataset(streetview_data_path, base_dirs_x, link_prop_path, **kwargs)
        train_data, val_data, test_data = satstimage_dataset.split_into((0.7, 0.1, 0.2))
        train_dataloader = DataLoader(train_data, **loader_kwargs)
        val_dataloader = DataLoader(val_data, **loader_kwargs)
        test_dataloader = DataLoader(test_data, **loader_kwargs)

        model = classifier_resnet50(weights_backbone=ResNet50_Weights.DEFAULT, num_classes=num_classes, num_backbones=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

        logger = Logger(os.path.join(log_dir, "satst2lu.json"), CONFIG)
        for epoch in range(3):
            model.train()
            tmp_loss = 0.0
            # data_tuple, data_tuple_x, y_tensor
            # data_tuple[i]: (img_tensor_x, transformed_x, mask_x, idx_x)
            for data_tuple, data_tuple_x, y_tensor in train_dataloader:
                img_tensors = [x[0].to(device) for x in data_tuple]
                img_tensors_x = [x[0].to(device) for x in data_tuple_x]
                y_tensor = y_tensor.to(device)
                optimizer.zero_grad()
                out = model([img_tensors, img_tensors_x])["out"]
                loss = torch.sum(loss_fn(out, y_tensor))
                loss.backward()
                optimizer.step()
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("train_loss", tmp_loss / len(train_dataloader))

            model.eval()
            tmp_loss = 0.0
            for data_tuple, data_tuple_x, y_tensor in val_dataloader:
                img_tensors = [x[0].to(device) for x in data_tuple]
                img_tensors_x = [x[0].to(device) for x in data_tuple_x]
                y_tensor = y_tensor.to(device)
                out = model([img_tensors, img_tensors_x])["out"]
                loss = torch.sum(loss_fn(out, y_tensor))
                tmp_loss += loss.clone().cpu().detach().item()
            logger.add_log("val_loss", tmp_loss / len(val_dataloader))

        logger.close()







