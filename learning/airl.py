import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import time
import os

from utility import *
from preprocessing.dataset import PatchDataset
from models.general import log
from logger import Logger


class AIRL:
    def __init__(self, generator, discriminator, use_index, datasets, model_dir, image_data=None, encoder=None, h_dim=10, emb_dim=10, f0=None,
                 hinge_loss=False, hinge_thresh=0.5, patch_size=256, device="cpu"):
        if hinge_thresh > 1.0 or hinge_thresh < 0.0:
            raise ValueError("hinge_thresh must be in [0, 1].")
        if encoder is not None and image_data is None:
            raise ValueError("image_data must be set when encoder is not None.")

        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.use_index = use_index  # whether inputs are the local link feature or not
        self.datasets = datasets  # list of Dataset  [train+test]
        self.model_dir = model_dir
        self.use_encoder = encoder is not None
        self.encoder = encoder
        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.f0 = f0  # h->w
        self.use_w = f0 is not None
        self.image_data = image_data
        self.hinge_loss = hinge_loss
        self.hinge_thresh = -log(tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False))
        self.patch_size = patch_size
        self.device = device

        self.sln = self.use_w

        self.link_num = len(self.datasets[0].nw_data.lids)
        self.output_channel = len(self.datasets)

        if self.use_encoder:
            self.encoder = self.encoder.to(device)
        if self.use_w:
            self.f0 = self.f0.to(device)
        if image_data is not None:
            self.image_data.set_voronoi(self.datasets[0].nw_data)
            self.image_data.compress_patches(self.encoder, patch_size, device=device)
    
    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
              train_ratio=0.8, d_epoch=5, lr_f0=0.01, lr_e=0.01, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file)  #loss_e,loss_g,loss_d,loss_e_val,loss_g_val,loss_d_val,criteria

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        if self.use_w:
            optimizer_0 = optim.Adam(self.f0.parameters(), lr=lr_f0)
        if self.use_encoder:
            optimizer_e = optim.Adam(self.encoder.parameters(), lr=lr_e)

        dataset_kargs = {"batch_size": batch_size, "shuffle": shuffle}
        dataloaders_real = [[DataLoader(tmp, **dataset_kargs)
                            for tmp in dataset.split_into((train_ratio, 1-train_ratio))]
                            for dataset in self.datasets]  # [train_dataloader, test_dataloader]
        
        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            epoch_loss_e = []
            epoch_loss_g = []
            epoch_loss_d = []
            for i, (dataloader_train, dataloader_val) in enumerate(dataloaders_real):  # transportation mode
                # batch
                # Grid: inputs, masks, next_links, link_idxs, h
                # Emb: global_state, adj_matrix, transition_matrix, h
                mode_loss_e = []
                mode_loss_g = []
                mode_loss_d = []
                epoch_loss_e_val = []
                epoch_loss_d_val = []
                epoch_loss_g_val = []

                self.train()
                for batch_real in dataloader_train:  # batch
                    # raw_data retain_grad=True
                    # Grid: (sum(links), 3, 3) corresponds to next_links
                    # Emb: (trip_num, link_num, link_num) sparse, corresponds to transition_matrix
                    batch_real = [tensor(tmp, dtype=torch.float32, device=self.device) for tmp in batch_real]
                    bs = batch_real[0].shape[0]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)

                    if self.use_encoder:
                        # append image_feature to the original feature
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        raw_data_fake = self.generator(batch_real[0], i, w)  # batch_fake requires_grad=False
                    else:
                        raw_data_fake = self.generator(batch_real[0], i)  # batch_fake requires_grad=False
                    batch_fake = self.datasets.get_fake_batch(batch_real, raw_data_fake)

                    for j in range(len(batch_real)):
                        batch_real[j] = batch_real[j].to(self.device)
                        batch_fake[j] = batch_fake[j].to(self.device)
                    raw_data_fake.to(self.device)

                    for j in range(d_epoch):
                        d_real = self.get_d(batch_real)  # discriminator inference performed inside
                        d_fake = self.get_d(batch_fake)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)
                        loss_e = loss_g + loss_d

                        # discriminator update
                        optimizer_d.zero_grad()
                        loss_d.backward(retain_graph=True)
                        optimizer_d.step()

                        mode_loss_d.append(loss_d.clone().detach().cpu().item())

                        if j == 0:
                            # f0 and encoder update
                            if self.use_w:
                                optimizer_0.zero_grad()
                            if self.use_encoder:
                                optimizer_e.zero_grad()

                            if self.use_w or self.use_encoder:
                                loss_e.backward(retain_graph=True)

                            if self.use_w:
                                optimizer_e.step()
                            if self.use_encoder:
                                optimizer_e.step()
                            mode_loss_e.append(loss_e.clone().detach().cpu().item())

                            # generator update
                            optimizer_g.zero_grad()
                            loss_g.backward(retain_graph=True)
                            optimizer_g.step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                # validation
                mode_loss_e_val = []
                mode_loss_g_val = []
                mode_loss_d_val = []

                self.eval()
                for batch_real in dataloader_val:
                    batch_real = [tensor(tmp, dtype=torch.float32, device=self.device) for tmp in batch_real]
                    raw_data_fake, batch_fake = self.generator.generate(batch_size, i)
                    d_real = self.get_d(batch_real)
                    d_fake = self.get_d(batch_fake)
                    loss_g, loss_d = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)

                    mode_loss_e_val.append(loss_e.clone().detach().cpu().item())
                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

            epoch_loss_e.append(np.mean(mode_loss_e))  # shape [num_modes]
            epoch_loss_g.append(np.mean(mode_loss_g))
            epoch_loss_d.append(np.mean(mode_loss_d))
            epoch_loss_e_val.append(np.mean(mode_loss_e_val))
            epoch_loss_g_val.append(np.mean(mode_loss_g_val))
            epoch_loss_d_val.append(np.mean(mode_loss_d_val))

            criteria = self.generator.get_criteria(self.datasets)
            if criteria < min_criteria:
                min_criteria = criteria
                self.generator.save(self.model_dir)
                self.discriminator.save(self.model_dir)

            log.add_log("loss_e", epoch_loss_e)
            log.add_log("loss_g", epoch_loss_g)
            log.add_log("loss_d", epoch_loss_d)
            log.add_log("loss_e_val", epoch_loss_e_val)
            log.add_log("loss_g_val", epoch_loss_g_val)
            log.add_log("loss_d_val", epoch_loss_d_val)
            log.add_log("criteria", criteria)

            t2 = time.perf_counter()
            print("epoch: {}, loss_e_val: {:.4f}, loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                e, epoch_loss_e_val[-1], epoch_loss_g_val[-1], epoch_loss_d_val[-1], criteria, t2 - t1))

        if image_file is not None:
            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            loss_g = np.array(log.data["loss_g"])
            loss_d = np.array(log.data["loss_d"])
            for i in range(loss_g.shape[1]):
                ax1.plot(loss_g[:, i], label="mode {}".format(i))
                ax2.plot(loss_d[:, i], label="mode {}".format(i))
            ax3.plot(log.data["criteria"])

            ax1.set_xlabel("epoch")
            ax2.set_xlabel("epoch")
            ax3.set_xlabel("epoch")
            ax1.set_ylabel("loss_g")
            ax2.set_ylabel("loss_d")
            ax3.set_ylabel("criteria")
            ax1.legend()
            ax2.legend()

            plt.savefig(image_file)
            plt.close()
        log.close()

    def loss(self, raw_data_fake, d_real, d_fake, hinge_loss=False):
        # raw_data_fake, d_fake: same shape, tensor
        # raw_data_fake, d: in [0, 1]
        loss_g = []
        loss_d = []
        for g_rate, d_r, d_f in zip(raw_data_fake, d_real, d_fake):
            if hinge_loss:
                l_g = -log(g_rate * d_f) + log(1. - g_rate * d_f)
                l_d = torch.max(-log(d_r), self.hinge_thresh) + torch.max(-log(1. - d_f), self.hinge_thresh)
            else:
                l_g = -log(g_rate * d_f) + log(1. - g_rate * d_f)
                l_d = -log(d_r) - log(1. - d_f)

            l_g = l_g.sum()
            l_d = l_d.sum()

            loss_g.append(l_g)
            loss_d.append(l_d)

        return loss_g, loss_d

    def get_d(self, batch):
        # batch
        # Grid: inputs, masks, next_links
        # Emb: global_state, adj_matrix, transition_matrix

        # d retain_grad=True
        # Grid: (sum(links), 3, 3) corresponds to next_links
        # Emb: (trip_num, link_num, link_num) sparse corresponds to transition_matrix
        input, mask, _ = batch
        d = self.discriminator(input)
        d = d * mask
        return d
    
    def cat_image_feature(self, batch, w=None):
        # w: (bs, w_dim)
        # index: (bs, 9)
        bs = batch[0].shape[0]
        image_feature = tensor(np.zeros((bs, self.link_num, self.emb_dim), dtype=np.float32), device=self.device)
        for i in range(len(self.image_data)):
            comp_feature = self.image_data.load_compressed_patches(i)
            comp_feature = tensor(comp_feature, dtype=torch.float32, device=self.device)
            image_feature = image_feature + self.encoder(comp_feature, source_i=i, w=w) / len(self.image_data)   # requires_grad=True (bs, link_num, emb_dim)

        # inputs: (bs, c, 3, 3) or (bs, link_num, feature_num)
        if self.use_index:
            # image_feature: tensor(link_num, emb_dim)
            # batch[3]: tensor(bs, 9)
            image_feature_tmp = image_feature[batch[3], :].transpose(1, 2).view(bs, -1, 3, 3)
            inputs = torch.cat((batch[0], image_feature_tmp), dim=1)
        else:
            inputs = torch.cat((batch[0], image_feature), dim=2)

        batch[0] = inputs
        return batch

    def generate(self, bs):
        # only for gnn
        if not self.use_encoder or self.use_index:
            raise Exception("This method is only for GNNEmb.")
        self.eval()
        hs = torch.randn(bs, self.h_dim, device=self.device)
        w = None
        if self.use_w:
            w = self.f0(hs)

        inputs = [tensor(self.real_data.feature_matrix, dtype=torch.float32, device=self.device).repeat(bs, 1, 1)]
        if self.use_encoder:
            # append image_feature to the original feature
            inputs = self.cat_image_feature(inputs, w=w)
        inputs = inputs[0]  # (bs, link_num, feature_num)
        adj_matrix = self.generator(inputs, None, w=w)  # (bs, oc, link_num, link_num)
        datasets = [self.datasets[i].get_dataset_from_adj(adj_matrix[:, i, :, :]) for i in range(self.output_channel)]
        return datasets

    def train(self):
        self.generator.train()
        self.discriminator.train()
        if self.use_w:
            self.f0.train()
        if self.use_encoder:
            self.encoder.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        if self.use_w:
            self.f0.eval()
        if self.use_encoder:
            self.encoder.eval()


# test
if __name__ == "__main__":
    import configparser
    import json
    from learning.generator import *
    from learning.discriminator import *
    from learning.encoder import *
    from learning.w_encoder import *
    from learning.util import get_models
    from preprocessing.network_processing import *
    from preprocessing.pp_processing import *
    from preprocessing.image_processing import *
    from preprocessing.dataset import *

    CONFIG = "/Users/dogawa/PycharmProjects/GANs/config/config_test.ini"
    config = configparser.ConfigParser()
    config.read(CONFIG, encoding="utf-8")
    # general
    read_general = config["GENERAL"]
    model_type = read_general["model_type"]  # cnn or gnn
    device = read_general["device"]
    # data
    read_data = config["DATA"]
    node_path = read_data["node_path"]
    link_path = read_data["link_path"]
    link_prop_path = read_data["link_prop_path"]

    pp_path = json.loads(read_data["pp_path"])  # list(str)
    image_data_path = read_data["image_data_path"] # str or None
    image_data_path = None if image_data_path == "null" else image_data_path
    # train setting
    read_train = config["TRAIN"]
    bs = int(read_train["bs"])  # int
    epoch = int(read_train["epoch"])  # int
    lr_g = float(read_train["lr_g"])  # float
    lr_d = float(read_train["lr_d"])  # float
    lr_f0 = float(read_train["lr_f0"])  # float
    lr_e = float(read_train["lr_e"])  # float
    shuffle = bool(read_train["shuffle"])  # bool
    train_ratio = float(read_train["train_ratio"])  # float
    d_epoch = int(read_train["d_epoch"])  # int
    # model setting
    read_model = config["MODELSETTING"]
    use_f0 = bool(read_model["use_f0"])  # bool
    emb_dim = int(read_model["emb_dim"])  # int
    in_emb_dim = json.loads(read_model["in_emb_dim"])  # int or None
    drop_out = float(read_model["drop_out"])  # float
    sn = bool(read_model["sn"])  # bool
    sln = bool(read_model["sln"])  # bool
    h_dim = int(read_model["h_dim"])  # int
    w_dim = int(read_model["w_dim"])  # int
    num_head = int(read_model["num_head"])  # int
    depth = int(read_model["depth"])  # int
    gamma = float(read_model["gamma"])   # float
    max_num = int(read_model["max_num"])  # int
    ext_coeff = float(read_model["ext_coeff"])  # float
    hinge_loss = bool(read_model["hinge_loss"])  # bool
    hinge_thresh = json.loads(read_model["hinge_thresh"])  # float or None
    patch_size = int(read_model["patch_size"])  # int
    # save setting
    read_save = config["SAVE"]
    model_dir = read_save["model_dir"]
    image_file = read_save["image_file"]  # str or None
    image_file = None if image_file == "null" else image_file

    # instance creation
    use_index = (model_type == "cnn")
    use_encoder = (image_data_path is not None)
    output_channel = len(pp_path)

    if model_type == "cnn":
        nw_data = NetworkCNN(node_path, link_path, link_prop_path=link_prop_path)
    else:
        nw_data = NetworkGNN(node_path, link_path, link_prop_path=link_prop_path)
    pp_list = [PP(ppath, nw_data) for ppath in pp_path]
    if model_type == "cnn":
        datasets = [GridDataset(pp, h_dim=h_dim) for pp in pp_list]
    else:
        datasets = [PPEmbedDataset(pp, h_dim=h_dim) for pp in pp_list]
    image_data = None
    num_source = 0
    if image_data_path is not None:
        image_data = ImageData(image_data_path)
        num_source = len(image_data)
    # model_names : [str] [discriminator, generator, (f0, w_encoder), (encoder)]
    model_names = ["CNNDis", "CNNGen"] if model_type == "cnn" else ["GNNDis", "GNNGen"]
    if use_f0:
        model_names += ["FNW"]
        model_names += ["CNNWEnc"] if use_encoder else ["GNNWEnc"]
    if use_encoder:
        model_names += ["CNNEnc"]

    kwargs = {
        "nw_data": nw_data,
        "output_channel": output_channel,
        "emb_dim": emb_dim,
        "in_emb_dim": in_emb_dim,
        "drop_out": drop_out,
        "sn": sn,
        "sln": sln,
        "h_dim": h_dim,
        "w_dim": w_dim,
        "num_head": num_head,
        "depth": depth,
        "gamma": gamma,
        "max_num": max_num,
        "ext_coeff": ext_coeff,
        "patch_size": patch_size,
        "num_source": num_source
    }

    if not use_f0 and not use_encoder:
        discriminator, generator = get_models(model_names, **kwargs)
        f0 = None
        w_encoder = None
        encoder = None
    elif use_f0 and not use_encoder:
        discriminator, generator, f0, w_encoder = get_models(model_names, **kwargs)
        encoder = None
    elif not use_f0 and use_encoder:
        discriminator, generator, encoder = get_models(model_names, **kwargs)
        f0 = None
        w_encoder = None
    else:
        discriminator, generator, f0, w_encoder, encoder = get_models(model_names, **kwargs)

    airl = AIRL(generator, discriminator, use_index, datasets, model_dir, image_data=image_data, encoder=encoder, h_dim=h_dim, emb_dim=emb_dim, f0=f0,
                 hinge_loss=hinge_loss, hinge_thresh=hinge_thresh, patch_size=patch_size, device=device)

    airl.train_models(CONFIG, epoch, bs, lr_g, lr_d, shuffle, train_ratio=train_ratio, d_epoch=d_epoch, lr_f0=lr_f0, lr_e=lr_e, image_file=image_file)



