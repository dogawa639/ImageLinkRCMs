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
from models.general import log


class AIRL:
    def __init__(self, generator, discriminator, use_index, datasets, model_dir, image_data=None, encoder=None, emb_dim=10, f0=None,
                 hinge_loss=False, hinge_thresh=0.5, device="cpu", sln=False):
        if hinge_thresh > 1.0 or hinge_thresh < 0.0:
            raise ValueError("hinge_thresh must be in [0, 1].")
        if encoder is not None and image_data is None:
            raise ValueError("image_data must be set when encoder is not None.")

        self.generator = generator
        self.discriminator = discriminator
        self.use_index = use_index  # whether inputs are the limited link feature or not
        self.datasets = datasets  # list of Dataset  train&validation
        self.model_dir = model_dir
        self.use_encoder = encoder is not None
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.f0 = f0  # h->w
        self.use_w = self.f0 is not None
        self.image_data = image_data
        self.hinge_loss = hinge_loss
        self.hinge_thresh = -log(tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False))
        self.device = device
        self.sln = sln

        self.link_num = len(self.datasets[0].nw_data.lids)
    
    def train(self, epochs, batch_size, lr_g, lr_d, shuffle,
              train_ratio=0.8, d_epoch=5, lr_f0=0.01, lr_e=0.01, image_file=None):
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

        result = {"loss_e": [], "loss_g": [], "loss_d": [],
                   "loss_e_val": [], "loss_g_val": [], "loss_d_val": [],
                   "criteria": []}
        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            epoch_loss_e = []
            epoch_loss_g = []
            epoch_loss_d = []
            for i, (dataloader_train, dataloader_val) in enumerate(dataloaders_real):
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
                for batch_real in dataloader_train:
                    # raw_data retain_grad=True
                    # Grid: (sum(links), 3, 3) corresponds to next_links
                    # Emb: (trip_num, link_num, link_num) sparse, corresponds to transition_matrix
                    bs = batch_real.shape[0]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)

                    if self.use_encoder:
                        # append image_feature to the original feature
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        raw_data_fake = self.generator(batch_real[0], i, w)  # batch_fake retain_grad=False
                    else:
                        raw_data_fake = self.generator(batch_real[0], i)  # batch_fake retain_grad=False
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
                            # f0 and encoder updata
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
                    raw_data_fake, batch_fake = self.generator.generate(batch_size, i)
                    d_real = self.get_d(batch_real)
                    d_fake = self.get_d(batch_fake)
                    loss_g, loss_d = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)

                    mode_loss_e_val.append(loss_e.clone().detach().cpu().item())
                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

            epoch_loss_e.append(np.mean(mode_loss_e))
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

            result["loss_e"].append(epoch_loss_e)
            result["loss_g"].append(epoch_loss_g)
            result["loss_d"].append(epoch_loss_d)
            result["loss_e_val"].append(epoch_loss_e_val)
            result["loss_g_val"].append(epoch_loss_g_val)
            result["loss_d_val"].append(epoch_loss_d_val)
            result["criteria"].append(criteria)

            t2 = time.perf_counter()
            print("epoch: {}, loss_e_val: {:.4f}, loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                e, epoch_loss_e_val[-1], epoch_loss_g_val[-1], epoch_loss_d_val[-1], criteria, t2 - t1))

        if image_file is not None:
            fig = plt.figure()
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)

            loss_g = np.array(result["loss_g"])
            loss_d = np.array(result["loss_d"])
            for i in range(loss_g.shape[1]):
                ax1.plot(loss_g[:, i], label="mode {}".format(i))
                ax2.plot(loss_d[:, i], label="mode {}".format(i))
            ax3.plot(result["criteria"])

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

        dump_json(result, os.path.join(self.model_dir, "result.json"))
        return result

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
        bs = batch.shape[0]
        image_feature = tensor(np.zeros((self.link_num, self.emb_dim), dtype=np.float32), device=self.device)
        for patches in self.image_data.load_link_patches():
            # patches [image_data]
            for i, patch in enumerate(patches):
                image_feature[i, :] = image_feature[i, :] + self.encoder(patch, w) / len(self.image_data)  # requires_grad=True

        # inputs: (bs, c, 3, 3) or (bs, link_num, feature_num)
        if self.use_index:
            # image_feature: tensor(link_num, emb_dim)
            # batch[3]: tensor(bs, 9)
            image_feature_tmp = image_feature[batch[3], :].transpose(1, 2).view(bs, -1, 3, 3)
            inputs = torch.cat((inputs, image_feature_tmp), dim=1)
        else:
            image_feature_tmp = image_feature.expand(bs, self.link_num, self.emb_dim)
            inputs = torch.cat((inputs, image_feature_tmp), dim=2)

        batch[0] = inputs
        return batch
    
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



        





