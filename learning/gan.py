import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

import time
import os

from utility import *
from models.general import log


class GAN:
    def __init__(self, generator, discriminator, datasets, model_dir, use_global_state=False,
                 airl=True, hinge_loss=False, hinge_thresh=-1.0, device="cpu", sln=False):
        self.generator = generator
        self.discriminator = discriminator
        self.datasets = datasets  # list of Dataset  train&validation
        self.model_dir = model_dir
        self.use_global_state = use_global_state
        self.airl = airl
        self.hinge_loss = hinge_loss
        self.hinge_thresh = tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False)
        self.device = device
        self.sln = sln
    
    def train(self, epochs, batch_size, lr_g, lr_d, shuffle,
              train_ratio=0.8, d_epoch=5, image_file=None):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)

        dataset_kargs = {"batch_size": batch_size, "shuffle": shuffle}
        dataloaders_real = [[DataLoader(tmp, **dataset_kargs)
                            for tmp in dataset.split_into((train_ratio, 1-train_ratio))]
                            for dataset in self.datasets]  # [train_dataloader, test_dataloader]

        result = {"loss_g": [], "loss_d": [], "loss_g_val": [], "loss_d_val": [],"criteria": []}
        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            epoch_loss_g = []
            epoch_loss_d = []
            for i, (dataloader_train, dataloader_val) in enumerate(dataloaders_real):
                # batch
                # Grid: inputs, masks, next_links
                # Emb: global_state, adj_matrix, transition_matrix
                mode_loss_g = []
                mode_loss_d = []
                epoch_loss_d_val = []
                epoch_loss_g_val = []
                self.generator.train()
                self.discriminator.train()
                for batch_real in dataloader_train:
                    # raw_data retain_grad=True
                    # Grid: (sum(links), 3, 3) corresponds to next_links
                    # Emb: (trip_num, link_num, link_num) sparse corresponds to transition_matrix
                    if self.use_global_state:

                    self.generator.zero_grad()
                    if self.sln:
                        raw_data_fake, w = self.generator(batch_real[0], i)  # batch_fake retain_grad=False
                    else:
                        raw_data_fake = self.generator(batch_real[0], i)  # batch_fake retain_grad=False
                    fake_batch = self.datasets.get_fake_batch(batch_real, raw_data_fake)

                    for j in range(len(batch_real)):
                        batch_real[j] = batch_real[j].to(self.device)
                        batch_fake[j] = batch_fake[j].to(self.device)
                    raw_data_fake.to(self.device)

                    # d value: retain_grad=True, same shape as raw_data_fake
                    for _ in range(d_epoch):
                        self.discriminator.zero_grad()

                        d_real = self.get_d(batch_real)
                        d_fake = self.get_d(batch_fake)
                        _, loss_d = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)

                        loss_d.backward()
                        optimizer_d.step()

                        mode_loss_d.append(loss_d.detach().cpu().item())

                    d_real = self.discriminator.get_d(batch_real)
                    d_fake = self.discriminator.get_d(batch_fake)
                    loss_g, _ = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)
                    loss_g.backward()
                    optimizer_g.step()

                    mode_loss_g.append(loss_g.detach().cpu().item())

                # validation
                mode_loss_g_val = []
                mode_loss_d_val = []
                self.generator.eval()
                self.discriminator.eval()
                for batch_real in dataloader_val:
                    raw_data_fake, batch_fake = self.generator.generate(batch_size, i)
                    d_real = self.get_d(batch_real)
                    d_fake = self.get_d(batch_fake)
                    loss_g, loss_d = self.loss(raw_data_fake, d_real, d_fake, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.detach().cpu().item())
                    mode_loss_d_val.append(loss_d.detach().cpu().item())

            epoch_loss_g.append(np.mean(mode_loss_g))
            epoch_loss_d.append(np.mean(mode_loss_d))
            epoch_loss_g_val.append(np.mean(mode_loss_g_val))
            epoch_loss_d_val.append(np.mean(mode_loss_d_val))

            criteria = self.generator.get_criteria(self.datasets)
            if criteria < min_criteria:
                min_criteria = criteria
                self.generator.save(self.model_dir)
                self.discriminator.save(self.model_dir)

            result["loss_g"].append(epoch_loss_g)
            result["loss_d"].append(epoch_loss_d)
            result["loss_g_val"].append(epoch_loss_g_val)
            result["loss_d_val"].append(epoch_loss_d_val)
            result["criteria"].append(criteria)

            t2 = time.perf_counter()
            print("epoch: {}, loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                e, epoch_loss_g_val[-1], epoch_loss_d_val[-1], criteria, t2 - t1))

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
            if self.airl:
                if hinge_loss:
                    l_g = -log(g_rate * d_f) + log(1. - g_rate * d_f)
                    l_d = torch.max(-log(d_r), self.hinge_thresh) + torch.max(-log(1. - d_f), self.hinge_thresh)
                else:
                    l_g = -log(g_rate * d_f) + log(1. - g_rate * d_f)
                    l_d = -log(d_r) - log(1. - d_f)

            else:
                if hinge_loss:
                    l_g = -log(g_rate * d_f)
                    l_d = torch.max(-log(d_r), self.hinge_thresh) + torch.max(-log(1. - d_f), self.hinge_thresh)
                else:
                    l_g = -log(g_rate * d_f)
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



        





