import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import detect_anomaly

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

__all__ = ["MeshAIRL"]


class MeshAIRL:
    # maybe redundant to airl.py
    def __init__(self, generators, discriminators, dataset, model_dir,
                 hinge_loss=False, hinge_thresh=0.6, device="cpu"):
        if hinge_thresh > 1.0 or hinge_thresh < 0.0:
            raise ValueError("hinge_thresh must be in [0, 1].")
        if len(generators) != len(discriminators) or len(generators) != dataset.output_channel:
            raise ValueError("generators, discriminators and dataset.output_channel must be same length.")

        self.generators = [generator.to(device) for generator in generators]
        self.discriminators = [discriminator.to(device) for discriminator in discriminators]
        self.dataset = dataset # MeshDataset [train+test]
        self.model_dir = model_dir
        self.hinge_loss = hinge_loss
        self.hinge_thresh = -log(tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False))
        self.device = device

        self.mnw_data = dataset.mnw_data  # MeshNetwork
        self.output_channel = dataset.output_channel

    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
                     train_ratio=0.8, max_train_num=10000, d_epoch=5, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file,
                     figsize=(6.4, 4.8 * 3))  # loss_g,loss_d,loss_g_val,loss_d_val,accuracy,ll,criteria

        optimizer_gs = [optim.Adam(self.generators[i].parameters(), lr=lr_g) for i in range(self.output_channel)]
        optimizer_ds = [optim.Adam(self.discriminators[i].parameters(), lr=lr_d) for i in range(self.output_channel)]

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle}

        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            epoch_loss_g = []
            epoch_loss_d = []
            epoch_loss_d_val = []
            epoch_loss_g_val = []
            epoch_accuracy = []
            epoch_ll = []
            epoch_criteria = []
            for channel in range(self.output_channel):  # transportation mode
                self.dataset.set_channel(channel)
                datasets = self.dataset.split_into(
                    (min(train_ratio, max_train_num / len(self.dataset) * batch_size), 1 - train_ratio))
                dataloader_train, dataloader_val = [DataLoader(tmp, **dataset_kwargs)for tmp in datasets]  # [train_dataloader, val_dataloader]
                mode_loss_g = []
                mode_loss_d = []
                mode_loss_g_val = []
                mode_loss_d_val = []

                self.train()
                for state, context, next_state, mask, positions, pis in dataloader_train:  # batch
                    # state: (bs, prop_dim, 2d+1, 2d+1)
                    # context: (bs, context_num, 2d+1, 2d+1)
                    # next_state: (bs, 2d+1, 2d+1), 0 or 1
                    # mask: (bs, 2d+1, 2d+1), 0 or 1
                    # positions: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                    # pis: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                    inputs = torch.cat((state, context), dim=1).to(self.device)  # (bs, prop_dim+context_num, 2d+1, 2d+1)
                    next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                    mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                    positions = positions.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                    pis = pis.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)

                    logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)  # requires_grad=True, (bs, 2d+1, 2d+1)

                    # get fake next state
                    next_idxs = torch.multinomial(pi.clone().detach().view(pi.shape[0], -1), 1).squeeze(dim=1)  # (bs)
                    next_state_fake = torch.zeros_like(next_state).view(next_state.shape[0], -1)
                    next_state_fake[torch.arange(next_state.shape[0]), next_idxs] = 1.0
                    next_state_fake = next_state_fake.view(next_state.shape)

                    for j in range(d_epoch):
                        # loss function calculation
                        log_d_g, log_d_d = self.get_log_d(inputs, next_state, mask, positions, pis, pi, channel)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                        # discriminator update
                        optimizer_ds[channel].zero_grad()
                        with detect_anomaly():
                            loss_d.backward(retain_graph=True)
                        optimizer_ds[channel].step()

                        mode_loss_d.append(loss_d.clone().detach().cpu().item())

                        if j == 0:
                            # f0 and encoder update

                            optimizer_gs[channel].zero_grad()
                            with detect_anomaly():
                                loss_g.backward(retain_graph=True)

                            # generator update
                            optimizer_gs[channel].step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                            pi = pi.detach()

                        del log_d_g, log_d_d, loss_d, loss_g

                # validation
                self.eval()
                ll = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                for state, context, next_state, mask, positions, pis in dataloader_val:
                    inputs = torch.cat((state, context), dim=1).to(
                        self.device)  # (bs, prop_dim+context_num, 2d+1, 2d+1)
                    next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                    mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                    positions = positions.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                    pis = pis.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)

                    logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)  # requires_grad=True, (bs, 2d+1, 2d+1)

                    # loss function calculation
                    log_d_g, log_d_d = self.get_log_d(inputs, next_state, mask, positions, pis, pi,
                                                      channel)  # discriminator inference performed inside
                    loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

                    ll_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, next_state, mask, positions, pis, channel)
                    ll += ll_tmp.detach().cpu().item()
                    tp += tp_tmp.detach().cpu().item()
                    fp += fp_tmp.detach().cpu().item()
                    tn += tn_tmp.detach().cpu().item()
                    fn += fn_tmp.detach().cpu().item()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                criteria = -ll

                epoch_loss_g.append(np.mean(mode_loss_g))
                epoch_loss_d.append(np.mean(mode_loss_d))
                epoch_loss_g_val.append(np.mean(mode_loss_g_val))
                epoch_loss_d_val.append(np.mean(mode_loss_d_val))
                epoch_accuracy.append(accuracy)
                epoch_ll.append(ll)
                epoch_criteria.append(criteria)

            log.add_log("loss_g", epoch_loss_g)
            log.add_log("loss_d", epoch_loss_d)
            log.add_log("loss_g_val", epoch_loss_g_val)
            log.add_log("loss_d_val", epoch_loss_d_val)
            log.add_log("accuracy", epoch_accuracy)
            log.add_log("ll", epoch_ll)
            log.add_log("criteria", epoch_criteria)

            criteria = sum(epoch_criteria)

            if criteria < min_criteria:
                min_criteria = criteria
                print(f"save model. minimum criteria: {min_criteria}")
                self.save()

            t2 = time.perf_counter()
            print("epoch: {}, loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                e, epoch_loss_g_val[-1], epoch_loss_d_val[-1], criteria, t2 - t1))
        if image_file is not None:
            log.save_fig(image_file)
        log.close()

    def loss(self, log_d_g, log_d_d, hinge_loss=False):
        # log_d_real, log_d_fake: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if hinge_loss:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = torch.max(-log_d_d[0], self.hinge_thresh) + torch.max(-log_d_d[1], self.hinge_thresh)
        else:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = -log_d_d[0] - log_d_d[1]

        return l_g.sum(), l_d.sum()

    def get_log_d(self, inputs, next_state, mask, positions, pis, pi, i):
        # batch
        # inputs: (bs, prop_dim+context_num, 2d+1, 2d+1)
        # next_state: (bs, 2d+1, 2d+1), 0 or 1
        # mask: (bs, 2d+1, 2d+1), 0 or 1
        # positions: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
        # pis: (bs, max_agent_num, output_channel, 2d+1, 2d+1)

        # generator output
        # pi : (bs, 2d+1, 2d+1)

        # discriminator inputs
        # input: (bs, total_feature, 2d+1, 2d+1)
        # positions: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # pis: (bs, num_agents, output_channel, 2d+1, 2d+1)
        f_val = self.discriminators[i](inputs, positions, pis)  # (bs, 2d+1, 2d+1)
        f_val_masked = f_val * mask
        f_val_clone_masked = f_val_masked.clone().detach()
        pi_clone = pi.clone().detach()
        log_d_g = (f_val_clone_masked - log(torch.exp(f_val_clone_masked) + pi)) * pi_clone
        log_1_d_g = (log(pi) - log(torch.exp(f_val_clone_masked) + pi)) * pi_clone

        log_d_d = (f_val_masked - log(torch.exp(f_val_masked) + pi_clone)) * next_state
        log_1_d_d = (log(pi_clone) - log(torch.exp(f_val_masked) + pi_clone)) * pi_clone

        return (log_d_g, log_1_d_g), (log_d_d, log_1_d_d)

    def get_creteria(self, inputs, next_state, mask, positions, pis, i):
        # ll, TP, FP, FN, TN: scalar
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        f_val, util, ext, val = self.discriminators[i].get_vals(inputs, positions, pis)
        ext = ext.sum(dim=1)  # (bs, 2d+1, 2d+1)
        q = util + ext + self.discriminators[i].gamma * val
        q = torch.where(mask > 0, q, tensor(-9e15, dtype=torch.float32, device=q.device))
        # choose maximum q
        pi_q = F.softmax(q.view(q.shape[0], -1), dim=-1)  # (bs, (2*d+1)^2)
        ll = log((pi_q.reshape(*q.shape) * next_state).sum(dim=-1)).sum()

        pred_state = (pi_q == pi_q.max(dim=1, keepdim=True)[0]).reshape(*q.shape).to(torch.float32)
        tp = (next_state * pred_state).sum()
        fp = (mask * (1 - next_state) * pred_state).sum()
        tn = (mask * (1 - next_state) * (1 - pred_state)).sum()
        fn = (next_state * (1 - pred_state)).sum()
        return ll, tp, fp, tn, fn

    def get_pi_from_logits(self, logits):
        # logits: tensor(bs, total_feature, 2d+1, 2d+1)
        # pi: (bs, total_feature, 2d+1, 2d+1)
        pi = F.softmax(logits.view(logits.shape[0], -1), dim=-1).reshape(*logits.shape)
        return pi

    def train(self):
        for i in range(self.output_channel):
            self.generators[i].train()
            self.discriminators[i].train()

    def eval(self):
        for i in range(self.output_channel):
            self.generators[i].eval()
            self.discriminators[i].eval()

    def save(self):
        for i in range(self.output_channel):
            self.generators[i].save(self.model_dir, i)
            self.discriminators[i].save(self.model_dir, i)

    def load(self):
        for i in range(self.output_channel):
            self.generators[i].load(self.model_dir, i)
            self.discriminators[i].load(self.model_dir, i)


