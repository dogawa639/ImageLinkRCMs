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
import matplotlib.colors as colors
import matplotlib.animation as animation
import datetime

import time
import os

from utility import *
from preprocessing.dataset import PatchDataset
from models.general import log
from logger import Logger

__all__ = ["MeshAIRLStatic"]


class MeshAIRLStatic:
    # maybe redundant to airl.py
    def __init__(self, generators, discriminators, dataset, model_dir,
                 image_data=None, encoders=None, hinge_loss=False, hinge_thresh=0.6, device="cpu"):
        if hinge_thresh > 1.0 or hinge_thresh < 0.0:
            raise ValueError("hinge_thresh must be in [0, 1].")
        if len(generators) != len(discriminators) or len(generators) != dataset.output_channel:
            raise ValueError("generators, discriminators and dataset.output_channel must be same length.")

        self.generators = [generator.to(device) for generator in generators]  # generator:UNetGen
        self.discriminators = [discriminator.to(device) for discriminator in discriminators]  # discriminator: UNetDisStatic
        self.encoders = None if encoders is None else [encoder.to(device) for encoder in encoders]
        self.dataset = dataset  # MeshDatasetStatic for only training
        self.model_dir = model_dir
        self.image_data = image_data  # MeshImageData
        self.hinge_loss = hinge_loss
        self.hinge_thresh = -log(tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False))
        self.device = device

        self.use_encoder = encoders is not None and image_data is not None

        self.mnw_data = dataset.mnw_data  # MeshNetwork
        self.output_channel = dataset.output_channel

    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
                     train_ratio=0.8, d_epoch=5, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file, fig_file=os.path.join(self.model_dir, "log.png"),
                     figsize=(6.4, 4.8 * 3))  # loss_g,loss_d,loss_g_val,loss_d_val,accuracy,ll,criteria
        epsilon = 0.0

        optimizer_gs = [optim.Adam(self.generators[i].parameters(), lr=lr_g) for i in range(self.output_channel)]
        optimizer_ds = [optim.Adam(self.discriminators[i].parameters(), lr=lr_d) for i in range(self.output_channel)]
        optimizer_es = None if self.encoders is None else [optim.Adam(self.encoders[i].parameters(), lr=lr_g) for i in range(self.output_channel)]

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle, "drop_last": True}

        print(f"Split dataset into train({train_ratio}) and val({1 - train_ratio})")
        dataset_train, dastaset_val = self.dataset.split_into((train_ratio, 1 - train_ratio))
        dataloaders_train = [DataLoader(dataset_tmp, **dataset_kwargs) for dataset_tmp in dataset_train.get_sub_datasets()]  # [MeshDatasetStaticSub] len: channel
        dataloaders_val = [DataLoader(dataset_tmp, **dataset_kwargs) for dataset_tmp in dastaset_val.get_sub_datasets()]  # [MeshDatasetStaticSub] len: channel

        print("Training starts.")
        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            epoch_loss_g = []
            epoch_loss_d = []
            epoch_loss_d_val = []
            epoch_loss_g_val = []
            epoch_accuracy = []
            epoch_ll = []
            epoch_dist = []
            epoch_criteria = []
            epoch_bs = []
            for channel in range(self.output_channel):  # transportation mode
                mode_loss_g = []
                mode_loss_d = []
                mode_loss_g_val = []
                mode_loss_d_val = []

                self.train()
                for state, context, next_state, mask, idx in dataloaders_train[channel]:  # batch
                    # state: (bs, prop_dim, 2d+1, 2d+1)
                    # context: (bs, context_num, 2d+1, 2d+1)
                    # next_state: (bs, 2d+1, 2d+1), 0 or 1
                    # mask: (bs, 2d+1, 2d+1), 0 or 1
                    # idx: (bs, 2), agent position index (y_idx, x_idx)
                    state = state.to(self.device)
                    context = context.to(self.device)
                    next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                    mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                    if self.use_encoder:
                        self.encoders[channel].train()
                    inputs, inputs_other = self.cat_all_feature(state, context, idx, channel)  # (bs, prop_dim+context_num+image_num, 2d+1, 2d+1). requires_grad=True

                    # pi of the target transportation
                    for i in range(self.output_channel):
                        self.generators[i].eval()
                    self.generators[channel].train()
                    logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)  # requires_grad=True, (bs, 2d+1, 2d+1)
                    # pi of other transportations
                    logits_other = [self.generators[i](inputs_other[i]) for i in range(self.output_channel) if i != channel]
                    logits_other = [torch.where(mask > 0, logits_tmp, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits_tmp.device)) for logits_tmp in logits_other]
                    if len(logits_other) > 0:
                        pi_other = torch.cat([self.get_pi_from_logits(logits_tmp).unsqueeze(1) for logits_tmp in logits_other], dim=1)  # detached, tensor(bs, output_channel-1, 2d+1, 2d+1)
                    else:
                        pi_other = torch.zeros((state.shape[0], 0, state.shape[2], state.shape[3]), dtype=torch.float32, device=state.device)

                    # get fake next state
                    next_idxs = torch.multinomial(pi.clone().detach().view(pi.shape[0], -1), 1).squeeze(dim=1)  # (bs)
                    next_state_fake = torch.zeros_like(next_state).view(next_state.shape[0], -1)
                    next_state_fake[torch.arange(next_state.shape[0]), next_idxs] = 1.0
                    next_state_fake = next_state_fake.view(next_state.shape)

                    for j in range(d_epoch):
                        # loss function calculation
                        next_state_epsilon = torch.full_like(next_state, epsilon)
                        next_state_epsilon = torch.where(next_state == 0, next_state_epsilon,
                                                         tensor(1.0, dtype=torch.float32,
                                                                device=next_state_epsilon.device))
                        for i in range(self.output_channel):
                            self.discriminators[i].eval()
                        self.discriminators[channel].train()
                        log_d_g, log_d_d = self.get_log_d(inputs.clone().detach(), pi_other, next_state_epsilon, mask, pi,
                                                          channel)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                        # discriminator update
                        optimizer_ds[channel].zero_grad()
                        #with detect_anomaly():
                        loss_d.backward(retain_graph=True)
                        optimizer_ds[channel].step()

                        mode_loss_d.append(loss_d.clone().detach().cpu().item())

                        if j == 0:
                            # f0 and encoder update

                            optimizer_gs[channel].zero_grad()
                            if self.use_encoder:
                                optimizer_es[channel].zero_grad()
                            #with detect_anomaly():
                            loss_g.backward(retain_graph=True)

                            # generator update
                            optimizer_gs[channel].step()
                            if self.use_encoder:
                                optimizer_es[channel].step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                            pi = pi.detach()

                        del log_d_g, log_d_d, loss_d, loss_g
                # update learning rate
                if e > 0 and e % 5 == 0:
                    gamma = 0.5
                    for param_group in optimizer_gs[channel].param_groups:
                        param_group['lr'] = param_group['lr'] * gamma
                    for param_group in optimizer_ds[channel].param_groups:
                        param_group['lr'] = param_group['lr'] * gamma
                    if self.use_encoder:
                        for param_group in optimizer_es[channel].param_groups:
                            param_group['lr'] = param_group['lr'] * gamma

                # validation
                self.eval()
                ll = 0.0
                dist = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                bs_all = 0
                showval = 5
                for state, context, next_state, mask, idx in dataloaders_val[channel]:  # batch
                    # state: (bs, prop_dim, 2d+1, 2d+1)
                    # context: (bs, context_num, 2d+1, 2d+1)
                    # next_state: (bs, 2d+1, 2d+1), 0 or 1
                    # mask: (bs, 2d+1, 2d+1), 0 or 1
                    # idx: (bs, 2), agent position index (y_idx, x_idx)
                    state = state.to(self.device)
                    context = context.to(self.device)
                    next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                    mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                    if self.use_encoder:
                        self.encoders[channel].eval()
                    inputs, inputs_other = self.cat_all_feature(state, context, idx,
                                                                channel)  # (bs, prop_dim+context_num+image_num, 2d+1, 2d+1). requires_grad=True

                    # pi of the target transportation
                    for i in range(self.output_channel):
                        self.generators[i].eval()
                    logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)  # requires_grad=True, (bs, 2d+1, 2d+1)
                    # pi of other transportations
                    logits_other = [self.generators[i](inputs_other[i]) for i in range(self.output_channel) if
                                    i != channel]
                    logits_other = [torch.where(mask > 0, logits_tmp, tensor(-9e15, dtype=torch.float32,
                                                                             device=logits_tmp.device)) for logits_tmp
                                    in logits_other]
                    if len(logits_other) > 0:
                        pi_other = torch.cat(
                            [self.get_pi_from_logits(logits_tmp).unsqueeze(1) for logits_tmp in logits_other],
                            dim=1)  # detached, tensor(bs, output_channel-1, 2d+1, 2d+1)
                    else:
                        pi_other = torch.zeros((state.shape[0], 0, state.shape[2], state.shape[3]), dtype=torch.float32,
                                               device=state.device)

                    # loss function calculation
                    next_state_epsilon = torch.full_like(next_state, epsilon)
                    next_state_epsilon = torch.where(next_state == 0, next_state_epsilon,
                                                     tensor(1.0, dtype=torch.float32,
                                                            device=next_state_epsilon.device))
                    for i in range(self.output_channel):
                        self.discriminators[i].eval()
                    log_d_g, log_d_d = self.get_log_d(inputs, pi_other, next_state_epsilon, mask, pi,
                                                      channel)  # discriminator inference performed inside
                    loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

                    if e % 10 == 0:
                        # show intermediate result samples
                        ll_tmp, d_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, pi_other, next_state,
                                                                                          mask, channel, showval=showval)
                        showval = 0
                    else:
                        ll_tmp, d_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, pi_other, next_state, mask, channel)
                    ll += ll_tmp.detach().cpu().item()
                    dist += d_tmp.detach().cpu().numpy().sum()
                    tp += tp_tmp.detach().cpu().item()
                    fp += fp_tmp.detach().cpu().item()
                    tn += tn_tmp.detach().cpu().item()
                    fn += fn_tmp.detach().cpu().item()
                    bs_all += state.shape[0]
                if tp + fp == 0:
                    accuracy = 0.0
                else:
                    accuracy = (tp + tn) / (tp + fp + tn + fn)
                criteria = dist

                epoch_loss_g.append(np.mean(mode_loss_g))
                epoch_loss_d.append(np.mean(mode_loss_d))
                epoch_loss_g_val.append(np.mean(mode_loss_g_val))
                epoch_loss_d_val.append(np.mean(mode_loss_d_val))
                epoch_accuracy.append(accuracy)
                epoch_ll.append(ll)
                epoch_dist.append(dist)
                epoch_bs.append(bs_all)
                epoch_criteria.append(criteria)

            log.add_log("loss_g", epoch_loss_g)
            log.add_log("loss_d", epoch_loss_d)
            log.add_log("loss_g_val", epoch_loss_g_val)
            log.add_log("loss_d_val", epoch_loss_d_val)
            log.add_log("accuracy", epoch_accuracy)
            log.add_log("ll", epoch_ll)
            log.add_log("dist", epoch_dist)
            log.add_log("bs_all", epoch_bs)
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
        print("Training ends.")

    def pretrain_models(self, conf_file, epochs, batch_size, lr_g, shuffle,
                     train_ratio=0.8):
        epsilon = 0.0

        optimizer_gs = [optim.Adam(self.generators[i].parameters(), lr=lr_g) for i in range(self.output_channel)]
        optimizer_es = None if self.encoders is None else [optim.Adam(self.encoders[i].parameters(), lr=lr_g) for i in range(self.output_channel)]

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle, "drop_last": True}

        print(f"Split dataset into train({train_ratio}) and val({1 - train_ratio})")
        dataset_train, dastaset_val = self.dataset.split_into((train_ratio, 1 - train_ratio))
        dataloaders_train = [DataLoader(dataset_tmp, **dataset_kwargs) for dataset_tmp in dataset_train.get_sub_datasets()]  # [MeshDatasetStaticSub] len: channel

        print("Pretraining starts.")
        min_criteria = 1e10
        for e in range(epochs):
            t1 = time.perf_counter()

            for channel in range(self.output_channel):  # transportation mode
                self.train()
                for state, context, next_state, mask, idx in dataloaders_train[channel]:  # batch
                    # state: (bs, prop_dim, 2d+1, 2d+1)
                    # context: (bs, context_num, 2d+1, 2d+1)
                    # next_state: (bs, 2d+1, 2d+1), 0 or 1
                    # mask: (bs, 2d+1, 2d+1), 0 or 1
                    # idx: (bs, 2), agent position index (y_idx, x_idx)
                    state = state.to(self.device)
                    context = context.to(self.device)
                    next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                    mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                    if self.use_encoder:
                        self.encoders[channel].train()
                    inputs, inputs_other = self.cat_all_feature(state, context, idx, channel)  # (bs, prop_dim+context_num+image_num, 2d+1, 2d+1). requires_grad=True

                    # pi of the target transportation
                    for i in range(self.output_channel):
                        self.generators[i].eval()
                    self.generators[channel].train()
                    logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))

                    loss_g = F.binary_cross_entropy_with_logits(logits, next_state, reduction="mean")

                    optimizer_gs[channel].zero_grad()
                    if self.use_encoder:
                        optimizer_es[channel].zero_grad()
                    #with detect_anomaly():
                    loss_g.backward(retain_graph=True)

                    # generator update
                    optimizer_gs[channel].step()
                    if self.use_encoder:
                        optimizer_es[channel].step()

                    del loss_g

            self.save()

            t2 = time.perf_counter()
            print("epoch: {}, time: {:.4f}".format(e, t2 - t1))
        print("Pretraining ends.")

    def test_models(self, conf_file, dataset, image_file=None):
        print("test start.")
        dataset_kwargs = {"batch_size": 16, "shuffle": False, "drop_last": False}
        epsilon = 0.0

        epoch_accuracy = []
        epoch_ll = []
        epoch_ll0 = []
        epoch_dist = []
        epoch_criteria = []
        epoch_bs = []
        epoch_bs_count = []
        dataloaders_test = [DataLoader(dataset_tmp, **dataset_kwargs) for dataset_tmp in
                             dataset.get_sub_datasets()]  # [MeshDatasetStaticSub] len: channel

        for channel in range(self.output_channel):  # transportation mode
            mode_loss_g = []
            mode_loss_d = []

            self.eval()
            ll = 0.0
            dist = 0.0
            tp = 0.0
            fp = 0.0
            tn = 0.0
            fn = 0.0
            ll0 = 0.0
            bs_all = 0
            bs_count = 0
            showval = 5
            for state, context, next_state, mask, idx in dataloaders_test[channel]:  # batch
                # state: (bs, prop_dim, 2d+1, 2d+1)

                # context: (bs, context_num, 2d+1, 2d+1)
                # next_state: (bs, 2d+1, 2d+1), 0 or 1
                # mask: (bs, 2d+1, 2d+1), 0 or 1
                # idx: (bs, 2), agent position index (y_idx, x_idx)
                state = state.to(self.device)
                context = context.to(self.device)
                next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                if self.use_encoder:
                    self.encoders[channel].eval()
                inputs, inputs_other = self.cat_all_feature(state, context, idx,
                                                            channel)  # (bs, prop_dim+context_num+image_num, 2d+1, 2d+1). requires_grad=True

                # pi of the target transportation
                for i in range(self.output_channel):
                    self.generators[i].eval()
                logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                              device=logits.device))
                pi = self.get_pi_from_logits(logits)  # requires_grad=True, (bs, 2d+1, 2d+1)
                # pi of other transportations
                logits_other = [self.generators[i](inputs_other[i]) for i in range(self.output_channel) if
                                i != channel]
                logits_other = [torch.where(mask > 0, logits_tmp, tensor(-9e15, dtype=torch.float32,
                                                                         device=logits_tmp.device)) for logits_tmp
                                in logits_other]
                if len(logits_other) > 0:
                    pi_other = torch.cat(
                        [self.get_pi_from_logits(logits_tmp).unsqueeze(1) for logits_tmp in logits_other],
                        dim=1)  # detached, tensor(bs, output_channel-1, 2d+1, 2d+1)
                else:
                    pi_other = torch.zeros((state.shape[0], 0, state.shape[2], state.shape[3]), dtype=torch.float32,
                                           device=state.device)

                # loss function calculation
                next_state_epsilon = torch.full_like(next_state, epsilon)
                next_state_epsilon = torch.where(next_state == 0, next_state_epsilon,
                                                 tensor(1.0, dtype=torch.float32,
                                                        device=next_state_epsilon.device))
                for i in range(self.output_channel):
                    self.discriminators[i].eval()
                log_d_g, log_d_d = self.get_log_d(inputs, pi_other, next_state_epsilon, mask, pi,
                                                  channel)  # discriminator inference performed inside
                loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                mode_loss_g.append(loss_g.clone().detach().cpu().item())
                mode_loss_d.append(loss_d.clone().detach().cpu().item())

                ll_tmp, d_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, pi_other, next_state, mask,
                                                                                  channel, showval=showval)
                showval = 0

                ll += ll_tmp.detach().cpu().item()
                dist += d_tmp.detach().cpu().numpy().sum()
                tp += tp_tmp.detach().cpu().item()
                fp += fp_tmp.detach().cpu().item()
                tn += tn_tmp.detach().cpu().item()
                fn += fn_tmp.detach().cpu().item()
                ll0 -= torch.log(mask.sum(dim=(1, 2))).sum().detach().cpu().item()
                bs_all += state.shape[0]
                bs_count += 1
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            criteria = dist
            epoch_accuracy.append(accuracy)
            epoch_ll.append(ll)
            epoch_ll0.append(ll0)
            epoch_dist.append(dist)
            epoch_criteria.append(criteria)
            epoch_bs.append(bs_all)
            epoch_bs_count.append(bs_count)

        for i in range(self.output_channel):
            print(f"channel {i}: accuracy: {epoch_accuracy[i]}, ll: {epoch_ll[i]} (Ave. {epoch_ll[i] / epoch_bs[i]}), ll0: {epoch_ll0[i]}, dist: {epoch_dist[i]}, criteria: {epoch_criteria[i]}, total_row: {epoch_bs[i]}, bs: {epoch_bs[i] / epoch_bs_count[i]}")

        # save result
        with open(os.path.join(self.model_dir, "test_result.txt"), "w") as f:
            for i in range(self.output_channel):
                f.write(f"channel {i}: accuracy: {epoch_accuracy[i]}, ll: {epoch_ll[i]} (Ave. {epoch_ll[i] / epoch_bs[i]}), ll0: {epoch_ll0[i]}, dist: {epoch_dist[i]}, criteria: {epoch_criteria[i]}, total_row: {epoch_bs[i]}, bs: {epoch_bs[i] / epoch_bs_count[i]}\n")

        print("test end.")

    def loss(self, log_d_g, log_d_d, hinge_loss=False):
        # log_d_real, log_d_fake: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if hinge_loss:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = torch.max(-log_d_d[0], self.hinge_thresh) + torch.max(-log_d_d[1], self.hinge_thresh)
        else:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = -log_d_d[0] - log_d_d[1]

        return l_g.sum(), l_d.sum()

    def get_log_d(self, inputs, pi_other, next_state, mask, pi, i):
        # batch
        # inputs: (bs, prop_dim+context_num, 2d+1, 2d+1)
        # pi_other: (bs, output_channel-1, 2d+1, 2d+1)
        # next_state: (bs, 2d+1, 2d+1), 0 or 1
        # mask: (bs, 2d+1, 2d+1), 0 or 1

        # generator output
        # pi : (bs, 2d+1, 2d+1)

        # discriminator inputs
        # input: (bs, total_feature, 2d+1, 2d+1)
        # positions: (bs, num_agents, output_channel, 2d+1, 2d+1)
        # pis: (bs, num_agents, output_channel, 2d+1, 2d+1)
        f_val = self.discriminators[i](inputs, pi_other)  # (bs, 2d+1, 2d+1)
        f_val_masked = f_val * mask
        f_val_clone_masked = f_val_masked.clone().detach()
        pi_clone = pi.clone().detach()

        next_idxs = torch.multinomial(pi_clone.view(pi_clone.shape[0], -1), 1).squeeze(dim=1)  # (bs)
        next_state_fake = torch.zeros_like(next_state).view(next_state.shape[0], -1)
        next_state_fake[torch.arange(next_state.shape[0]), next_idxs] = 1.0
        next_state_fake = next_state_fake.view(next_state.shape)

        log_d_g = (f_val_clone_masked - log(torch.exp(f_val_clone_masked) + pi)) * next_state_fake
        log_1_d_g = (log(pi) - log(torch.exp(f_val_clone_masked) + pi)) * next_state_fake

        log_d_d = (f_val_masked - log(torch.exp(f_val_masked) + pi_clone)) * next_state
        log_1_d_d = (log(pi_clone) - log(torch.exp(f_val_masked) + pi_clone)) * next_state_fake

        return (log_d_g, log_1_d_g), (log_d_d, log_1_d_d)

    def get_creteria(self, inputs, pi_other, next_state, mask, i, showval=0):
        # ll, TP, FP, FN, TN: scalar
        # f_val, util, val: (bs, 2d+1, 2d+1)
        # ext: (bs, num_agents, 2d+1, 2d+1)
        f_val, util, ext, val = self.discriminators[i].get_vals(inputs, pi_other)
        q = util + ext + self.discriminators[i].gamma * val
        q = torch.where(mask > 0, q, tensor(-9e15, dtype=torch.float32, device=q.device))
        # choose maximum q
        pi_q = F.softmax(q.view(q.shape[0], -1), dim=-1)  # (bs, (2*d+1)^2)
        logits = self.generators[i](inputs)  # (bs, 2*d+1, 2*d+1)
        logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32, device=logits.device))
        pi_g = F.softmax(logits.reshape(inputs.shape[0], -1), dim=-1)  # (bs, (2*d+1)^2)

        ll = log((pi_q * next_state.reshape(pi_q.shape)).sum(dim=-1)).sum()

        pred_state = (pi_q == pi_q.max(dim=1, keepdim=True)[0]).reshape(*q.shape).to(torch.float32)  # (bs, 2d+1, 2d+1)

        x_range = torch.arange(pred_state.shape[-1]).reshape(1, 1, -1).to(pred_state.device)
        y_range = torch.arange(pred_state.shape[-1]).reshape(1, -1, 1).to(pred_state.device)

        x_idxs_pred = (x_range * pred_state).clone().detach().cpu().sum(dim=-1).sum(dim=-1)  # (bs)
        y_idxs_pred = (y_range * pred_state).clone().detach().cpu().sum(dim=-1).sum(dim=-1)  # (bs)

        x_idxs_true = (x_range * next_state).clone().detach().cpu().sum(dim=-1).sum(dim=-1)  # (bs)
        y_idxs_true = (y_range * next_state).clone().detach().cpu().sum(dim=-1).sum(dim=-1)  # (bs)

        dx = x_idxs_pred - x_idxs_true
        dy = y_idxs_pred - y_idxs_true

        d_total = torch.sqrt(torch.square(dx * self.mnw_data.x_size) + np.square(dy * self.mnw_data.y_size))

        tp = (next_state * pred_state).sum()
        fp = (mask * (1 - next_state) * pred_state).sum()
        tn = (mask * (1 - next_state) * (1 - pred_state)).sum()
        fn = (next_state * (1 - pred_state)).sum()

        # plot figures
        if showval > 0:
            showval = min(inputs.shape[0], showval)
            fig = plt.figure(figsize=(8, 1 * showval))
            showprop = min(6, self.mnw_data.prop_dim+1)
            col = 4 + showprop
            for j in range(showval):
                ax = fig.add_subplot(showval, col, j * col + 1)  # pi_q, pi_g, next_state, mask
                ax.imshow(pi_q[j].reshape(*q.shape[1:]).detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Pi_q {j}")
                ax.set_xticks([]); ax.set_yticks([])
                ax = fig.add_subplot(showval, col, j * col + 2)
                ax.imshow(pi_g[j].reshape(*q.shape[1:]).detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Pi_g {j}")
                ax.set_xticks([]); ax.set_yticks([])
                ax = fig.add_subplot(showval, col, j * col + 3)
                ax.imshow(next_state[j].detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Next_state {j}")
                ax.set_xticks([]); ax.set_yticks([])
                ax = fig.add_subplot(showval, col, j * col + 4)
                ax.imshow(mask[j].detach().cpu().numpy(), cmap="gray")
                ax.set_title(f"Mask {j}")
                ax.set_xticks([]); ax.set_yticks([])
                for k in range(showprop):
                    ax = fig.add_subplot(showval, col, j * col + 5 + k)
                    ax.imshow(inputs[j, k].detach().cpu().numpy(), cmap="gray")
                    ax.set_title(f"Prop {k} {j}")
                    ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f"creteria{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"))
            plt.clf()
            plt.close()

        return ll, d_total, tp, fp, tn, fn

    def get_pi_from_logits(self, logits):
        # logits: tensor(bs, 1, 2d+1, 2d+1)
        # pi: (bs, 1, 2d+1, 2d+1)
        pi = F.softmax(logits.view(logits.shape[0], -1), dim=-1).reshape(*logits.shape)
        return pi

    def cat_all_feature(self, state, context, idx, channel):
        # state: (bs, prop_dim, 2d+1, 2d+1)
        # context: (bs, context_num, 2d+1, 2d+1)
        # idx: (bs, 2), agent position index (y_idx, x_idx)
        # channel: int, transportation mode
        # return: (bs, prop_dim+context_num+image_feature_dim, 2d+1, 2d+1)
        img_feature = None
        if self.use_encoder:
            for bs in range(state.shape[0]):
                images = self.image_data.load_mesh_images(idx[bs, 0], idx[bs, 1],
                                                          self.dataset.d)[0].to(self.device)  # tensor((2d+1)^2, c, h, w)
                compressed = [self.encoders[tmp_channel].compress(images, 0) for tmp_channel in range(self.output_channel)]  # [tensor((2d+1)^2, emb_dim) or tuple]
                compressed = [compressed[tmp_channel][0] if self.encoders[tmp_channel].output_atten else compressed[tmp_channel] for tmp_channel in range(self.output_channel)]  # [tensor((2d+1)^2, emb_dim)]
                tmp_feature = self.encoders[channel](compressed[channel])  # currently only one image is used. tensor((2d+1)^2, emb_dim)
                tmp_other_feature = [self.encoders[tmp_channel](compressed[tmp_channel].clone().detach()).clone().detach() for tmp_channel in range(self.output_channel)]  # [tensor((2d+1)^2, emb_dim)]
                tmp_feature = torch.permute(tmp_feature, (1, 0)).reshape(1, -1, self.dataset.d * 2 + 1, self.dataset.d * 2 + 1)
                tmp_other_feature = [torch.permute(tmp_other_feature[i], (1, 0)).reshape(1, -1, self.dataset.d * 2 + 1, self.dataset.d * 2 + 1) for i in range(len(tmp_other_feature))]
                if img_feature is None:
                    img_feature = tmp_feature
                    other_feature = tmp_other_feature
                else:
                    img_feature = torch.cat((img_feature, tmp_feature), dim=0)  # [tensor(bs, emb_dim, 2d+1, 2d+1)]
                    other_feature = [torch.cat((other_feature[i], tmp_other_feature[i]), dim=0) for i in range(len(other_feature))]  # [tensor(bs, emb_dim, 2d+1, 2d+1)]
        if img_feature is not None:
            print_prop = False
            if print_prop:
                print(f"state  dim: {state.shape[1]}, mean: {state.mean(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}, std: {state.std(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}")
                print(f"context  dim: {context.shape[1]}, mean: {context.mean(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}, std: {context.std(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}")
                print(f"img_feature  dim: {img_feature.shape[1]}, mean: {img_feature.mean(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}, std: {img_feature.std(dim=(0, 2, 3)).clone().detach().cpu().numpy().tolist()}")
            return torch.cat((state, context, img_feature), dim=1), [torch.cat((state, torch.zeros_like(context), other_feature[i]), dim=1) for i in range(len(other_feature))]
        else:
            return torch.cat((state, context), dim=1), [torch.cat((state, torch.zeros_like(context)), dim=1) for _ in range(self.output_channel)]

    def show_attention_map(self, img_tensor, show=True):
        # img_tensor: tensor(bs2, c, h, width) or (c, h, width)
        # self.encoders[i]: output_atten=True
        if len(img_tensor.shape) == 3:
            bs = 1
        else:
            bs = img_tensor.shape[0]
        fig = plt.figure(figsize=(4 * bs, 4 * self.output_channel))
        img_tensor = img_tensor.to(self.device)
        attens = []
        for channel in range(self.output_channel):
            _, atten = self.encoders[channel].compress(img_tensor)  # atten: (bs, h, w)
            atten = atten.clone().detach().cpu().numpy()
            attens.append(atten)
        if show:
            for channel in range(self.output_channel):
                bs = attens[channel].shape[0]
                for i in range(bs):
                    ax = fig.add_subplot(self.output_channel, bs, channel * bs + i + 1)
                    ax.imshow(attens[channel][i], cmap="gray")
                    ax.set_title(f"channel {channel} bs {i}")
                    ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.show()
        return attens

    def train(self):
        for i in range(self.output_channel):
            self.generators[i].train()
            self.discriminators[i].train()
            if self.use_encoder:
                self.encoders[i].train()

    def eval(self):
        for i in range(self.output_channel):
            self.generators[i].eval()
            self.discriminators[i].eval()
            if self.use_encoder:
                self.encoders[i].eval()

    def save(self):
        for i in range(self.output_channel):
            self.generators[i].save(self.model_dir, i)
            self.discriminators[i].save(self.model_dir, i)
            if self.use_encoder:
                self.encoders[i].save(self.model_dir, i)

    def load(self, model_dir=None):
        model_dir = model_dir if model_dir is not None else self.model_dir
        for i in range(self.output_channel):
            self.generators[i].load(model_dir, i)
            self.discriminators[i].load(model_dir, i)
            if self.use_encoder:
                self.encoders[i].load(model_dir, i)

    # visualize
    def show_one_path(self, channel, aid):
        traj_idx = self.dataset.mesh_traj_data.get_traj_idx_one_agent(channel, aid)
        self.mnw_data.show_path(traj_idx)

