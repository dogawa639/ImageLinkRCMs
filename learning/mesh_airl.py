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
        self.dataset = dataset  # MeshDataset [train+test]
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

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle, "drop_last": True}

        dataloaders_train = []  # len: channel
        dataloaders_val = []  # len: channel
        for i in range(self.output_channel):
            self.dataset.set_channel(i)
            datasets = self.dataset.split_into(
                (min(train_ratio, max_train_num / len(self.dataset) * batch_size), 1 - train_ratio))
            dataloader_train, dataloader_val = [DataLoader(tmp, **dataset_kwargs) for tmp in datasets]
            dataloaders_train.append(dataloader_train)
            dataloaders_val.append(dataloader_val)
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
                for state, context, next_state, mask, positions, pis in dataloaders_train[channel]:  # batch
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
                        next_state_epsilon = torch.full_like(next_state, 1e-3)
                        next_state_epsilon = torch.where(next_state == 0, next_state_epsilon,
                                                         tensor(1.0, dtype=torch.float32,
                                                                device=next_state_epsilon.device))
                        log_d_g, log_d_d = self.get_log_d(inputs, next_state_epsilon, mask, positions, pis, pi,
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
                            #with detect_anomaly():
                            loss_g.backward(retain_graph=True)

                            # generator update
                            optimizer_gs[channel].step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                            pi = pi.detach()

                        del log_d_g, log_d_d, loss_d, loss_g

                # validation
                self.eval()
                ll = 0.0
                dist = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                bs_all = 0
                for state, context, next_state, mask, positions, pis in dataloaders_val[channel]:
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
                    next_state_epsilon = torch.full_like(next_state, 1e-3)
                    next_state_epsilon = torch.where(next_state == 0, next_state_epsilon, tensor(1.0, dtype=torch.float32, device=next_state_epsilon.device))
                    log_d_g, log_d_d = self.get_log_d(inputs, next_state, mask, positions, pis, pi,
                                                      channel)  # discriminator inference performed inside
                    loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

                    ll_tmp, d_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, next_state, mask, positions, pis, channel)
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

    def test_models(self, conf_file, dataset, image_file=None):
        print("test start.")
        dataset_kwargs = {"batch_size": 16, "shuffle": False, "drop_last": True}

        epoch_accuracy = []
        epoch_ll = []
        epoch_dist = []
        epoch_criteria = []
        epoch_bs = []
        for channel in range(self.output_channel):  # transportation mode
            dataset.set_channel(channel)
            dataloaders = DataLoader(dataset, **dataset_kwargs)  # [train_dataloader, val_dataloader]
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
            for state, context, next_state, mask, positions, pis in dataloaders:
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

                ll_tmp, d_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(inputs, next_state, mask, positions, pis,
                                                                           channel)
                ll += ll_tmp.detach().cpu().item()
                dist += d_tmp.detach().cpu().numpy().sum()
                tp += tp_tmp.detach().cpu().item()
                fp += fp_tmp.detach().cpu().item()
                tn += tn_tmp.detach().cpu().item()
                fn += fn_tmp.detach().cpu().item()
                ll0 += (log_d_g[0] * next_state).sum().detach().cpu().item()
                bs_all += state.shape[0]
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            criteria = dist
            epoch_accuracy.append(accuracy)
            epoch_ll.append(ll)
            epoch_dist.append(dist)
            epoch_criteria.append(criteria)
            epoch_bs.append(bs_all)

        for i in range(self.output_channel):
            print(f"channel {i}: accuracy: {epoch_accuracy[i]}, ll: {epoch_ll[i]}, dist: {epoch_dist[i]}, criteria: {epoch_criteria[i]}, bs: {epoch_bs[i]}")

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
        return ll, d_total, tp, fp, tn, fn

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

    # visualize
    def show_one_path(self, channel, aid):
        dataset_one = self.dataset.get_mesh_dataset_one_agent(channel, aid)
        traj_idx = dataset_one.mesh_traj_data.get_traj_idx_one_agent(channel, aid)
        self.mnw_data.show_path(traj_idx)

    def show_val(self, channel, aid, save_path):
        #dataset_one = self.dataset.get_mesh_dataset_one_agent(channel, aid)
        self.dataset.return_id = True
        self.dataset.set_channel(channel)
        traj_idx = self.dataset.mesh_traj_data.get_traj_idx_one_agent(channel, aid)
        dataloader = DataLoader(self.dataset, batch_size=8, shuffle=False, drop_last=False)

        y_range = np.arange(self.dataset.d * 2 + 1).reshape(1, -1, 1)
        x_range = np.arange(self.dataset.d * 2 + 1).reshape(1, 1, -1)
        qs = None
        exts = None
        pis_self = None
        other_traj_idxs = []  # (time, output_channel, num_points, 4) 4: y, x, u, v
        for state, context, next_state, mask, positions, pis, aids in dataloader:
            inputs = torch.cat((state, context), dim=1).to(self.device)
            next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
            mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
            positions = positions.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
            pis = pis.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
            aids = aids.to(self.device)  # (bs, max_agent_num)

            logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
            logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                          device=logits.device))
            pi = self.get_pi_from_logits(logits)
            f_val, util, ext, val = self.discriminators[channel].get_vals(inputs, positions, pis)
            ext = ext.sum(dim=1)  # (bs, 2d+1, 2d+1)
            q = util + ext + self.discriminators[channel].gamma * val

            mask = mask.detach().cpu().numpy()
            q = q.detach().cpu().numpy()
            ext = ext.detach().cpu().numpy()
            pi = pi.detach().cpu().numpy()
            aids = aids.detach().cpu().numpy()
            positions = positions.detach().cpu().numpy()
            pis = pis.detach().cpu().numpy()

            mask = mask[aids == aid]
            q = q[aids == aid]
            ext = ext[aids == aid]
            pi = pi[aids == aid]
            positions = positions[aids == aid]
            pis = pis[aids == aid]

            q = np.where(mask > 0, q, np.nan)
            ext = np.where(mask > 0, ext, np.nan)
            pi = np.where(mask > 0, pi, np.nan)

            if qs is None:
                qs = q
                exts = ext
                pis_self = pi
            else:
                qs = np.concatenate((qs, q), axis=0)
                exts = np.concatenate((exts, ext), axis=0)
                pis_self = np.concatenate((pis_self, pi), axis=0)

            # other agents
            for i in range(positions.shape[0]):  # time
                cur_traj_idx = []
                for c in range(self.output_channel):
                    active_positions = positions[i, (positions[i, :, c, :, :].sum(axis=(1, 2)) > 0), c, :, :]  # (num_agents, 2d+1, 2d+1)
                    active_pis = pis[i, (pis[i, :, c, :, :].sum(axis=(1, 2)) > 0), c, :, :]  # (num_agents, 2d+1, 2d+1)
                    y_idxs = (y_range * (active_positions == 0.0) * mask[i]).sum(axis=(1, 2)) + traj_idx[i, 0] - self.dataset.d  # (num_agents)
                    x_idxs = (x_range * (active_positions == 0.0) * mask[i]).sum(axis=(1, 2)) + traj_idx[i, 1] - self.dataset.d  # (num_agents)
                    y_idxs_next = (y_range * (active_pis == 0.0) * mask[i]).sum(axis=(1, 2)) + traj_idx[i, 0] - self.dataset.d   # (num_agents)
                    x_idxs_next = (x_range * (active_pis == 0.0) * mask[i]).sum(axis=(1, 2)) + traj_idx[i, 1] - self.dataset.d   # (num_agents)
                    us = x_idxs_next - x_idxs
                    vs = y_idxs_next - y_idxs
                    cur_traj_idx.append(np.array(np.stack((y_idxs, x_idxs, us, vs), axis=-1)))
                other_traj_idxs.append(cur_traj_idx)

        q_min, q_max = np.nanmin(qs), np.nanmax(qs)
        ext_min, ext_max = np.nanmin(exts), np.nanmax(exts)
        pi_min, pi_max = np.nanmin(pis_self), np.nanmax(pis_self)
        nan_mesh = np.full((self.dataset.d * 2 + 1, self.dataset.d * 2 + 1), np.nan)
        #nan_mesh = np.zeros((self.mnw_data.h_dim, self.mnw_data.w_dim), dtype=np.float32)

        color_traj = colors.hsv_to_rgb([channel / self.output_channel, 1.0, 1.0])
        color_next = colors.hsv_to_rgb([channel / self.output_channel, 0.5, 1.0])

        color_traj_other = colors.hsv_to_rgb([[tmp_channel / self.output_channel, 1.0, 0.5] for tmp_channel in range(self.output_channel)])
        color_next_other = colors.hsv_to_rgb([[tmp_channel / self.output_channel, 0.5, 0.5] for tmp_channel in range(self.output_channel)])

        def update(i):
            plt.clf()
            center_idx = traj_idx[i]  # (y_idx, x_idx)
            y_idx_min = max(0, center_idx[0] - self.dataset.d)
            y_idx_max = min(self.mnw_data.h_dim, center_idx[0] + self.dataset.d + 1)
            x_idx_min = max(0, center_idx[1] - self.dataset.d)
            x_idx_max = min(self.mnw_data.w_dim, center_idx[1] + self.dataset.d + 1)
            y_idx_min_local = y_idx_min - center_idx[0] + self.dataset.d
            y_idx_max_local = y_idx_max - center_idx[0] + self.dataset.d
            x_idx_min_local = x_idx_min - center_idx[1] + self.dataset.d
            x_idx_max_local = x_idx_max - center_idx[1] + self.dataset.d

            q_tmp = np.copy(nan_mesh)
            ext_tmp = np.copy(nan_mesh)
            pi_tmp = np.copy(nan_mesh)

            #q_tmp[y_idx_min:y_idx_max, x_idx_min:x_idx_max] = qs[i, y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local]
            #ext_tmp[y_idx_min:y_idx_max, x_idx_min:x_idx_max] = exts[i, y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local]
            #pi_tmp[y_idx_min:y_idx_max, x_idx_min:x_idx_max] = pis_self[i, y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local]
            q_tmp[y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local] = qs[i, y_idx_min_local:y_idx_max_local,
                                                              x_idx_min_local:x_idx_max_local]
            ext_tmp[y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local] = exts[i, y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local]
            pi_tmp[y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local] = pis_self[i, y_idx_min_local:y_idx_max_local, x_idx_min_local:x_idx_max_local]

            # trajectory
            x = traj_idx[:i, 1]
            y = self.mnw_data.h_dim - traj_idx[:i, 0] - 1
            u = traj_idx[1:i + 1, 1] - traj_idx[:i, 1]
            v = - (traj_idx[1:i + 1, 0] - traj_idx[:i, 0])
            # next step
            x_next = traj_idx[i, 1]
            y_next = self.mnw_data.h_dim - traj_idx[i, 0] - 1
            u_next = traj_idx[i + 1, 1] - traj_idx[i, 1] if i < len(traj_idx) - 1 else 0
            v_next = - (traj_idx[i + 1, 0] - traj_idx[i, 0]) if i < len(traj_idx) - 1 else 0

            # trajectories of other agents
            x_other = [np.concatenate([other_traj_idxs[j][tmp_channel][:, 1] for j in range(i)], axis=0) for tmp_channel
                       in range(self.output_channel)] if i > 0 else [[] for _ in range(self.output_channel)] # (output_channel, num_agents)
            y_other = [self.mnw_data.h_dim - 1 - np.concatenate([other_traj_idxs[j][tmp_channel][:, 0] for j in range(i)], axis=0) for tmp_channel
                       in range(self.output_channel)] if i > 0 else [[] for _ in range(self.output_channel)]  # (output_channel, num_agents)
            u_other = [np.concatenate([other_traj_idxs[j][tmp_channel][:, 2] for j in range(i)], axis=0) for tmp_channel
                       in range(self.output_channel)] if i > 0 else [[] for _ in range(self.output_channel)]
            v_other = [-np.concatenate([other_traj_idxs[j][tmp_channel][:, 3] for j in range(i)], axis=0) for tmp_channel
                       in range(self.output_channel)] if i > 0 else [[] for _ in range(self.output_channel)]
            x_other_next = [other_traj_idxs[i][tmp_channel][:, 1] for tmp_channel in range(self.output_channel)]
            y_other_next = [self.mnw_data.h_dim - 1 - other_traj_idxs[i][tmp_channel][:, 0] for tmp_channel in range(self.output_channel)]
            u_other_next = [other_traj_idxs[i][tmp_channel][:, 2] for tmp_channel in range(self.output_channel)]
            v_other_next = [-other_traj_idxs[i][tmp_channel][:, 3] for tmp_channel in range(self.output_channel)]

            # agent position
            common_feature_num = self.mnw_data.prop_dim - self.dataset.output_channel
            land_colors = colors.hsv_to_rgb([[tmp_channel/(common_feature_num+1), 0.5, 1.0] for tmp_channel in range(common_feature_num+1)])
            land_colors[0] = [0.0, 0.0, 0.0]
            land_colors = np.array(land_colors)
            channel_idxs = np.arange(1, 1+common_feature_num).reshape(-1, 1, 1)
            vals_idx = (self.mnw_data.get_prop_array()[self.dataset.output_channel:, :, :] * channel_idxs).sum(axis=0).astype(int)  # (h, w)
            vals = np.flipud(land_colors[vals_idx])  # (h, w, 3)
            plt.subplot(1, 4, 1)
            ax = plt.gca()
            im = self.mnw_data.show_grid(ax=ax, vals=vals)
            plt.plot([x_idx_min - 0.5, x_idx_max - 0.5, x_idx_max - 0.5, x_idx_min - 0.5, x_idx_min - 0.5],
                     [self.mnw_data.h_dim - y_idx_min + 0.5, self.mnw_data.h_dim - y_idx_min + 0.5, self.mnw_data.h_dim - y_idx_max + 0.5, self.mnw_data.h_dim - y_idx_max + 0.5, self.mnw_data.h_dim - y_idx_min + 0.5], linewidth=2.0, color="red")
            ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color=color_traj)
            ax.quiver(x_next, y_next, u_next, v_next, angles='xy', scale_units='xy', scale=1, color=color_next)
            for tmp_channel in range(self.output_channel):
                ax.quiver(x_other[tmp_channel], y_other[tmp_channel], u_other[tmp_channel], v_other[tmp_channel],
                          angles='xy', scale_units='xy', scale=1, color=color_traj_other[tmp_channel])
                ax.quiver(x_other_next[tmp_channel], y_other_next[tmp_channel], u_other_next[tmp_channel],
                          v_other_next[tmp_channel], angles='xy', scale_units='xy', scale=1,
                          color=color_next_other[tmp_channel])

            # q function
            plt.subplot(1, 4, 2)
            ax = plt.gca()
            vals = np.flipud(q_tmp)
            im = ax.imshow(vals, vmin=q_min, vmax=q_max, interpolation="bilinear")
            # grid
            ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
            ax.grid(which='minor', color="black", linestyle='-', linewidth=0.5)
            # ticks labels
            ax.set_xticks(np.arange(0, vals.shape[1], 5))
            ax.set_yticks(np.arange(0, vals.shape[0], 5))
            # remove minor ticks
            ax.tick_params(which="minor", bottom=False, left=False)

            plt.colorbar(im, ax=ax)

            # external reward
            plt.subplot(1, 4, 3)
            ax = plt.gca()
            vals = np.flipud(ext_tmp)
            im = ax.imshow(vals, vmin=ext_min, vmax=ext_max, interpolation="bilinear")
            # grid
            ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
            ax.grid(which='minor', color="black", linestyle='-', linewidth=0.5)
            # ticks labels
            ax.set_xticks(np.arange(0, vals.shape[1], 5))
            ax.set_yticks(np.arange(0, vals.shape[0], 5))
            # remove minor ticks
            ax.tick_params(which="minor", bottom=False, left=False)
            plt.colorbar(im, ax=ax)

            # policy
            plt.subplot(1, 4, 4)
            ax = plt.gca()
            vals = np.flipud(pi_tmp)
            im = ax.imshow(vals, vmin=pi_min, vmax=pi_max, interpolation="bilinear")
            # grid
            ax.set_xticks(np.arange(-0.5, vals.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, vals.shape[0], 1), minor=True)
            ax.grid(which='minor', color="black", linestyle='-', linewidth=0.5)
            # ticks labels
            ax.set_xticks(np.arange(0, vals.shape[1], 5))
            ax.set_yticks(np.arange(0, vals.shape[0], 5))
            # remove minor ticks
            ax.tick_params(which="minor", bottom=False, left=False)
            plt.colorbar(im, ax=ax)


        fig = plt.figure(figsize=(6.4*3, 4.8))
        anim = animation.FuncAnimation(fig, update, frames=len(traj_idx), interval=100)
        anim.save(save_path, writer="imagemagick")









