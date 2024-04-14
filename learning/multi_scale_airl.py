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
from models.general import log, FF
from logger import Logger

from learning.airl import AIRL
from learning.mesh_airl import MeshAIRL

__all__ = ["MesoAIRL", "MicroAIRL"]


class MesoAIRL(AIRL):
    def __init__(self, micro_airl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_airl = micro_airl

        self.feature_num = self.datasets[0].nw_data.feature_num
        self.alpha = FF(self.feature_num, 1, act_fn=F.sigmoid, sn=False).to(self.device)

        self.utils_micro = None
        self.exts_micro = None

    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
                     train_ratio=0.8, max_train_num=10000, d_epoch=5, lr_f0=0.01, lr_e=0.0, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file, figsize=(
        6.4, 4.8 * 3))  # loss_e,loss_g,loss_d,loss_e_val,loss_g_val,loss_d_val,accuracy,ll,criteria
        self.set_micro_based_reward()

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        optimizer_alpha = optim.Adam(self.alpha.parameters(), lr=lr_d)
        train_encoder = lr_e > 0.0 and self.use_encoder
        if self.use_w:
            optimizer_0 = optim.Adam(self.f0.parameters(), lr=lr_f0)
        if train_encoder:
            optimizer_e = optim.Adam(self.encoder.parameters(), lr=lr_e)

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle}
        dataloaders_real = [[DataLoader(tmp, **dataset_kwargs)
                             for tmp in dataset.split_into(
                (min(train_ratio, max_train_num / len(dataset) * batch_size), 1 - train_ratio))]
                            for dataset in self.datasets]  # [train_dataloader, test_dataloader]

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
            for i, (dataloader_train, dataloader_val) in enumerate(dataloaders_real):  # transportation mode
                # batch
                # Grid: inputs, masks, next_links, link_idxs, h
                # Emb: global_state, adj_matrix, transition_matrix, h
                mode_loss_g = []
                mode_loss_d = []
                mode_loss_g_val = []
                mode_loss_d_val = []

                self.train()
                for batch_real in dataloader_train:  # batch
                    # raw_data requires_grad=True
                    # Grid: (sum(links), 3, 3) corresponds to next_links
                    # Emb: (trip_num, link_num, link_num) sparse, corresponds to transition_matrix
                    batch_real = [tmp.to_dense().to(torch.float32).to(self.device) for tmp in batch_real]
                    bs = batch_real[0].shape[0]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)
                    if self.use_encoder:
                        # append image_feature to the original feature
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        logits = self.generator(batch_real[0],
                                                w=w)  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    else:
                        logits = self.generator(
                            batch_real[0])  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    mask = batch_real[1].unsqueeze(1)
                    if self.use_index:
                        mask = mask.view(-1, 1, logits.shape[-2], logits.shape[-1])
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)

                    for j in range(d_epoch):
                        # loss function calculation
                        log_d_g, log_d_d = self.get_log_d(batch_real, pi, i,
                                                          w=w)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                        # discriminator update
                        optimizer_d.zero_grad()
                        optimizer_alpha.zero_grad()
                        with detect_anomaly():
                            loss_d.backward(retain_graph=True)
                        optimizer_d.step()
                        optimizer_alpha.step()

                        mode_loss_d.append(loss_d.clone().detach().cpu().item())

                        if j == 0:
                            # f0 and encoder update
                            if self.use_w:
                                optimizer_0.zero_grad()
                            if train_encoder:
                                optimizer_e.zero_grad()

                            optimizer_g.zero_grad()
                            with detect_anomaly():
                                loss_g.backward(retain_graph=True)

                            if self.use_w:
                                optimizer_0.step()
                            if train_encoder:
                                optimizer_e.step()

                            # generator update
                            optimizer_g.step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                            pi = pi.detach()
                            w = w.detach()

                        del log_d_g, log_d_d, loss_d, loss_g

                # validation
                self.eval()
                ll = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                for batch_real in dataloader_val:
                    batch_real = [tmp.to_dense().to(torch.float32).to(self.device) for tmp in batch_real]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)
                    if self.use_encoder:
                        # append image_feature to the original feature
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        logits = self.generator(batch_real[0], w=w)  # raw_data_fake requires_grad=True
                    else:
                        logits = self.generator(batch_real[0])  # raw_data_fake requires_grad=True
                    mask = batch_real[1].unsqueeze(1)
                    if self.use_index:
                        mask = mask.view(-1, 1, logits.shape[-2], logits.shape[-1])
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
                    pi = self.get_pi_from_logits(logits)
                    log_d_g, log_d_d = self.get_log_d(batch_real, pi, i,
                                                      w=w)  # discriminator inference performed inside
                    loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

                    ll_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_criteria(batch_real, pi, i, w=w)
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

    def get_log_d(self, batch, pi, i, w=None):
        # batch
        # Grid: inputs, masks, next_links, link_idxs, hs
        # Emb: global_state, adj_matrix, transition_matrix, hs

        # d retain_grad=True
        # Grid: (sum(links), 3, 3) corresponds to next_links
        # Emb: (trip_num, link_num, link_num) sparse corresponds to transition_matrix
        # output: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if self.use_index:
            input, mask, next_link, idxs, _ = batch
            idxs = idxs.detach().cpu().numpy().astype(int)
            mask = mask.view(mask.shape[0], input.shape[-2], input.shape[-1])
            next_link = next_link.view(next_link.shape[0], input.shape[-2], input.shape[-1])
        else:
            input, mask, next_link, _ = batch
        _, util_macro, ext_macro, val = self.discriminator.get_vals(input, pi, i=i, w=w)  # Grid(sum(links), 3, 3), Emb (bs, 2d+1, 2d+1)
        alpha = self.alpha(input[:, :self.feature_num, :, :].permute(0, 2, 3, 1)).squeeze(-1)  # (bs, *choice_space)
        util_micro = tensor(self.utils_micro[i, :], dtype=torch.float32, device=self.device)  # (link_num)
        ext_micro = tensor(self.exts_micro[i, :], dtype=torch.float32, device=self.device)  # (link_num)

        if self.use_index:
            util_micro = util_micro[idxs.flatten()].reshape(util_macro.shape)
            ext_micro = ext_micro[idxs.flatten()].reshape(ext_macro.shape)

        util = util_macro * alpha + torch.clip(util_micro, util_macro.min(), util_macro.max()) * (1.0 - alpha)
        ext = ext_macro * alpha + torch.clip(ext_micro, ext_macro.min(), ext_macro.max()) * (1.0 - alpha)

        f_val = self.discriminator.get_f_val(util, ext, val)  # (bs, *choice_space)
        f_val_masked = f_val * mask
        f_val_clone_masked = f_val_masked.clone().detach()
        pi_i_clone = pi.clone().detach()[:, i, :, :]
        # f_val: (bs, oc, *choice_space), pi: (bs, oc, *choice_space)
        log_d_g = (f_val_clone_masked - log(torch.exp(f_val_clone_masked) + pi[:, i, :, :])) * pi_i_clone
        log_1_d_g = (log(pi[:, i, :, :]) - log(torch.exp(f_val_clone_masked) + pi[:, i, :, :])) * pi_i_clone

        log_d_d = (f_val_masked - log(torch.exp(f_val_masked) + pi_i_clone)) * next_link
        log_1_d_d = (log(pi_i_clone) - log(torch.exp(f_val_masked) + pi_i_clone)) * pi_i_clone

        return (log_d_g, log_1_d_g), (log_d_d, log_1_d_d)

    def get_criteria(self, batch, pi, i, w=None):
        # ll, TP, FP, FN, TN
        if self.use_index:
            input, mask, next_mask, idxs, _ = batch
            idxs = idxs.detach().cpu().numpy().astype(int)
            mask = mask.view(mask.shape[0], input.shape[-2], input.shape[-1])  # (*, 3, 3)
            next_mask = next_mask.view(next_mask.shape[0], input.shape[-2], input.shape[-1])  # (*, 3, 3)
        else:
            input, mask, next_mask, _ = batch
        _, util_macro, ext_macro, val = self.discriminator.get_vals(input, pi, i=i,
                                                                    w=w)  # Grid(sum(links), 3, 3), Emb (bs, 2d+1, 2d+1)
        alpha = self.alpha(input[:, :self.feature_num, :, :].permute(0, 2, 3, 1)).squeeze(-1)  # (bs, *choice_space)
        util_micro = tensor(self.utils_micro[i, :], dtype=torch.float32, device=self.device)  # (link_num)
        ext_micro = tensor(self.exts_micro[i, :], dtype=torch.float32, device=self.device)  # (link_num)

        if self.use_index:
            util_micro = util_micro[idxs.flatten()].reshape(util_macro.shape)
            ext_micro = ext_micro[idxs.flatten()].reshape(ext_macro.shape)

        util = util_macro * alpha + torch.clip(util_micro, util_macro.min(), util_macro.max()) * (1.0 - alpha)
        ext = ext_macro * alpha + torch.clip(ext_micro, ext_macro.min(), ext_macro.max()) * (1.0 - alpha)

        q = util + ext + self.discriminator.gamma * val
        q = torch.where(mask > 0, q, tensor(-9e15, dtype=torch.float32, device=q.device))
        # choose maximum q
        if self.use_index:  # choose from choice_space
            q = q.view(q.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)  # (*, 9)
            next_mask = next_mask.view(next_mask.shape[0], -1)  # (*, 9)
        pi_q = F.softmax(q, dim=-1)
        ll = log((pi_q * next_mask).sum(dim=-1)).sum()

        pred_mask = (q == q.max(dim=1, keepdim=True)[0]).to(torch.float32)
        tp = (next_mask * pred_mask).sum()
        fp = (mask * (1 - next_mask) * pred_mask).sum()
        tn = (mask * (1 - next_mask) * (1 - pred_mask)).sum()
        fn = (next_mask * (1 - pred_mask)).sum()
        return ll, tp, fp, tn, fn

    def get_velocity(self):
        # [[velocity]] (link_num, output_channel)
        velocities = [[self.datasets[0].nw_data.edges[lid].prop["ped_velocity"], self.datasets[0].nw_data.edges[lid].prop["veh_velocity"]] for lid in self.datasets[0].nw_data.lids]
        return np.array(velocities)

    def set_micro_based_reward(self):
        # utlity, interaction (output_channel, link_num)
        velocities = self.get_velocity()
        utils = [[] for _ in range(self.output_channel)]
        exts = [[] for _ in range(self.output_channel)]
        for channel in range(self.output_channel):
            for link in range(len(self.datasets[0].nw_data.lids)):
                _, sample_macro = self.micro_airl.sample_given_v(channel, velocities[link, channel])
                utils[channel].append(sample_macro[0])
                exts[channel].append(sample_macro[1])
        self.utils_micro = np.array(utils)
        self.exts_micro = np.array(exts)

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.alpha.train()
        if self.use_w:
            self.f0.train()
        if self.use_encoder:
            self.encoder.train()

    def eval(self):
        self.generator.eval()
        self.discriminator.eval()
        self.alpha.eval()
        if self.use_w:
            self.f0.eval()
        if self.use_encoder:
            self.encoder.eval()

    def save(self):
        self.generator.save(self.model_dir)
        self.discriminator.save(self.model_dir)

        torch.save(self.alpha.to("cpu").state_dict(), os.path.join(self.model_dir, "alpha.pt"))
        self.alpha.to(self.device)


    def load(self):
        self.generator.load(self.model_dir)
        self.discriminator.load(self.model_dir)

        self.alpha.to("cpu")
        self.alpha.load_state_dict(torch.load(os.path.join(self.model_dir, "alpha.pt")))
        self.alpha.to(self.device)



class MicroAIRL(MeshAIRL):
    def __init__(self, dt=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dt = dt
        self.micro_stat = None
        self.macro_stat = None
        self.micro_mean = None
        self.micro_cov = None
        self.macro_mean = None
        self.macro_cov = None

    def calc_statistics(self, out_dir=None):
        # utility, interaction, traffic, velocity
        self.dataset.return_id = True
        results_micro = [None for _ in range(self.output_channel)]  # [df([utility, interaction, traffic, velocity])], len: output_channel
        results_macro = [None for _ in range(self.output_channel)]  # [df([utility, interaction, traffic, velocity])], len: output_channel
        for channel in range(self.output_channel):
            utility_micro = []
            interaction_micro = []
            traffic_micro = []  # mode wise
            velocity_micro = []
            utility_macro = {}  # {"id", utility_sum}
            interaction_macro = {}  # {"id", interaction_sum}
            velocity_macro = {}  # {"id", [velocity]}

            self.dataset.set_channel(channel)
            dataloader = DataLoader(self.dataset, batch_size=4, shuffle=False, num_workers=2)
            self.generators[channel].eval()
            self.discriminators[channel].eval()
            for state, context, next_state, mask, positions, pis, aids in dataloader:
                # state: (bs, prop_dim, 2d+1, 2d+1)
                # context: (bs, context_num, 2d+1, 2d+1)
                # next_state: (bs, 2d+1, 2d+1), 0 or 1
                # mask: (bs, 2d+1, 2d+1), 0 or 1
                # positions: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                # pis: (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                # aids: (bs)
                inputs = torch.cat((state, context), dim=1).to(self.device)  # (bs, prop_dim+context_num, 2d+1, 2d+1)
                next_state = next_state.to(self.device)  # (bs, 2d+1, 2d+1)
                mask = mask.to(self.device)  # (bs, 2d+1, 2d+1)
                positions = positions.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                pis = pis.to(self.device)  # (bs, max_agent_num, output_channel, 2d+1, 2d+1)
                aids = aids.numpy().tolist()  # (bs)

                logits = self.generators[channel](inputs)  # raw_data_fake requires_grad=True, (bs, 2d+1, 2d+1)
                logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                              device=logits.device))

                f_val, util, ext, val = self.discriminators[channel].get_vals(inputs, positions, pis)  # f_val, util, val: (bs, 2d+1, 2d+1), ext: (bs, num_agents, 2d+1, 2d+1)
                util = (util * next_state).sum(dim=(1, 2)).clone().detach().cpu().numpy().tolist()  # (bs)
                ext = (ext * next_state.unsqueeze(1)).clone().detach().cpu().numpy().sum(axis=(1, 2, 3)).tolist()  # (bs)
                traf = (positions.clone().detach().cpu().numpy().sum(axis=(3, 4)) > 0).sum(axis=1) # (bs, output_channel)
                traf[:, channel] = traf[:, channel] + 1  # (bs, output_channel)
                traf = traf.tolist()
                dxs = (next_state.clone().detach().cpu().numpy() * np.arange(2 * self.dataset.d + 1).reshape(1, 1, -1)).sum(
                    axis=(1, 2)) - self.dataset.d  # (bs)
                dys = (next_state.clone().detach().cpu().numpy() * np.arange(2 * self.dataset.d + 1).reshape(1, -1, 1)).sum(
                    axis=(1, 2)) - self.dataset.d  # (bs)
                velocity = (np.sqrt((dxs * self.mnw_data.x_size) ** 2 + (dys * self.mnw_data.y_size) ** 2) / self.dt).tolist()

                utility_micro.extend(util)
                interaction_micro.extend(ext)
                traffic_micro.extend(traf)
                velocity_micro.extend(velocity)

                for i, aid in enumerate(aids):
                    if aid not in utility_macro:
                        utility_macro[aid] = 0
                        interaction_macro[aid] = 0
                        velocity_macro[aid] = []
                    utility_macro[aid] += util[i]
                    interaction_macro[aid] += ext[i]
                    velocity_macro[aid].append(velocity[i])
            traffic_macro = len(utility_macro)
            val = []
            for i in range(len(utility_micro)):
                val.append([utility_micro[i], interaction_micro[i], *traffic_micro[i], velocity_micro[i]])
            results_micro[channel] = pd.DataFrame(val, columns=["utility", "interaction", *[f"traffic_{i}" for i in range(self.output_channel)], "velocity"])
            val = []
            for aid in utility_macro:
                val.append([utility_macro[aid], interaction_macro[aid], traffic_macro, np.mean(velocity_macro[aid])])
            results_macro[channel] = pd.DataFrame(val, columns=["utility", "interaction", "traffic", "velocity"])
            if out_dir is not None:
                results_micro[channel].to_csv(os.path.join(out_dir, "micro_{}.csv".format(channel)))
                results_macro[channel].to_csv(os.path.join(out_dir, "macro_{}.csv".format(channel)))
        self.micro_stat = results_micro
        self.macro_stat = results_macro

    def calc_multivariate(self):
        if self.micro_stat is None or self.macro_stat is None:
            self.calc_statistics()

        micro_mean = [None for _ in range(self.output_channel)]
        micro_cov = [None for _ in range(self.output_channel)]
        macro_mean = [None for _ in range(self.output_channel)]
        macro_cov = [None for _ in range(self.output_channel)]

        for channel in range(self.output_channel):
            micro_mean[channel] = self.micro_stat[channel].mean()
            micro_cov[channel] = self.micro_stat[channel].cov()
            macro_mean[channel] = self.macro_stat[channel].mean()
            macro_cov[channel] = self.macro_stat[channel].cov()

        self.micro_mean = micro_mean
        self.micro_cov = micro_cov
        self.macro_mean = macro_mean
        self.macro_cov = macro_cov

        col1_micro = ["utility", "interaction", *[f"traffic_{i}" for i in range(self.output_channel)]]
        col1_macro = ["utility", "interaction", "traffic"]
        col2 = ["velocity"]

        self.mu1s_micro = [micro_mean[channel][col1_micro].values for channel in range(self.output_channel)]
        self.mu2s_micro = [micro_mean[channel][col2].values for channel in range(self.output_channel)]
        self.sig12sig22Ts_micro = [micro_cov[channel].loc[col1_micro, col2].values @ np.linalg.pinv(micro_cov[channel].loc[col2, col2].values) for channel in range(self.output_channel)]
        self.cov12s_micro = [micro_cov[channel].loc[col1_micro, col1_micro].values - micro_cov[channel].loc[col1_micro, col2].values @ np.linalg.pinv(micro_cov[channel].loc[col2, col2].values) @ micro_cov[channel].loc[col2, col1_micro].values for channel in range(self.output_channel)]

        self.mu1s_macro = [macro_mean[channel][col1_macro].values for channel in range(self.output_channel)]
        self.mu2s_macro = [macro_mean[channel][col2].values for channel in range(self.output_channel)]
        self.sig12sig22Ts_macro = [macro_cov[channel].loc[col1_macro, col2].values @ np.linalg.pinv(macro_cov[channel].loc[col2, col2].values) for channel in range(self.output_channel)]
        self.cov12s_macro = [macro_cov[channel].loc[col1_macro, col1_macro].values - macro_cov[channel].loc[col1_macro, col2].values @ np.linalg.pinv(macro_cov[channel].loc[col2, col2].values) @ macro_cov[channel].loc[col2, col1_macro].values for channel in range(self.output_channel)]


    def conditional_dist_v(self, channel, v):
        # velocity [m/s]
        # conditional distribution of utility, interaction, traffic given velocity
        if self.micro_mean is None:
            self.calc_multivariate()

        # micro
        mu12_micro = self.mu1s_micro[channel] + self.sig12sig22Ts_micro[channel] @ (v - self.mu2s_micro[channel])
        # macro
        mu12_macro = self.mu1s_macro[channel] + self.sig12sig22Ts_macro[channel] @ (v - self.mu2s_macro[channel])

        return mu12_micro, self.cov12s_micro[channel], mu12_macro, self.cov12s_macro[channel]

    def sample_given_v(self, channel, v):
        # utility, interaction, traffic given velocity
        mu12_micro, sigma12_micro, mu12_macro, sigma12_macro = self.conditional_dist_v(channel, v)
        sample_micro = np.random.multivariate_normal(mu12_micro, sigma12_micro)
        sample_macro = np.random.multivariate_normal(mu12_macro, sigma12_macro)
        return sample_micro, sample_macro




