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
        self.use_encoder = image_data is not None and encoder is not None
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
            self._load_image_feature()  # self.comp_feature: (link_num, comp_dim)
        if self.use_w:
            self.f0 = self.f0.to(device)
    
    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
              train_ratio=0.8, max_train_num=10000, d_epoch=5, lr_f0=0.01, lr_e=0.0, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file)  #loss_e,loss_g,loss_d,loss_e_val,loss_g_val,loss_d_val,criteria

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        train_encoder = lr_e > 0.0 and self.use_encoder
        if self.use_w:
            optimizer_0 = optim.Adam(self.f0.parameters(), lr=lr_f0)
        if train_encoder:
            optimizer_e = optim.Adam(self.encoder.parameters(), lr=lr_e)

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle}
        dataloaders_real = [[DataLoader(tmp, **dataset_kwargs)
                            for tmp in dataset.split_into((min(train_ratio, max_train_num / len(dataset) * batch_size), 1-train_ratio))]
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
                    batch_real = [tensor(tmp.to_dense(), dtype=torch.float32, device=self.device) for tmp in batch_real]
                    bs = batch_real[0].shape[0]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)
                    if self.use_encoder:
                        # append image_feature to the original feature
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        logits = self.generator(batch_real[0], w=w)  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    else:
                        logits = self.generator(batch_real[0])  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    logits = torch.where(batch_real[1].unsqueeze(1) > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                              device=logits.device))
                    pi = self.get_pi_from_logits(logits)
                    batch_fake = self.datasets[i].get_fake_batch(batch_real, logits.clone().detach()[:, i, :, :]) # batch_fake requires_grad=False

                    for j in range(d_epoch):
                        # loss function calculation
                        log_d_real = self.get_log_d(batch_real, pi, i, w=w)  # discriminator inference performed inside
                        log_d_fake = self.get_log_d(batch_fake, pi, i, w=w)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(log_d_real, log_d_fake, self.hinge_loss)

                        # discriminator update
                        optimizer_d.zero_grad()
                        with detect_anomaly():
                            loss_d.backward(retain_graph=True)
                        optimizer_d.step()

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

                        del log_d_fake, log_d_real, loss_d, loss_g

                # validation
                self.eval()
                ll = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                for batch_real in dataloader_val:
                    batch_real = [tensor(tmp.to_dense(), dtype=torch.float32, device=self.device) for tmp in batch_real]
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
                    logits = torch.where(batch_real[1].unsqueeze(1) > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                                        device=logits.device))
                    pi = self.get_pi_from_logits(logits)
                    batch_fake = self.datasets[i].get_fake_batch(batch_real,
                                                                 logits[:, i, :, :])  # batch_fake requires_grad=False
                    log_d_real = self.get_log_d(batch_real, pi, i, w=w)
                    log_d_fake = self.get_log_d(batch_fake, pi, i, w=w)

                    loss_g, loss_d = self.loss(log_d_real, log_d_fake, self.hinge_loss)

                    mode_loss_g_val.append(loss_g.clone().detach().cpu().item())
                    mode_loss_d_val.append(loss_d.clone().detach().cpu().item())

                    ll_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_creteria(batch_real, pi, i, w=w)
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
                self.generator.save(self.model_dir)
                self.discriminator.save(self.model_dir)

            t2 = time.perf_counter()
            print("epoch: {}, loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                e, epoch_loss_g_val[-1], epoch_loss_d_val[-1], criteria, t2 - t1))

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

    def loss(self, log_d_real, log_d_fake, hinge_loss=False):
        # log_d_real, log_d_fake: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if hinge_loss:
            l_g = -log_d_fake[0][0] + log_d_fake[0][1]
            l_d = torch.max(-log_d_real[1][0], self.hinge_thresh) + torch.max(-log_d_fake[1][1], self.hinge_thresh)
        else:
            l_g = -log_d_fake[0][0] + log_d_fake[0][1]
            l_d = -log_d_real[1][0] - log_d_fake[1][1]

        return l_g.sum(), l_d.sum()

    def get_log_d(self, batch, pi, i, w=None):
        # batch
        # Grid: inputs, masks, next_links, link_idxs, hs
        # Emb: global_state, adj_matrix, transition_matrix, hs

        # d retain_grad=True
        # Grid: (sum(links), 3, 3) corresponds to next_links
        # Emb: (trip_num, link_num, link_num) sparse corresponds to transition_matrix
        # output: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if self.use_index:
            input, mask, _, _, _ = batch
        else:
            input, mask, _, _ = batch
        f_val = self.discriminator(input, pi, i=i, w=w)
        f_val_masked = f_val * mask
        f_val_clone_masked = f_val_masked.clone().detach()
        pi_i_clone = pi.clone().detach()[:, i, :, :]
        # f_val: (bs, oc, *choice_space), pi: (bs, oc, *choice_space)
        log_d_g = f_val_clone_masked - log(torch.exp(f_val_clone_masked) + pi[:, i, :, :])
        log_1_d_g = log(pi[:, i, :, :]) - log(torch.exp(f_val_clone_masked) + pi[:, i, :, :])

        log_d_d = f_val_masked - log(torch.exp(f_val_masked) + pi_i_clone)
        log_1_d_d = log(pi_i_clone) - log(torch.exp(f_val_masked) + pi_i_clone)

        return (log_d_g, log_1_d_g), (log_d_d, log_1_d_d)

    def get_creteria(self, batch, pi, i, w=None):
        # ll, TP, FP, FN, TN
        if self.use_index:
            input, mask, next_mask, _, _ = batch
        else:
            input, mask, next_mask, _ = batch
        f_val, util, ext, val = self.discriminator.get_vals(input, pi, i=i, w=w)  # (bs, *choice_space)
        q = util + ext + self.discriminator.gamma * val
        q = torch.where(mask > 0, q, tensor(-9e15, dtype=torch.float32, device=q.device))
        # choose maximum q
        if self.use_index:  # choose from choice_space
            q = q.view(q.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)
            next_mask = next_mask.view(next_mask.shape[0], -1)
        pi_q = F.softmax(q, dim=-1)
        ll = log((pi_q * next_mask).sum(dim=-1)).sum()

        pred_mask = (q == q.max(dim=1, keepdim=True)[0]).to(torch.float32)
        tp = (next_mask * pred_mask).sum()
        fp = (mask * (1-next_mask) * pred_mask).sum()
        tn = (mask * (1-next_mask) * (1-pred_mask)).sum()
        fn = (next_mask * (1-pred_mask)).sum()
        return ll, tp, fp, tn, fn

    
    def cat_image_feature(self, batch, w=None):
        # w: (bs, w_dim)
        # index: (bs, 9)
        image_feature = self.encoder(self.comp_feature, w=w)  # (bs, link_num, emb_dim)

        # inputs: (bs, c, 3, 3) or (bs, link_num, feature_num). c or feature_num are total_feature_num + emb_dim
        if self.use_index:
            # image_feature: tensor(link_num, emb_dim)
            # batch[3]: tensor(bs, 9)
            image_feature_tmp = image_feature[batch[3], :].transpose(1, 2).view(batch[3].shape[0], -1, 3, 3)
            inputs = torch.cat((batch[0], image_feature_tmp), dim=1)
        else:
            print(image_feature.shape, batch[0].shape)
            inputs = torch.cat((batch[0], image_feature), dim=2)

        batch = [inputs, *batch[1:]]
        return batch

    def generate(self, bs, w=None):
        # only for gnn
        if not self.use_encoder or self.use_index:
            raise Exception("This method is only for GNNEmb.")
        self.eval()

        inputs = [tensor(self.real_data.feature_matrix, dtype=torch.float32, device=self.device).repeat(bs, 1, 1)]
        if self.use_encoder:
            # append image_feature to the original feature
            inputs = self.cat_image_feature(inputs, w=w)
        inputs = inputs[0]  # (bs, link_num, feature_num)
        adj_matrix = self.generator(inputs, None, w=w)  # (bs, oc, link_num, link_num)
        datasets = [self.datasets[i].get_dataset_from_adj(adj_matrix[:, i, :, :]) for i in range(self.output_channel)]
        return datasets

    def get_pi_from_logits(self, logits):
        # logits: tensor(bs, oc, *choice_space)
        # pi: (bs, oc, *choice_space)
        # self.use_index: logits are local -> choice from choice_space
        # not self.use_index: logits are global -> choice from dim=-1
        if self.use_index:
            shape = logits.shape
            pi = F.softmax(logits.view(shape[0], shape[1], -1), dim=2).view(shape)
        else:
            pi = F.softmax(logits, dim=-1)
        return pi

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

    def _load_image_feature(self):
        comp_feature = None
        comp_dim = None
        for i in range(len(self.image_data)):
            tmp_feature = self.image_data.load_compressed(i)
            if tmp_feature is not None:
                tmp_feature = np.expand_dims(tmp_feature, 0)
                comp_dim = tmp_feature.shape[-1]
                if comp_feature is None:
                    comp_feature = np.zeros((i, comp_dim), dtype=np.float32)
            elif comp_dim is not None:
                tmp_feature = np.zeros((1, comp_dim), dtype=np.float32)
            if tmp_feature is not None:
                comp_feature = np.concatenate((comp_feature, tmp_feature), axis=0)
        if comp_feature is None:
            print("No image feature is loaded.")
        else:
            self.comp_feature = tensor(comp_feature, dtype=torch.float32, device=self.device)


