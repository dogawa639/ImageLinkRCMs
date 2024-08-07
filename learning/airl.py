import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import detect_anomaly, Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shap

import time
import os

from utility import *
from preprocessing.dataset import PatchDataset
from models.general import log
from logger import Logger
__all__ = ["AIRL"]


class AIRL:
    def __init__(self, generator, discriminator, use_index, datasets, model_dir, image_data=None, encoder=None, h_dim=10, emb_dim=10, f0=None,
                 hinge_loss=False, hinge_thresh=0.6, patch_size=256, use_compressed_image=True, device="cpu"):
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
        self.image_data = image_data  # CompressedImageData if use_compressed_image else LinkImageData
        self.hinge_loss = hinge_loss
        self.hinge_thresh = -log(tensor(hinge_thresh, dtype=torch.float32, device=device, requires_grad=False))
        self.patch_size = patch_size
        self.use_compressed_image = use_compressed_image
        self.device = device

        self.sln = self.use_w

        self.link_num = len(self.datasets[0].nw_data.lids)
        self.output_channel = len(self.datasets)

        if self.use_encoder:
            self.encoder = self.encoder.to(device)
            if self.use_compressed_image:
                self._load_image_feature()  # self.comp_feature: (link_num, comp_dim)
        if self.use_w:
            self.f0 = self.f0.to(device)

        self.explainer = None
    
    def train_models(self, conf_file, epochs, batch_size, lr_g, lr_d, shuffle,
              ratio=(0.8, 0.2), max_train_num=10000, d_epoch=5, lr_f0=0.01, lr_e=0.0, image_file=None):
        log = Logger(os.path.join(self.model_dir, "log.json"), conf_file, fig_file=os.path.join(self.model_dir, "log.png"), figsize=(6.4, 4.8 * 3))  #loss_e,loss_g,loss_d,loss_e_val,loss_g_val,loss_d_val,accuracy,ll,criteria

        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        train_encoder = lr_e > 0.0 and self.use_encoder
        if self.use_w:
            optimizer_0 = optim.Adam(self.f0.parameters(), lr=lr_f0)
        if train_encoder:
            optimizer_e = optim.Adam(self.encoder.parameters(), lr=lr_e)

        dataset_kwargs = {"batch_size": batch_size, "shuffle": shuffle}
        dataloaders_real = [[DataLoader(tmp, **dataset_kwargs)
                            for tmp in dataset.split_into((min(ratio[0], max_train_num / len(dataset) * batch_size), ratio[1]))]
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
                    batch_real = [tmp.to_dense().clone().detach().to(device=self.device) for tmp in batch_real]
                    bs = batch_real[0].shape[0]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)
                    if self.use_encoder:
                        # append image_feature to the original feature
                        self.encoder.train()
                        batch_real = self.cat_image_feature(batch_real, w=w)

                    if self.sln:
                        logits = self.generator(batch_real[0], w=w)  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    else:
                        logits = self.generator(batch_real[0])  # raw_data_fake requires_grad=True, (bs, oc, *choice_space)
                    mask = batch_real[1].unsqueeze(1)
                    if self.use_index:
                        mask = mask.view(-1, 1, logits.shape[-2], logits.shape[-1])
                    logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                              device=logits.device))
                    pi = self.get_pi_from_logits(logits)

                    for j in range(d_epoch):
                        # loss function calculation
                        log_d_g, log_d_d = self.get_log_d(batch_real, pi, i, w=w)  # discriminator inference performed inside
                        loss_g, loss_d = self.loss(log_d_g, log_d_d, self.hinge_loss)

                        # discriminator update
                        optimizer_d.zero_grad()
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
                            loss_g.backward(retain_graph=True)

                            if self.use_w:
                                optimizer_0.step()
                            if train_encoder:
                                optimizer_e.step()

                            # generator update
                            optimizer_g.step()
                            mode_loss_g.append(loss_g.clone().detach().cpu().item())

                            pi = pi.detach()
                            if w is not None:
                                w = w.detach()
                            batch_real = [tmp.detach() for tmp in batch_real]

                        del log_d_g, log_d_d, loss_d, loss_g

                # validation
                self.eval()
                ll = 0.0
                tp = 0.0
                fp = 0.0
                tn = 0.0
                fn = 0.0
                for batch_real in dataloader_val:
                    batch_real = [tmp.to_dense().clone().detach().to(device=self.device) for tmp in batch_real]
                    w = None
                    if self.use_w:
                        w = self.f0(batch_real[-1])  # (bs, w_dim)
                    if self.use_encoder:
                        # append image_feature to the original feature
                        self.encoder.eval()
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
                accuracy = (tp) / (tp + fp)
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

            if e % max(int(epochs / 3), 1) == 0:
                optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g / 10)
                optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d / 10)
                if self.use_w:
                    optimizer_0 = optim.Adam(self.f0.parameters(), lr=lr_f0 / 10)
                if train_encoder:
                    optimizer_e = optim.Adam(self.encoder.parameters(), lr=lr_e / 10)

            if e % 5 == 0:
                if not os.path.exists(os.path.join(self.model_dir, "joint_prob")):
                    os.mkdir(os.path.join(self.model_dir, "joint_prob"))
                self.get_joint_prob(os.path.join(self.model_dir, "joint_prob", f"joint_prob_{e}.csv"))

            t2 = time.perf_counter()
            print(f"epoch: {e}")
            for i in range(len(epoch_loss_g)):
                print("   mode: {}  loss_g_val: {:.4f}, loss_d_val: {:.4f}, criteria: {:.4f}, time: {:.4f}".format(
                i, epoch_loss_g_val[i], epoch_loss_d_val[i], epoch_criteria[i], t2 - t1))

        if image_file is not None:
            log.save_fig(image_file)
        log.close()

    def test(self, datasets):
        dataset_kwargs = {"batch_size": 16, "shuffle": False}
        dataloaders_test = [DataLoader(dataset, **dataset_kwargs)for dataset in datasets]

        print("test start.")
        self.eval()
        ll0_test = []
        accuracy_test = []
        ll_test = []
        criteria_test = []
        for i, dataloader_test in enumerate(dataloaders_test):
            mode_loss_g_test = []
            mode_loss_d_test = []
            ll0 = 0.0
            ll = 0.0
            tp = 0.0
            fp = 0.0
            tn = 0.0
            fn = 0.0
            for batch_real in dataloader_test:
                batch_real = [tmp.to_dense().clone().detach().to(self.device) for tmp in batch_real]
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

                mode_loss_g_test.append(loss_g.clone().detach().cpu().item())
                mode_loss_d_test.append(loss_d.clone().detach().cpu().item())

                ll_tmp, tp_tmp, fp_tmp, tn_tmp, fn_tmp = self.get_criteria(batch_real, pi, i, w=w)
                ll += ll_tmp.detach().cpu().item()
                tp += tp_tmp.detach().cpu().item()
                fp += fp_tmp.detach().cpu().item()
                tn += tn_tmp.detach().cpu().item()
                fn += fn_tmp.detach().cpu().item()
                ll0 -= np.log(mask.clone().detach().cpu().numpy().reshape(mask.shape[0], -1).sum(1)).sum()
            accuracy = (tp) / (tp + fp)
            criteria = -ll

            ll0_test.append(ll0)
            accuracy_test.append(accuracy)
            ll_test.append(ll)
            criteria_test.append(criteria)
        for i in range(len(accuracy_test)):
            print("test accuracy {}: {:.4f}, ll0: {:.4f}, ll: {:.4f}, criteria: {:.4f}".format(i, accuracy_test[i], ll0_test[i], ll_test[i],
                                                                                  criteria_test[i]))
        # save result
        with open(os.path.join(self.model_dir, "test_result.txt"), "w") as f:
            for i in range(len(accuracy_test)):
                f.write("test accuracy {}: {:.4f}, ll0: {:.4f}, ll: {:.4f}, criteria: {:.4f}\n".format(i, accuracy_test[i], ll0_test[i], ll_test[i], criteria_test[i]))
        # create results_dict with suffix i
        results_dict = dict()
        for i in range(len(accuracy_test)):
            results_dict[f"accuracy_{i}"] = accuracy_test[i]
            results_dict[f"ll0_{i}"] = ll0_test[i]
            results_dict[f"ll_{i}"] = ll_test[i]
            results_dict[f"criteria_{i}"] = criteria_test[i]
                        
        print("test end.")
        return results_dict

    def get_joint_prob(self, csv_path=None):
        f = self.datasets[0].nw_data.feature_num
        c = self.datasets[0].nw_data.context_feature_num
        lid2idx = {lid: i for i, lid in enumerate(self.datasets[0].nw_data.lids)}
        joint_pis = []
        joint_pis_q = []
        for tmp_link in self.datasets[0].nw_data.lids:
            trip_input = np.zeros((1, f + c, 3, 3), dtype=np.float32)
            trip_mask = np.zeros((1, 1, 9), dtype=np.float32)
            trip_link_idxs = np.zeros((1, 9), dtype=np.int32)

            feature_mat, _ = self.datasets[0].nw_data.get_feature_matrix(tmp_link, normalize=True)
            action_edge = self.datasets[0].nw_data.get_action_edge(tmp_link)

            trip_input[0, 0:f, :, :] = feature_mat
            trip_mask[0, :] = [len(edges) > 0 and j != 4 for j, edges in enumerate(action_edge)]
            trip_link_idxs[0, :] = [lid2idx[np.random.choice(edges)] if len(edges) > 0 else 0 for edges in action_edge]
            trip_input = torch.tensor(trip_input, dtype=torch.float32, device=self.device)
            trip_mask = torch.tensor(trip_mask, dtype=torch.bool, device=self.device)
            trip_link_idxs = torch.tensor(trip_link_idxs, dtype=torch.int32, device=self.device)

            trip_input, _, _, _ = self.cat_image_feature([trip_input, None, None, trip_link_idxs], w=None)

            logits = self.generator(trip_input)  # (1, oc, *choice_space)
            logits = torch.where(trip_mask.reshape(1, 1, *logits.shape[2:]) > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                                  device=logits.device))
            pi = self.get_pi_from_logits(logits).clone().detach()
            joint_pis.append(pi.cpu().numpy().squeeze(0).reshape(logits.shape[1], -1))  # (oc, len(choice_space))

            # get choice prob based on Q function
            _, util, ext, val = self.discriminator.get_vals(trip_input, pi)
            logits_q = util + ext + 0.9 * val
            logits_q = torch.where(trip_mask.reshape(1, 1, *logits_q.shape[2:]) > 0, logits_q, tensor(-9e15, dtype=torch.float32, device=logits_q.device))
            pi_q = self.get_pi_from_logits(logits_q).clone().detach()
            joint_pis_q.append(pi_q.cpu().numpy().squeeze(0).reshape(logits_q.shape[1], -1)) 

        joint_pis = np.array(joint_pis)  # (link_num, oc, len(choice_space))
        joint_pis_q = np.array(joint_pis_q)  # (link_num, oc, len(choice_space))
        if csv_path is not None:
            df = pd.DataFrame(joint_pis.reshape(joint_pis.shape[0], -1), columns=[f"pi_{i}(l{j}))" for i in range(joint_pis.shape[1]) for j in range(joint_pis.shape[2])])
            df["lid"] = self.datasets[0].nw_data.lids
            df.to_csv(csv_path, index=False)

            df = pd.DataFrame(joint_pis_q.reshape(joint_pis_q.shape[0], -1), columns=[f"pi_{i}(l{j}))" for i in range(joint_pis_q.shape[1]) for j in range(joint_pis_q.shape[2])])
            df["lid"] = self.datasets[0].nw_data.lids
            df.to_csv(csv_path.replace(".csv", "_q.csv"), index=False)
        
        return joint_pis, joint_pis_q

    def get_shap(self, datasets, i, sample_num=10):
        if self.use_w:
            raise NotImplementedError("get_shap is not implemented when use_w is True.")
        # shap for generator
        dataset_kwargs = {"batch_size": 1, "shuffle": False}
        dataloaders_test = [DataLoader(dataset, **dataset_kwargs) for dataset in datasets]
        self.eval()

        inputs = None
        for tmp_i, dataloader_test in enumerate(dataloaders_test):
            if tmp_i != i:
                continue
            for batch_real in dataloader_test:
                # Grid: (sum(links), 3, 3) corresponds to next_links
                # Emb: (trip_num, link_num, link_num) sparse, corresponds to transition_matrix
                batch_real = [tmp.to_dense().clone().detach().to(device=self.device) for tmp in batch_real]
                bs = batch_real[0].shape[0]
                w = None
                if self.use_w:
                    w = self.f0(batch_real[-1])  # (bs, w_dim)
                if self.use_encoder:
                    # append image_feature to the original feature
                    batch_real = self.cat_image_feature(batch_real, w=w)

                if inputs is None:
                    inputs = batch_real[0]
                else:
                    inputs = torch.cat([inputs, batch_real[0]], dim=0)
                if inputs.shape[0] >= 2 * sample_num:
                    break
            break

        inputs_test = inputs[sample_num:2 * sample_num, ...]
        inputs = inputs[:sample_num, ...]
        inputs_shape = inputs.shape
        if self.use_index:
            # inputs: (bs, c, 3, 3)
            link_feature_num = inputs_shape[1]
            inputs = inputs.clone().detach().cpu().numpy().reshape(inputs_shape[0], -1)
            self.generator.eval()
            self.generator.to("cpu")
            f = lambda x: self.generator(Variable(torch.from_numpy(x.reshape(-1, *inputs_shape[1:]))),
                                         i=i).detach().cpu().numpy().sum((1, 2))  # x: (bs, num_var)
            print(inputs.shape)
            print(f(inputs).shape)
            explainer = shap.KernelExplainer(f, inputs)

            inputs_shape = inputs_test.shape
            inputs_test = inputs_test.detach().cpu().numpy().reshape(inputs_shape[0], -1)
            print(f(inputs_test).shape)
            sv = explainer.shap_values(inputs_test)  # (bs, num_var)
            print(sv.shape)
            print("shap_values", sv)
            sv_reshaped = sv.reshape(-1, *inputs_shape[1:]).transpose(0, 2, 3, 1).reshape(-1, link_feature_num)
        else:
            # inputs: (bs, link_num, c)
            link_feature_num = inputs_shape[2]
            inputs = inputs.clone().detach().cpu().numpy().reshape(inputs_shape[0], -1)
            self.generator.eval()
            self.generator.to("cpu")
            f = lambda x: self.generator(Variable(torch.from_numpy(x.reshape(-1, *inputs_shape[1:]))),
                                         i=i).detach().cpu().numpy().sum((1, 2))
            print(inputs.shape)
            print(f(inputs).shape)
            explainer = shap.KernelExplainer(f, inputs)

            inputs_shape = inputs_test.shape
            inputs_test = inputs_test.detach().cpu().numpy().reshape(inputs_shape[0], -1)  # (bs, num_var)
            print(f(inputs_test).shape)
            sv = explainer.shap_values(inputs_test)
            print(sv.shape)
            print("shap_values", sv)
            sv_reshaped = sv.reshape(-1, *inputs_shape[1:]).reshape(-1, link_feature_num)

        shap.initjs()
        shap.summary_plot(sv_reshaped, feature_names=[f'Feature{i}' for i in range(link_feature_num)], plot_type="dot", show=False)
        plt.savefig(os.path.join(self.model_dir, f"shap_dot_{i}.png"))
        plt.show()
        shap.initjs()
        shap.summary_plot(sv_reshaped, feature_names=[f'Feature{i}' for i in range(link_feature_num)], plot_type="bar",
                          show=False)
        plt.savefig(os.path.join(self.model_dir, f"shap_bar_{i}.png"))
        plt.show()
        self.generator.to(self.device)
        return sv_reshaped

    def show_attention_map(self, idxs, show=True, save_file=None):
        len_imgs = 0
        fig = plt.figure(figsize=(5, 5 * len(idxs)))
        attens = []
        for i in idxs:
            images = self.image_data.load_link_image(i)  # list(tensor(n, c, h, w))
            len_imgs = max(len_imgs, len(images))
            attens_tmp = []
            for num_source, img in enumerate(images):
                img = img.to(self.device)
                compressed = self.encoder.compress(img, num_source=num_source)
                if type(compressed) is not tuple:
                    print("Attention map is only available for ViT encoder.")
                    raise NotImplementedError
                atten = compressed[1].detach().cpu().numpy().squeeze(axis=0)
                ax = fig.add_subplot(len(idxs), len_imgs * 2, (len_imgs * 2) * i + num_source + 1)
                img_show = img.detach().cpu().numpy().squeeze(axis=0).transpose(1, 2, 0)
                img_show = ((img_show - img_show.min()) / (img_show.max() - img_show.min()) * 255).astype(np.uint8)
                ax.imshow(img_show)
                ax = fig.add_subplot(len(idxs), len_imgs * 2, (len_imgs * 2) * i + num_source + 1 + len_imgs)
                ax.imshow(atten, interpolation='bilinear')
                attens_tmp.append(atten)
                if save_file is not None:
                    plt.savefig(save_file.replace(".png", f"_{i}_{num_source}.png"))
            attens.append(attens_tmp)
        if show:
            plt.show()
        print("show_attention_map end.")
        return attens

    def show_encoder_shap(self, idxs, source=0, show=True, save_file=None):
        # img_tensor: tensor(bs2, c, h, width) or (c, h, width)
        self.eval()
        self.encoder.train_all()
        fig = plt.figure(figsize=(5, 5 * len(idxs)))
        for i in idxs:
            images = self.image_data.load_link_image(i)  # [tensor(n, c, h, w)]
            if images is None:
                print(f"Image for link idx {i} does not exist.")
                continue
            # only use first source
            img_tensor = images[0].to(self.device)

            if self.explainer is None:
                cnt = 0
                backgrounds = []
                while len(backgrounds) < 30 and cnt < 100:
                    tmp = self.image_data.load_link_image(cnt)
                    if tmp is not None:
                        backgrounds.append(tmp[0])
                    cnt += 1
                backgrounds = torch.cat(backgrounds, dim=0).to(self.device)
                self.explainer = shap.DeepExplainer(_Encoder(self.encoder, source), backgrounds)

            shap_values = self.explainer.shap_values(img_tensor).reshape(img_tensor.shape)  # [(bs, c, w, h)] or [(c, w, h)]
            shap_values = [sv if len(sv.shape) == 4 else np.expand_dims(sv, 0) for sv in shap_values]  # [(bs, c, w, h)]
            if show:
                shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]  # [(bs, w, h, c)]
                img_numpy = np.swapaxes(np.swapaxes(img_tensor.clone().detach().cpu().numpy(), 1, -1), 1, 2)  # (n, w, h, c)
                shap.image_plot(shap_numpy, (img_numpy * np.array([44.63, 45.54, 45.94]).reshape(1, 1, 1, -1) + np.array([110.96, 110.76, 102.67]).reshape(1, 1, 1, -1)).astype(np.uint8), show=False)
                if save_file is not None:
                    path = save_file.replace(".png", f"_{i}.png")
                    plt.savefig(path)
                else:
                    plt.show()
        return shap_values
    
    def show_img_shap(self, img_tensor, show=True, save_file=None):
        # img_tensor: tensor(bs2, c, h, width)
        self.eval()
        self.encoder.train()
        fig = plt.figure(figsize=(5, 5))
        img_tensor = img_tensor.to(self.device)

        if self.explainer is None:
            cnt = 0
            backgrounds = []
            while len(backgrounds) < 5 and cnt < 100:
                tmp = self.image_data.load_link_image(cnt)
                if tmp is not None:
                    backgrounds.append(tmp[0])
                cnt += 1
            backgrounds = torch.cat(backgrounds, dim=0).to(self.device)
            self.explainer = shap.DeepExplainer(_Encoder(self.encoder, 0), backgrounds)

        shap_values = self.explainer.shap_values(img_tensor)  # [(bs, c, w, h)] or [(c, w, h)]
        shap_values = [sv if len(sv.shape) == 4 else np.expand_dims(sv, 0) for sv in shap_values]  # [(bs, c, w, h)]
        if show:
            shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]  # [(bs, w, h, c)]
            img_numpy = np.swapaxes(np.swapaxes(img_tensor.clone().detach().cpu().numpy(), 1, -1), 1, 2)  # (n, w, h, c)
            shap.image_plot(shap_numpy, (img_numpy * np.array([44.63, 45.54, 45.94]).reshape(1, 1, 1, -1) + np.array([110.96, 110.76, 102.67]).reshape(1, 1, 1, -1)).astype(np.uint8), show=False)
            if save_file is not None:
                plt.savefig(save_file)
            else:
                plt.show()
        return shap_values

    def loss(self, log_d_g, log_d_d, hinge_loss=False):
        # log_d_g, log_d_d: (log d_for_g, log 1-d_for_g), (log d_for_d, log 1-d_for_d)
        if hinge_loss:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = torch.max(-log_d_d[0], self.hinge_thresh) + torch.max(-log_d_d[1], self.hinge_thresh)
        else:
            l_g = -log_d_g[0] + log_d_g[1]
            l_d = -log_d_d[0] - log_d_d[1]

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
            input, mask, next_link, _, _ = batch
            mask = mask.view(mask.shape[0], input.shape[-2], input.shape[-1])
            next_link = next_link.view(next_link.shape[0], input.shape[-2], input.shape[-1])
        else:
            input, mask, next_link, _ = batch
        f_val = self.discriminator(input, pi, i=i, w=w)
        f_val_masked = torch.where(mask > 0.99, f_val, tensor(0.0, dtype=torch.float32, device=f_val.device))
        f_val_clone_masked = f_val_masked.clone().detach()
        pi = torch.where(mask > 0.99, pi[:, i, :, :], tensor(0.0, dtype=torch.float32, device=pi.device))
        pi_clone = pi.clone().detach()

        # f_val: (bs, oc, *choice_space), pi: (bs, oc, *choice_space)
        log_d_g = (f_val_clone_masked - log(torch.exp(f_val_clone_masked) + pi_clone)) * pi
        log_1_d_g = (log(pi) - log(torch.exp(f_val_clone_masked) + pi_clone)) * pi

        log_d_d = (f_val_masked - log(torch.exp(f_val_masked) + pi_clone)) * next_link
        log_1_d_d = (log(pi_clone) - log(torch.exp(f_val_masked) + pi_clone)) * pi_clone

        return (log_d_g, log_1_d_g), (log_d_d, log_1_d_d)

    #def get_shap_values(self, inputs_train, pi_train, inputs_test, pi_test, i):
        # input Grid: (sum(links), c, 3, 3) corresponds to next_links
        # input Emb: (trip_num, link_num, link_num, c) sparse corresponds to transition_matrix
        # output, pi: (bs, *choice_space)
    #    def f(inputs_pi):
            # inputs_pi: (bs, *choice_space)
    #        f_val = self.discriminator(inputs, pi, i=i)
    #        return f_val

    def get_criteria(self, batch, pi, i, w=None):
        # ll, TP, FP, FN, TN
        if self.use_index:
            input, mask, next_mask, _, _ = batch
            mask = mask.view(mask.shape[0], input.shape[-2], input.shape[-1])  # (*, 3, 3)
            next_mask = next_mask.view(next_mask.shape[0], input.shape[-2], input.shape[-1])  # (*, 3, 3)
        else:
            input, mask, next_mask, _ = batch
        f_val, util, ext, val = self.discriminator.get_vals(input, pi, i=i, w=w)  # (bs, *choice_space)
        q = util + ext + self.discriminator.gamma * val
        q = torch.where(mask > 0, q, tensor(-9e15, dtype=torch.float32, device=q.device))
        # choose maximum q
        if self.use_index:  # choose from choice_space
            q = q.view(q.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)  # (*, 9)
            next_mask = next_mask.view(next_mask.shape[0], -1)  # (*, 9)
        pi_q = F.softmax(q, dim=-1)
        if self.sln:
            logits = self.generator(input, w=w)  # raw_data_fake requires_grad=True
        else:
            logits = self.generator(input)  # raw_data_fake requires_grad=True
        mask = mask.unsqueeze(1)
        if self.use_index:
            mask = mask.view(-1, 1, logits.shape[-2], logits.shape[-1])
        logits = torch.where(mask > 0, logits, tensor(-9e15, dtype=torch.float32,
                                                      device=logits.device))
        pi_g = self.get_pi_from_logits(logits)[:, i, :, :].view(*next_mask.shape)  # (bs, 9)
        ll = log((pi_g * next_mask.view(pi_g.shape)).sum(dim=-1)).sum()

        pred_mask = (pi_g == pi_g.max(dim=-1, keepdim=True)[0]).to(torch.float32)
        mask = mask.view(*next_mask.shape)
        tp = (next_mask * pred_mask).sum()
        fp = (mask * (1-next_mask) * pred_mask).sum()
        tn = (mask * (1-next_mask) * (1-pred_mask)).sum()
        fn = (next_mask * (1-pred_mask)).sum()
        return ll, tp, fp, tn, fn

    def cat_image_feature(self, batch, w=None):
        # w: (bs, w_dim)
        # index: (bs, 9)
        if self.use_compressed_image:  # 2 step compression
            image_feature = self.encoder(self.comp_feature, w=w)  # (bs, link_num, mid_dim) -> (bs, link_num, emb_dim)
        else:  # single step compression
            idx_set = None
            if self.use_index:
                idx_set = set(np.unique(batch[3].clone().detach().cpu().numpy()))
            self._load_and_compress_image(idx_set)  # self.comp_feature: (link_num, emb_dim)
            image_feature = self.encoder(self.comp_feature, w=w)  # (bs, link_num, emb_dim)

        # inputs: (bs, c, 3, 3) or (bs, link_num, feature_num). c or feature_num are total_feature_num + emb_dim
        if self.use_index:
            # image_feature: tensor(link_num, emb_dim)
            # batch[3]: tensor(bs, 9)
            if image_feature.dim() == 3:
                image_feature_tmp = image_feature[0, batch[3].long(), :] * (batch[3].unsqueeze(-1) >= 0)
            else:
                image_feature_tmp = image_feature[batch[3].long(), :] * (batch[3].unsqueeze(-1) >= 0)
            image_feature_tmp = image_feature_tmp.transpose(1, 2).view(batch[3].shape[0], -1, 3, 3)
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

    def save(self):
        self.generator.save(self.model_dir)
        self.discriminator.save(self.model_dir)
        if self.use_encoder:
            self.encoder.save(self.model_dir)

    def load(self):
        self.generator.load(self.model_dir)
        self.discriminator.load(self.model_dir)
        if self.use_encoder:
            self.encoder.load(self.model_dir)

    def _load_image_feature(self):
        # load compressed image feature and store in self.comp_feature
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
            self.comp_feature = tensor(comp_feature, dtype=torch.float32, device=self.device, requires_grad=False)  # (link_num, comp_dim)  comp_dim: mid_dim

    def _load_and_compress_image(self, idx_set=None):
        # only when use_compressed_image is False
        # image_data: CompressedImageData
        comp_feature = None
        comp_dim = None
        for i in range(len(self.image_data.lids)):
            images = self.image_data.load_link_image(i)  # list(tensor(n, c, h, w))
            tmp_feature = None
            for num_source, img in enumerate(images):
                if idx_set is not None and i not in idx_set:
                    break
                img = img.to(self.device)
                compressed = self.encoder.compress(img, num_source=num_source)
                if type(compressed) == tuple:
                    compressed = compressed[0]
                compressed = compressed.view(1, -1)
                if tmp_feature is None:
                    tmp_feature = compressed
                else:
                    tmp_feature = tmp_feature + compressed  # (emb_dim)
            if tmp_feature is not None:
                if comp_feature is None:  # initialize comp_feature
                    comp_dim = tmp_feature.shape[-1]
                    comp_feature = torch.zeros((len(self.image_data.lids), comp_dim), dtype=torch.float32, device=self.device)
                comp_feature[i, :] = tmp_feature
        if comp_feature is None:
            print("No image feature is loaded.")
        else:
            self.comp_feature = comp_feature  # (link_num, comp_dim)  comp_dim: mid_dim


class _Encoder(nn.Module):
    # for shap calculation
    def __init__(self, model, num_source=0):
        super().__init__()
        self.model = model
        self.num_source = num_source

    def forward(self, x):
        compressed = self.model.compress(x, self.num_source)
        if type(compressed) is tuple:
            compressed = compressed[0]
        #return self.model(compressed).sum(dim=-1).view(-1, 1)  # tensor((2d+1)^2, emb_dim)
        res = self.model(compressed)
        res = res.sum(dim=-1, keepdim=True)
        return res  # tensor(bs, 1)
    
class _Encoder2(nn.Module):
    def __init__(self, model, num_source=0):
        super().__init__()
        self.model = model
        self.num_source = num_source
    def forward(self, x):
        return self.model.compress_forward(x, self.num_source)
