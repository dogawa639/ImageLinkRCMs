import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.sparse import csr_matrix, coo_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime

import os

from utility import MapSegmentation
from preprocessing.pp_processing import PP

__all__ = ["GridDataset", "PPEmbedDataset", "ImageDataset", "calc_loss"]


class GridDataset(Dataset):
    # 3*3 grid choice set
    def __init__(self, pp_data, nw_data, h_dim=10, normalize=True):
        # nw_data: NWDataCNN
        self.pp_data = pp_data
        self.nw_data = nw_data
        self.h_dim = h_dim

        lid2idx = {lid: i for i, lid in enumerate(nw_data.lids)}

        f = nw_data.feature_num
        c = nw_data.context_feature_num
        # inputs: [sum(sample_num), f+c, 3, 3]
        # masks: [sum(sample_num), 9]
        # next_links: [sum(sample_num), 9]
        inputs = np.zeros((0, f + c, 3, 3), dtype=np.float32)
        masks = np.zeros((0, 9), dtype=np.float32)
        next_links = np.zeros((0, 9), dtype=np.float32)
        link_idxs = np.zeros((0, 9), dtype=np.int32)
        hs = np.zeros((0, h_dim), dtype=np.float32)
        for tid in pp_data.tids:
            path = pp_data.path_dict[tid]["path"]
            d_node_id = pp_data.path_dict[tid]["d_node"]
            if len(path) == 0:
                continue
            sample_num = len(path) - 1
            trip_input = np.zeros((sample_num, f + c, 3, 3), dtype=np.float32)  # [link, state, state]
            trip_mask = np.zeros((sample_num,9), dtype=np.float32)  # [link, mask]
            trip_next_link = np.zeros((sample_num,9), dtype=np.float32)  # [link, one_hot]
            trip_link_idxs = np.zeros((sample_num,9), dtype=np.int32)  # [link, next_link_idxs]

            for i in range(sample_num):
                tmp_link = path[i]
                next_link = path[i+1]
                feature_mat, _ = nw_data.get_feature_matrix(tmp_link, normalize=normalize)
                context_mat, action_edge = nw_data.get_context_matrix(tmp_link, d_node_id, normalize=normalize)

                trip_input[i, 0:f, :, :] = feature_mat
                trip_input[i, f:f+c, :, :] = context_mat
                trip_mask[i, :] = [len(edges) > 0 for edges in action_edge]
                trip_next_link[i, :] = [next_link in edges for edges in action_edge]
                trip_link_idxs[i, :] = [lid2idx[np.random.choice(edges)] if len(edges) > 0 else 0 for edges in action_edge]

            trip_input = trip_input[np.isnan(trip_input).sum(axis=(1, 2, 3)) == 0]
            if len(trip_input) == 0:
                continue
            inputs = np.concatenate((inputs, trip_input), axis=0)
            masks = np.concatenate((masks, trip_mask), axis=0)
            next_links = np.concatenate((next_links, trip_next_link), axis=0)
            link_idxs = np.concatenate((link_idxs, trip_link_idxs), axis=0)
            hs = np.concatenate((hs, np.repeat(np.random.randn(1, h_dim), len(trip_input), axis=0)), axis=0)

        self.inputs = tensor(inputs, retain_grad=False)
        self.masks = tensor(masks, retain_grad=False)
        self.next_links = tensor(next_links, retain_grad=False)
        self.link_idxs = tensor(link_idxs, retain_grad=False)
        self.hs = tensor(hs, retain_grad=False)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # [f+c, 3, 3], [9], [9], [9]
        return self.inputs[idx], self.masks[idx], self.next_links[idx], self.link_idxs[idx], self.hs[idx]
    
    def split_into(self, ratio):
        # ratio: [train, test, validation]
        pp_data = self.pp_data.split_into(ratio)
        return [GridDataset(pp_data[i], self.nw_data) for i in range(len(ratio))]

    def get_fake_batch(self, real_batch, g_output):
        mask = real_batch[1]
        logits = torch.where(mask > 0, g_output, tensor(-9e15, dtype=torch.float32, device=g_output.device))
        next_link_prob = F.softmax(logits, dim=-1)
        next_links = torch.multinomial(next_link_prob.view(-1, g_output.shape[-1]), num_samples=1).squeeze()
        next_links_one_hot = F.one_hot(next_links, num_classes=g_output.shape[-1]).view(g_output.shape)
        return real_batch[0], real_batch[1], next_links_one_hot


class PPEmbedDataset(Dataset):
    # nw_data: NWDataGNN
    # Global state (link_num, graph_node_feature_num) + PP pos embedding, PP adj matrix (link_num, link_num)
    def __init__(self, pp_data, nw_data, h_dim=10):
        # global state: output of GNN
        self.pp_data = pp_data
        self.nw_data = nw_data
        self.feature_mat = nw_data.get_feature_matrix()
        self.link_num, self.feature_num = self.feature_mat.shape

        self.pp_pos_embeddings = [self.get_pp_pos_embedding(path) for path in pp_data.load_edge_list()]
        self.pp_adj_matrices = [self.get_pp_adj_matrix(path) for path in pp_data.load_edge_list()]

        self.hs = torch.randn((len(self.pp_data), h_dim), dtype=torch.float32, requires_grad=False)

    def __len__(self):
        return len(self.pp_data)
    
    def __getitem__(self, idx):
        kargs = {"dtype": torch.float32, "retain_grad": False}
        # [link_num, feature_num], [link_num, link_num]
        return (tensor(self.feature_mat, **kargs),
                tensor(self.nw_data.a_matrix.toarray(), **kargs).to_sparse(),  #隣接行列
                tensor(self.pp_adj_matrices[idx].toarray(), **kargs).to_sparse(),
                tensor(self.hs[idx], **kargs))

    def get_pp_pos_embedding(self, path):
        # path: [link_id] or [edge obj]
        pos_embedding = np.zeros((self.link_num, self.feature_num), dtype=np.float32)
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for i, edge in enumerate(path):
            if type(edge) != int:
                edge = edge.id
            if i % 2 == 0:
                pos_embedding[lid2idx[edge], :] = np.sin(np.arange(self.feature_num) / 10000 ** (i / self.feature_num))
            else:
                pos_embedding[lid2idx[edge], :] = np.cos(np.arange(self.feature_num) / 10000 ** ((i - 1) / self.feature_num))
        return csr_matrix(pos_embedding)
    
    def get_pp_adj_matrix(self, path):
        # path: [link_id] or [edge obj]
        row = []
        col = []
        lid2idx = {lid: i for i, lid in enumerate(self.nw_data.lids)}
        for i, edge in enumerate(path):
            if type(edge) != int:
                edge = edge.id
            if i < len(path) - 1:
                row.append(lid2idx[edge])
                col.append(lid2idx[path[i+1]])
        adj_matrix = coo_matrix((np.ones(len(row)), (row, col)), shape=(self.link_num, self.link_num), dtype=np.float32).to_csr()
        return adj_matrix
    
    def get_dataset_from_logits(self, logits):
        # logits: [trip_num, link_num, link_num] tensor(cpu, detached) each element is real
        # return: PPEmbedDataset
        exp_logits = torch.exp(logits)
        origin = (exp_logits.sum(axis=1, keepdim=False) > 0) & (exp_logits.sum(axis=2, keepdim=False) == 0)  # どこからも入ってきていないが，出ていっているリンク (trip_num, link_num)
        origin = origin[(origin.sum(axis=1) > 0), :]  # originがないtripは除く
        trip_num = origin.shape[0]

        logits_origin = torch.einsum('ijk, ij -> ij', logits, origin)  # (trip_num, link_num)
        origin_prob = logits_origin.softmax(dim=-1)  # (trip_num, link_num)

        paths = [[] for i in range(trip_num)]
        path_links = [{} for i in range(trip_num)]
        prev_idxs = [None for i in range(trip_num)]

        for i in range(trip_num):
            origin_idx = torch.multinomial(origin_prob[i, :], num_samples=1)  # (1,)
            paths[i].append(self.nw_data.lids[origin_idx.item()])
            path_links[i].add(origin_idx.item())
            prev_idxs[i] = origin_idx.item()

        goon = False
        for _ in range(logits.shape[1]):
            for i in range(trip_num):
                prev_idx = prev_idxs[i]
                if exp_logits[i, prev_idx, :].sum() == 0:  # 次に移動するリンクがない場合
                    continue
                
                next_idx = torch.multinomial(exp_logits[i, prev_idx, :], num_samples=1)  # (1,)
                if next_idx in path_links[i]:  # すでに通ったリンクの場合
                    continue
                paths[i].append(self.nw_data.lids[next_idx.item()])
                path_links[i].add(next_idx.item())
                prev_idxs[i] = next_idx.item()
                goon = True
            if not goon:
                break
        pp_data = [[i+1, paths[i][j], paths[i][j+1], paths[i][-1]] for i in range(trip_num) for j in range(len(paths[i]) - 1)]  # ID, a, k, b
        pp_data = PP(pd.DataFrame(pp_data, columns=["ID", "a", "k", "b"]), self.nw_data)

        return PPEmbedDataset(pp_data, self.grobal_state, self.nw_data)

    def split_into(self, ratio):
        # ratio: [train, test, validation]
        pp_data = self.pp_data.split_into(ratio)
        return [PPEmbedDataset(pp_data[i], self.grobal_state, self.nw_data) for i in range(len(ratio))]

    def get_fake_batch(self, real_batch, g_output):
        mask = real_batch[1]
        g_output = torch.where(mask > 0, g_output, torch.full_like(g_output, -9e15))  # (trip_num, link_num, link_num)
        next_link_prob = F.softmax(g_output, dim=-1)
        next_links = torch.multinomial(next_link_prob.view(-1, g_output.shape[-1]), num_samples=1).squeeze()  # (trip_num * link_num)
        next_links_one_hot = F.one_hot(next_links, num_classes=g_output.shape[-1]).view(g_output.shape)
        return real_batch[0], real_batch[1], next_links_one_hot, real_batch[3]


class ImageDataset(Dataset):
    # for unsupervised learning
    def __init__(self, files, additional_files=None, output_files=None, input_shape=(520, 520), num_sample=200):
        # files:[png file]
        # output_files:[npy file] (optional)
        super(ImageDataset, self).__init__()
        self.files = files
        self.additional_files = additional_files
        self.output_files = output_files
        if not additional_files is None:
            if len(files) != len(additional_files):
                print("files and additional_files must have same length.")
                raise(ValueError)
        if not output_files is None:
            if len(files) != len(output_files):
                print("files and output_files must have same length.")
                raise(ValueError)
        self.input_shape = input_shape
        self.num_sample = num_sample

        self.ra_params = [
            (-90, 90),  # degree
            (0.1, 0.1),  # translate
            (0.9, 1.1),  # scale
            (-1, 1, -1, 1),  # shear
            self.input_shape  # img_size
        ]

        self.rc = transforms.RandomCrop(input_shape)  # random crop
        self.ra = transforms.RandomAffine(*self.ra_params[0:-1])

        self._load_data()  # self.input_tensors
        self._split_data(num_sample=num_sample)  # self.cropped_tensor

    def __del__(self):
        if not self.output_files is None:
            for pt_file in self.output_tensors:
                os.remove(pt_file)
            for pt_file in self.cropped_output:
                os.remove(pt_file)

    def __getitem__(self, i):
        tmp_tensor = self.cropped_tensor[i]
        additional_tensor = tensor([]) if len(self.cropped_additional) == 0 else self.cropped_additional[i]
        one_hot_tensor =tensor([]) if len(self.cropped_output) == 0 else torch.load(self.cropped_output[i])
        return self._random_transform(tmp_tensor, additional_tensor, one_hot_tensor)
    
    def __len__(self):
        return self.cropped_tensor.shape[0]

    def affine_by_origin_idx(self, img, idx_transformed):
        # img (N, C, H, W)
        # idx_transformed (N, 1, H, W)
        idx_transformed = idx_transformed.to("cpu")
        img_channels = img.shape[1]
        img_resolv = img.shape[2] * img.shape[3]

        idx_transformed = idx_transformed.repeat(1, img_channels, 1, 1)
        null_idx = idx_transformed < 0
        batch_num = (np.arange(img.numel()) / img_resolv).astype(int) * img_resolv  # when img shape (bs, C, H, W)
        batch_num = tensor(batch_num, dtype=int, device="cpu")
        batch_num = batch_num.view(img.shape)
        idx_transformed = idx_transformed + batch_num.reshape(img.shape)
        idx_transformed[null_idx] = 0

        new_img = tensor(np.zeros(img.shape), dtype=img.dtype, device=img.device)
        new_img.reshape(-1)[:] = img.reshape(-1)[idx_transformed.reshape(-1)]
        new_img.reshape(-1)[null_idx.reshape(-1)] = 0

        return new_img

    def resplit_data(self):
        self._split_data(num_sample=self.num_sample)

    def _load_data(self):
        input_tensors = []
        additional_tensors = []
        # input files
        for file in self.files:
            input_image = Image.open(file)
            input_image = input_image.convert("RGB")
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(input_image)  # (3, H, W)
            input_tensors.append(input_tensor)
        # additional input
        if self.additional_files is not None:
            for file in self.additional_files:
                input_image = Image.open(file)
                input_image = input_image.convert("RGB")
                preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                input_tensor = preprocess(input_image)  # (3, H, W)
                additional_tensors.append(input_tensor)
        self.input_tensors = input_tensors
        self.additional_tensors = additional_tensors

        # output file (supervised)
        output_tensors = []
        self.map_seg = None
        if self.output_files is not None:
            map_seg = MapSegmentation(self.output_files)
            self.map_seg = map_seg
            for file in self.output_files:
                output_tensor = torch.tensor(map_seg.convert_file(file), dtype=torch.uint8)
                pt_file = file.replace(".png", ".pt")
                torch.save(output_tensor, pt_file)
                output_tensors.append(pt_file)
        self.output_tensors = output_tensors

    def _split_data(self, num_sample=200):
        # split data into some pices with shape (3, H, W)
        input_shape = self.input_shape

        cropped_tensor = torch.tensor(np.zeros((num_sample * len(self.input_tensors), 3, input_shape[0], input_shape[1]), dtype=np.float32))
        cropped_additional = torch.tensor(np.zeros((num_sample * len(self.input_tensors), 3, input_shape[0], input_shape[1]), dtype=np.float32))
        class_num = 0
        if not self.output_files is None:
            class_num = self.map_seg.class_num
        cropped_output = []  # path of npy files
        for i in range(len(self.input_tensors)):
            input_tensor = self.input_tensors[i]
            if input_tensor.shape[1] < input_shape[0] or input_tensor.shape[2] < input_shape[1]:
                print("Too small input image.")
                raise(ValueError)
            for num in range(num_sample):
                rc_params = transforms.RandomCrop.get_params(input_tensor, input_shape)
                cropped_tensor[i * num_sample + num, :, :, :] = transforms.functional.crop(input_tensor, *rc_params)
                if not self.additional_files is None:
                    cropped_additional[i * num_sample + num, :, :, :] = transforms.functional.crop(self.additional_tensors[i], *rc_params)
                if not self.output_files is None:
                    pt_tile = self.output_tensors[i].replace(".pt", f"_{num}.pt")
                    cropped_output_tmp = transforms.functional.crop(torch.load(self.output_tensors[i]), *rc_params)
                    torch.save(cropped_output_tmp, pt_tile)
                    cropped_output.append(pt_tile)
        self.cropped_tensor = cropped_tensor
        self.cropped_additional = cropped_additional
        self.cropped_output = cropped_output

    def _get_affine_origin_idx(self, ra_params):
        # idx of tensor before transform for each pixel after transform
        idx = tensor(np.arange(self.input_shape[0] * self.input_shape[1], dtype=int).reshape((1, self.input_shape[0], self.input_shape[1])))
        idx_transformed = transforms.functional.affine(idx, *ra_params, interpolation=transforms.InterpolationMode.NEAREST, fill=-1)

        return idx_transformed

    def _get_mask(self, ra_params):
        # mask for img_tensor
        mask = tensor(np.ones((1, self.input_shape[0], self.input_shape[1]), dtype=np.float32))
        mask = transforms.functional.affine(mask, *ra_params)  # mask for both tensor

        return mask

    def _random_transform(self, img_tensor, additional_tensor, one_hot_tensor):
        # tensor (3, H, W)
        ra_params = self.ra.get_params(*self.ra_params)

        transformed = transforms.functional.affine(img_tensor, *ra_params)
        if len(additional_tensor) == 0:
            additional_transformed = tensor([])
        else:
            additional_transformed = transforms.functional.affine(additional_tensor, *ra_params)
        if len(one_hot_tensor) == 0:
            one_hot_transformed = tensor([])
        else:
            one_hot_tensor = transforms.functional.affine(one_hot_tensor, *ra_params)

        mask = self._get_mask(ra_params)
        idx_transformed = self._get_affine_origin_idx(ra_params) # idx for original tensor

        rand = torch.rand(2)
        if rand[0] < 0.5:
            # vflip
            transformed = transforms.functional.vflip(transformed)
            if len(additional_tensor) > 0:
                additional_transformed = transforms.functional.vflip(additional_transformed)
            if len(one_hot_tensor) > 0:
                one_hot_transformed = transforms.functional.vflip(one_hot_transformed)
            mask = transforms.functional.vflip(mask)
            idx_transformed = transforms.functional.vflip(idx_transformed)
        if rand[1] < 0.5:
            # hflip
            transformed = transforms.functional.hflip(transformed)
            if len(additional_tensor) > 0:
                additional_transformed = transforms.functional.hflip(additional_transformed)
            if len(one_hot_tensor) > 0:
                one_hot_transformed = transforms.functional.hflip(one_hot_transformed)
            mask = transforms.functional.hflip(mask)
            idx_transformed = transforms.functional.hflip(idx_transformed)

        return img_tensor, transformed, additional_tensor, additional_transformed, one_hot_tensor, one_hot_transformed, mask, idx_transformed
    
def calc_loss(model, dataset, dataloader, loss_fn, device, optimizer=None, scheduler=None, model_additional=None, optimizer_additional=None, scheduler_additionl=None, loss_fn_kwargs={}):
    if not optimizer is None:
        model.train()
    else:
        model.eval()

    tmp_loss = []
    tmp_loss_add = []
    for img_tensor, transformed, additional_tensor, additional_transformed, one_hot_tensor, one_hot_transformed, mask, idx_transformed in dataloader:
        img_tensor = img_tensor.to(device)
        transformed = transformed.to(device)
        additional_tensor = additional_tensor.to(device)
        additional_transformed =additional_transformed.to(device)
        one_hot_tensor = one_hot_tensor.to(device)
        one_hot_transformed = one_hot_transformed.to(device)
        mask = mask.to(device)

        if not optimizer is None:
            optimizer.zero_grad()
        if not optimizer_additional is None:
            optimizer_additional.zero_grad()

        y, y2 = _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset)

        loss = loss_fn(y, y2, **loss_fn_kwargs)
        loss_add = tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
        if additional_tensor.max() > 0:
            if not model_additional is None:
                y_add, y2_add = _get_y_y2(model_additional, img_tensor, transformed, mask, idx_transformed, dataset)
                loss_add = loss_fn(y_add, y2_add, **loss_fn_kwargs) # loss for additional file
                loss_add = loss_add + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
                if not optimizer_additional is None:
                    loss_add.backward(retain_graph=True)
                    optimizer_additional.step()
                loss = loss + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
            else:
                y_add, y2_add = _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset)
                loss = loss + loss_fn(y, y2, **loss_fn_kwargs) # loss for additional file
                loss = loss + loss_fn(y, y_add, **loss_fn_kwargs) # loss between file and additional
        if one_hot_tensor.max() > 0:
            loss = loss + loss_fn(y, one_hot_tensor, **loss_fn_kwargs)

        if not optimizer is None:
            loss.backward(retain_graph=True)
            optimizer.step()
        if not optimizer_additional is None:
            loss_add.backward(retain_graph=True)
            optimizer_additional.step()

        tmp_loss.append(loss.detach().cpu().item())
        tmp_loss_add.append(loss_add.detach().cpu().item())
    if not scheduler is None:
        scheduler.step()
    if not scheduler_additionl is None:
        scheduler_additionl.step()
    return tmp_loss, tmp_loss_add

def _get_y_y2(model, img_tensor, transformed, mask, idx_transformed, dataset):
    y = model(img_tensor)["out"]  # input (N, C, H, W)
    y = dataset.affine_by_origin_idx(y, idx_transformed)  # g
    y = y * mask
    y2 = model(transformed)["out"]  # filter (N, C, H, W)
    y2 = y2 * mask
    return y, y2


