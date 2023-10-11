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

import os

from util import log


class GAN:
    def __init__(self, generator, discriminator, datasets, model_dir, airl=True, hinge_loss=False, device="cpu"):
        self.generator = generator
        self.discriminator = discriminator
        self.datasets = datasets
        self.model_dir = model_dir
        self.airl = airl
        self.hinge_loss = hinge_loss
        self.device = device
    
    def train(self, epochs, batch_size, lr_g, lr_d, shuffle):
        optimizer_g = optim.Adam(self.generator.parameters(), lr=lr_g)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        scheduler_g = ConstantLR(optimizer_g)
        scheduler_d = ConstantLR(optimizer_d)

        dataloaders_real = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in self.datasets]

        self.generator.train()
        self.discriminator.train()

        raw_data_fake, datasets_fake  = self.generator.generate()  # raw_data: retain_grad=True
        dataloaders_fake = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in datasets_fake]

        d_real = self.discriminator.get_d(dataloaders_real)
        d_fake = self.discriminator.get_d(dataloaders_fake)

    def loss(self, raw_data_fake, d_real, d_fake):
        # raw_data_fake, d_fake: same shape, tensor
        # raw_data_fake: in [0, 1]
        if self.airl:
            if self.hinge_loss:
                loss_g = -log(raw_data_fake * d_fake) + log(1. - raw_data_fake * d_fake)
                loss_d = -log(d_real) - log(1. - d_fake)
            else:
                loss_g = -log(raw_data_fake * d_fake) + log(1. - raw_data_fake * d_fake)
                loss_d = -log(d_real) - log(1. - d_fake)

        else:
            if self.hinge_loss:
                loss_g = -log(raw_data_fake * d_fake)
                loss_d = -log(d_real) - log(1. - d_fake)
            else:
                loss_g = -log(raw_data_fake * d_fake)
                loss_d = -log(d_real) - log(1. - d_fake)
        





