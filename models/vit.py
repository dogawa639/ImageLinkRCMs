# reference  https://github.com/lucidrains/vit-pytorch.git
import numpy as np

import torch
from torch import tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from models.general import FF
from models.transformer import TransformerEncoder

__all__ = ["ViT"]


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, dropout=0.0, pool='cls', output_atten=False):
        super().__init__()
        channels, img_height, img_width = img_size
        patch_height, patch_width = patch_size

        if img_height % patch_height != 0 or img_width % patch_height != 0:
            raise ValueError("Image size must be divided by patch size.")

        self.pool = pool
        self.output_atten = output_atten

        patch_dim = channels * patch_height * patch_width
        self.split_height = img_height // patch_height
        self.split_width = img_width // patch_width
        num_patches = self.split_height * self.split_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            #nn.Linear(patch_dim, dim),
            FF(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(dim, num_head=heads, dropout=dropout, depth=depth, pre_norm=True, output_atten=True)
        self.mlp_head = FF(dim, num_classes)
        self.transformer_decoder = TransformerEncoder(dim, num_head=1, dropout=dropout, depth=1, pre_norm=True, output_atten=False)
        self.to_patch = nn.Sequential(
            nn.LayerNorm(dim),
            FF(dim, patch_dim)
        )

    def forward(self, img):
        # img: (bs, c, h, w)->(h, w, bs, c)
        img = torch.permute(img, (2, 3, 0, 1))
        # (h, w, bs, c) -> (bs, num_patches, patch_dim)
        patches = torch.stack([torch.stack(torch.hsplit(h_img, self.split_width), 0) for h_img in torch.vsplit(img, self.split_height)], 0)  # shape(split_height, split_width, h', w', bs, c)
        patches = torch.permute(patches, (4, 0, 1, 5, 2, 3))  # shape(split_height, split_width, h', w', bs, c) -> (bs, split_height, split_width, c, h', w')
        patches = patches.reshape(patches.shape[0], self.split_height * self.split_width, -1)

        bs, n, _ = patches.shape
        cls_token = self.cls_token.repeat(bs, 1, 1)  # (1, 1, dim)->(bs, n, dim)

        x = self.to_patch_embedding(patches)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x, atten = self.transformer(x, None)  # x: (bs, 1+num_patches, dim), atten: [(bs, 1+num_patches, 1+num_patches)]
        atten_all = atten[0]  # (bs, 1+num_patches, 1+num_patches)
        for i in range(1, len(atten)):
            atten_all = torch.matmul(atten[i], atten_all) + atten_all
        if self.pool == "cls":
            x = x[:, 0]
            atten = atten_all[:, 0, 1:]
        else:
            x = x.mean(dim=1)
            atten = torch.mean(atten_all[:, 1:, 1:], dim=1)
        if self.output_atten:
            return self.mlp_head(x), atten.view(atten.shape[0], self.split_height, self.split_width)  # (bs, num_classes), (bs, self.split_height, self.split_width)
        return self.mlp_head(x)  # (bs, num_classes)

    def auto_encoder(self, img):
        # img: (bs, c, h, w)->(h, w, bs, c)
        img = torch.permute(img, (2, 3, 0, 1))
        # (h, w, bs, c) -> (bs, num_patches, patch_dim)
        patches = torch.stack([torch.stack(torch.hsplit(h_img, self.split_width), 0) for h_img in torch.vsplit(img, self.split_height)], 0)  # shape(split_height, split_width, h', w', bs, c)
        patch_c, patch_h, patch_w = patches.shape[5], patches.shape[2], patches.shape[3]

        patches = torch.permute(patches, (4, 0, 1, 5, 2, 3))  # shape(split_height, split_width, h', w', bs, c) -> (bs, split_height, split_width, c, h', w')
        patches = patches.reshape(patches.shape[0], self.split_height * self.split_width, -1)

        bs, n, _ = patches.shape
        cls_token = self.cls_token.repeat(bs, 1, 1)  # (1, 1, dim)->(bs, n, dim)

        x = self.to_patch_embedding(patches)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x, atten = self.transformer(x, None)  # x: (bs, 1+num_patches, dim), atten: [(bs, 1+num_patches, 1+num_patches)]
        x = self.transformer_decoder(x, None)  # x: (bs, 1+num_patches, dim)
        x = self.to_patch(x)  # x: (bs, 1+num_patches, patch_dim)
        x = x[:, 1:, :].reshape(x.shape[0], self.split_height, self.split_width, patch_c, patch_h, patch_w)
        x = torch.cat([torch.cat([x[:, i, j, :, :, :] for j in range(self.split_width)], 3) for i in range(self.split_height)], 2)  # x: (bs, c, h, w)
        return x


