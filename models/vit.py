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
    def __init__(self, img_size, patch_size, num_classes, dim, depth, heads, dropout=0.0, pool='cls'):
        super().__init__()
        img_height, img_width, channels = img_size
        patch_height, patch_width = patch_size

        if img_height % patch_height != 0 or img_width % patch_height != 0:
            raise ValueError("Image size must be divided by patch size.")
        self.pool = pool

        patch_dim = channels * patch_height * patch_width
        self.split_height = img_height // patch_height
        self.split_width = img_width // patch_width
        num_patches = self.split_height * self.split_width

        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransformerEncoder(dim, num_head=heads, dropout=dropout, depth=depth, pre_norm=True)
        self.mlp_head = FF(dim, num_classes)

    def forward(self, img):
        # img: (bs, c, h, w)->(h, w, bs, c)
        img = torch.permute(img, (0, 2, 3, 1))
        # (h, w, bs, c) -> (bs, num_patches, patch_dim)
        patches = []
        patches = tensor([patches.extend(np.hsplit(h_img, self.split_width) for h_img in np.vsplit(img, self.split_height))])  # shape(split_height, split_width, h', w', bs, c)
        patches = torch.permute(patches, (4, 0, 1, 5, 2, 3))  # shape(bs, split_height, split_width, c, h', w')
        patches = patches.view(patches.shape[0], self.split_height * self.split_width, -1)

        bs, n, _ = patches.shape
        cls_token = self.cls_token.repeat(bs, n, 1)  # (1, 1, dim)->(bs, n, dim)

        x = torch.cat((cls_token, patches), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0] if self.pool == "cls" else x.mean(dim=1)
        return self.mlp_head(x)  # (bs, num_patches, num_classes)


