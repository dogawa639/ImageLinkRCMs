import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.deeplabv3 import resnet50, ResNet50_Weights
from models.cnn import ResNet
from models.transformer import TransformerEncoder
from models.vit import ViT
from models.gnn import GAT
from models.general import FF, SLN

import numpy as np

# forward input(satellite(bs, c, h, width), w) -> (bs, emb_dim)
# w: None -> bs=1
# w: tensor(bs, w_dim) -> bs=bs


class CNNEnc(nn.Module):
    def __init__(self, patch_size, emb_dim, mid_dim=1000, num_source=1, sln=True, w_dim=10):
        super().__init__()
        if type(patch_size) is int:
            patch_size = (3, patch_size, patch_size)
        self.patch_size = patch_size  #(c, h, w)
        self.emb_dim = emb_dim
        self.sln = sln
        self.w_dim = w_dim
        self.mid_dim = mid_dim

        self.resnet50 = nn.ModuleList([ResNet(3, depth=3, num_classes=mid_dim) for _ in range(num_source)])

        self.lin = nn.ModuleList([FF(self.mid_dim, emb_dim, self.mid_dim*2, bias=True) for _ in range(num_source)])
        self.norm0 = nn.LayerNorm(patch_size)
        if sln:
            self.norm1 = SLN(w_dim, self.mid_dim)
            self.norm2 = SLN(w_dim, emb_dim)  # layer norm at the last output
        else:
            self.norm1 = nn.LayerNorm(self.mid_dim)
            self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, w=None, source_i=0):
        # x: (bs2, mid_dim) or (mid_dim)
        # w: (bs1, w_dim) or (w_dim)
        # output: (bs1, bs2, emb_dim) or (bs2, emb_dim) or (emb_dim)

        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        use_bs = True
        if x.dim() == 3:
            use_bs = False
            x = x.unsqueeze(0)

        if w is not None and w.dim() == 2:
            bs1 = w.shape[0]
            x = x.expand(bs1, *x.shape)  # (bs1, bs2, 1000)

        if self.sln:
            self.norm1.set_w(w)
            self.norm2.set_w(w)
        x = self.norm1(x)
        x = self.lin[source_i](x)
        x = self.norm2(x)
        if not use_bs:
            x = x.squeeze(0)
        return x

    def compress(self, patch, num_source=0):
        # satellite: (bs2, c, h, width) or (c, h, width)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)  # (1, c, h, w)
        patch = self.norm0(patch)
        x = self.resnet50[num_source](patch)   # (bs2, 1000)
        return x

    def save(self, model_dir, i=None):
        if i is None:
            torch.save(self.state_dict(), model_dir + "/cnnenc.pth")
        else:
            torch.save(self.state_dict(), model_dir + f"/cnnenc_{i}.pth")

    def load(self, model_dir, i=None):
        if i is None:
            self.load_state_dict(torch.load(model_dir + "/cnnenc.pth"))
        else:
            self.load_state_dict(torch.load(model_dir + f"/cnnenc_{i}.pth"))


class ViTEnc(nn.Module):
    def __init__(self, patch_size, vit_patch_size, emb_dim, mid_dim=1000, num_source=1, sln=True, w_dim=10, depth=6, heads=1, dropout=0.0, output_atten=False):
        super().__init__()
        if type(patch_size) is int:
            patch_size = (3, patch_size, patch_size)
        self.patch_size = patch_size  # (c, h, w)
        self.emb_dim = emb_dim
        self.sln = sln
        self.w_dim = w_dim
        self.mid_dim = mid_dim
        self.output_atten = output_atten

        patch_dim = vit_patch_size[0] * vit_patch_size[1] * patch_size[0]
        self.vit = nn.ModuleList([ViT(patch_size, vit_patch_size, mid_dim, patch_dim // 2, depth, heads, dropout, output_atten=True) for _ in range(num_source)])

        self.lin = nn.ModuleList([FF(self.mid_dim, emb_dim, self.mid_dim*2, bias=True) for _ in range(num_source)])
        self.norm0 = nn.LayerNorm(patch_size)
        if sln:
            self.norm1 = SLN(w_dim, self.mid_dim)
            self.norm2 = SLN(w_dim, emb_dim)  # layer norm at the last output
        else:
            self.norm1 = nn.LayerNorm(self.mid_dim)
            self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, w=None, source_i=0):
        # x: (bs2, mid_dim) or (mid_dim)
        # w: (bs1, w_dim) or (w_dim)
        # output: (bs1, bs2, emb_dim) or (bs2, emb_dim) or (emb_dim)

        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        use_bs = True
        if x.dim() == 3:
            use_bs = False
            x = x.unsqueeze(0)

        if w is not None and w.dim() == 2:
            bs1 = w.shape[0]
            x = x.expand(bs1, *x.shape)  # (bs1, bs2, mid_dim)

        if self.sln:
            self.norm1.set_w(w)
            self.norm2.set_w(w)
        x = self.norm1(x)
        x = self.lin[source_i](x)
        x = self.norm2(x)
        if not use_bs:
            x = x.squeeze(0)
        return x

    def compress(self, patch, num_source=0):
        # satellite: (bs2, c, h, width) or (c, h, width)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)  # (1, c, h, w)
        x, atten = self.vit[num_source](self.norm0(patch))  # (bs2, mid_dim), (bs2, num_patches)
        if self.output_atten:
            return x, atten
        return x

    def save(self, model_dir, i=None):
        if i is None:
            torch.save(self.state_dict(), model_dir + "/vitenc.pth")
        else:
            torch.save(self.state_dict(), model_dir + f"/vitenc_{i}.pth")

    def load(self, model_dir, i=None):
        if i is None:
            self.load_state_dict(torch.load(model_dir + "/vitenc.pth"))
        else:
            self.load_state_dict(torch.load(model_dir + f"/vitenc_{i}.pth"))




