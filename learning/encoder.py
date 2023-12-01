import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from models.deeplabv3 import resnet50, ResNet50_Weights
from models.transformer import TransformerEncoder
from models.gnn import GAT
from models.general import FF, SLN

import numpy as np

# forward input(patch(bs, c, h, width), w) -> (bs, emb_dim)
# w: None -> bs=1
# w: tensor(bs, w_dim) -> bs=bs

class CNNEnc(nn.Module):
    def __init__(self, patch_size, emb_dim, num_source=1, sln=True, w_dim=10):
        super().__init__()
        if type(patch_size) is int:
            patch_size = (3, patch_size, patch_size)
        self.patch_size = patch_size  #(c, h, w)
        self.emb_dim = emb_dim
        self.sln = sln
        self.w_dim = w_dim
        self.mid_dim = 1000

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, True, True])
        for param in self.resnet50.parameters():
            param.requires_grad = False

        self.lin = nn.ModuleList([FF(self.mid_dim, emb_dim, self.mid_dim*2, bias=True) for _ in range(num_source)])
        self.norm0 = nn.LayerNorm(patch_size)
        if sln:
            self.norm1 = SLN(w_dim, self.mid_dim)
            self.norm2 = SLN(w_dim, emb_dim)  # layer norm at the last output
        else:
            self.norm1 = nn.LayerNorm(self.mid_dim)
            self.norm2 = nn.LayerNorm(emb_dim)

        self.seq = nn.Sequential(
            self.norm0,
            self.resnet50  # output: (1, 1000)
        )

    def forward(self, x, source_i=0, w=None):
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

    def compress(self, patch):
        # patch: (bs2, c, h, width) or (c, h, width)
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)  # (1, c, h, w)
        x = self.seq(patch)   # (bs2, 1000)
        return x

# test
if __name__ == "__main__":
    patch_size = (3, 256, 256)
    bs = 3
    emb_dim = 5
    num_source = 1
    w_dim = 6
    link_num = 10
    device = "mps"
    inputs = torch.randn(link_num, *patch_size).to(device)
    w = torch.randn(bs, w_dim).to(device)

    cnn = CNNEnc(patch_size, emb_dim=emb_dim, num_source=num_source, w_dim=w_dim).to(device)

    out = cnn(inputs, source_i=0, w=w)
    print(out.shape)






