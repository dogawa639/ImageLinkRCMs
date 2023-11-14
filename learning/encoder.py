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

        self.patch_size = patch_size  #(c, h, w)
        self.emb_dim = emb_dim
        self.sln = sln
        self.w_dim = w_dim
        self.mid_dim = 1000

        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT, replace_stride_with_dilation=[False, True, True])
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
            self.resnet50  # output: (bs, 1000)
        )

    def forward(self, patch, source_i=0, w=None):
        # patch: (c, h, width) or (bs, c, h, width)
        # w: (bs, w_dim) or (w_dim)
        # output: (bs, emb_dim)
        if w is not None and w.dim() == 2:
            bs = w.shape[0]
            if patch.dim() == 3:
                patch = patch.expand(bs, *patch.shape)
        elif patch.dim() == 3:
            patch = patch.expand(1, *patch.shape)

        if self.sln and w is None:
            raise Exception("w should be specified when sln is True")
        x = self.seq(patch)  # (bs, 1000)

        if self.sln:
            self.norm1.set_w(w)
            self.norm2.set_w(w)
        x = self.norm1(x)
        x = self.lin[source_i](x)
        x = self.norm2(x)
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
    inputs = torch.randn(*patch_size).to(device)
    w = torch.randn(bs, w_dim).to(device)

    cnn = CNNEnc(patch_size, emb_dim=emb_dim, num_source=num_source, w_dim=w_dim).to(device)

    out = cnn(inputs, source_i=0, w=w)
    print(out.shape)






