import torch
from torch import tensor, nn, optim
import torch.nn.functional as F
from cnn import CNN2L
from transformer import Transformer
from general import FF, SLN

import numpy as np

# forward input(patch(c, h, w), w) -> (bs, emb_dim)

class CNNEnc(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = CNN2L(1, 1, sln=True, w_dim=10)