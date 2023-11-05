import torch
from torch import tensor, nn
from torch.nn.utils import spectral_norm

import numpy as np

__all__ = ["Softplus", "softplus", "log", "FF", "SLN"]


def log(x):
    return torch.log(torch.clip(x, min=1e-6))


def softplus(x):
    return torch.max(tensor(0.0, device=x.device), x) + log(1.0 + torch.exp(-torch.abs(x)))


class Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}

    def forward(self, x):
        return torch.max(tensor(0.0, device=x.device), x) + log(1.0 + torch.exp(-torch.abs(x)))


class FF(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel, act_fn=lambda x : x, bias=True, sn=False):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channel = hidden_channel
        self.act_fn = act_fn

        if sn:
            self.ff1 = spectral_norm(nn.Linear(input_channel, hidden_channel, bias=bias))
            self.ff2 = spectral_norm(nn.Linear(hidden_channel, output_channel, bias=bias))
        else:
            self.ff1 = nn.Linear(input_channel, hidden_channel, bias=bias)
            self.ff2 = nn.Linear(hidden_channel, output_channel, bias=bias)

        self.seq = nn.Sequential(
            self.ff1,
            nn.ReLU(),
            self.ff2
        )

    def forward(self, x):
        return self.act_fn(self.seq(x))


class SLN(nn.Module):
    def __init__(self, w_dim, h_size):
        # w_dim: int
        # h_size: int or tuple
        super().__init__()
        self.w_dim = w_dim
        self.h_size = h_size
        if type(h_size) == int:
            self.h_flatten_size = h_size
            self.h_dim = 1
        else:
            self.h_flatten_size = np.prod(h_size)
            self.h_dim = len(h_size)
        self.ln = nn.LayerNorm(h_size)
        self.gamma = FF(w_dim, self.h_flatten_size, w_dim, bias=False)
        self.beta = FF(w_dim, self.h_flatten_size, w_dim, bias=False)

        self.w = None
        self.common_dim = None

    def forward(self, hidden):
        # hidden: (*, h_size)
        # self.w: tensor(bs, w_dim) or (w_dim)
        if self.w is None:
            raise Exception("w should be set before forward")
        add_dim = len(hidden.shape) - self.h_dim - (self.w.dim() - 1)
        if self.w.dim() == 1:  # w: tensor(w_dim) -> (*,w_dim)
            self.w = self.w.view(*[1]*add_dim, *self.w.shape).repeat(*hidden.shape[:-self.h_dim], 1)
        elif self.w.dim() == 2:  # w: tensor(bs, w_dim) -> (*,w_dim)
            self.w = self.w.view(self.w.shape[0], *[1]*add_dim, self.w.shape[1]).repeat(1, *hidden.shape[1:-self.h_dim], 1)
        else:
            raise Exception("w should be tensor(w_dim) or tensor(bs, w_dim)")

        gamma = self.gamma(self.w)  # (*, h_dim)
        beta = self.beta(self.w)  # (*, h_dim)
        ln = self.ln(hidden)  # (*, h_size)
        return gamma.view(ln.shape) * ln + beta.view(ln.shape)
    
    def set_w(self, w):
        self.w = w

