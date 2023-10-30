import torch
from torch import tensor, nn

__all__ = ["Softplus", "log", "FF", "SLN"]


def log(x):
    return torch.log(torch.clip(x, min=1e-6))


class Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}

    def forward(self, x):
        return torch.max(tensor(0.0, device=x.device), x) + log(1.0 + torch.exp(-torch.abs(x)))


class FF(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_channel, bias=True):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.hidden_channel = hidden_channel

        self.ff1 = nn.Linear(input_channel, hidden_channel, bias=bias)
        self.ff2 = nn.Linear(hidden_channel, output_channel, bias=bias)

        self.seq = nn.Sequential(
            self.ff1,
            nn.ReLU(),
            self.ff2
        )

    def forward(self, x):
        return self.seq(x)


class SLN(nn.Module):
    def __init__(self, input_size, parameter_size):
        # input_size: w, parameter_size: hidden
        super().__init__()
        self.input_size = input_size
        self.parameter_size = parameter_size
        self.ln = nn.LayerNorm(parameter_size)
        self.gamma = FF(input_size, parameter_size, input_size, bias=False)
        self.beta = FF(input_size, parameter_size, input_size, bias=False)

    def forward(self, hidden, w):
        # hidden: (*, parameter_size)
        # w: (*, input_size)
        gamma = self.gamma(w)
        beta = self.beta(w)
        ln = self.ln(hidden)
        return gamma * ln + beta

