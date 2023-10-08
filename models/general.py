import torch
from torch import tensor, nn



def log(x):
    return torch.log(torch.clip(x, min=1e-6))

class Softplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}

    def forward(self, x):
        return torch.max(tensor(0.0, device=x.device), x) + log(1.0 + torch.exp(-torch.abs(x)))
