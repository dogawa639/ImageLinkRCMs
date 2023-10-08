from models.general import *

import torch
from torch import tensor, nn
import torch.nn.functional as F

__all__ = ["CNN2L", "CNN2LPositive", "CNN2LNegative"]

class CNN2L(nn.Module):
    def __init__(self, input_channel, output_channel):
        # forward: (B, C, 3, 3)->(B, C', 3, 3)
        super().__init__()

        self.input_channel = input_channel
        self.output_channel = output_channel

        # convolution
        self.cov1 = nn.Conv2d(self.input_channel, self.input_channel*8, 3, padding=1, bias=False)#(bs, in_channel, 3, 3)->(bs, in_channel*8, 3, 3)
        self.cov2 = nn.Conv2d(self.input_channel*8, self.input_channel*16, 3, padding=1, bias=False)#(bs, in_channel*8, 3, 3)->(bs, in_channel*32, 3, 3)
        # fully connected
        #reshape (bs, in_channel*32*3*3)
        #self.fc1 = nn.Linear(self.input_channel*16*3*3, self.input_channel*16*3*3*2, bias=False)
        #self.fc2 = nn.Linear(self.input_channel*16*3*3*2, 9 * self.output_channel, bias=False)# output (bs, 9 * oc)
        #fully convolution
        self.fc1 = nn.Conv2d(self.input_channel*16, self.input_channel, 1, padding=0, bias=False)  #(bs, in_channel*16, 3, 3)->(bs, in_channel, 3, 3)
        self.fc2 = nn.Conv2d(self.input_channel*2, self.output_channel, 1, padding=0, bias=False)  #(bs, in_channel*2, 3, 3)->(bs, oc, 3, 3)

        self.sequence1 = nn.Sequential(
            self.cov1,
            Softplus(),
            nn.AvgPool2d(3, stride=1, padding=1),
            self.cov2,
            Softplus(),
            nn.AvgPool2d(3, stride=1, padding=1)
        )
        self.sequence2 = nn.Sequential(
            self.fc1,
            Softplus()
        )
        self.sequence3 = nn.Sequential(
            self.fc2
        )

    def forward(self, x):
        y = self.sequence1(x)
        y = self.sequence2(y)
        return self.sequence3(torch.cat([y, x], dim=1))
    

class CNN2LPositive(CNN2L):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.sequence3 = nn.Sequential(
            self.fc2,
            Softplus()
        )

    def forward(self, x):
        y = self.sequence1(x)
        y = self.sequence2(y)
        return self.sequence3(torch.cat([y, x], dim=1))
    

class CNN2LNegative(CNN2L):
    def __init__(self, *args, **kwards):
        super().__init__(*args, **kwards)
        self.sequence3 = nn.Sequential(
            self.fc2,
            Softplus()
        )

    def forward(self, x):
        y = self.sequence1(x)
        y = self.sequence2(y)
        return -self.sequence3(torch.cat([y, x], dim=1))

