import torch
from torch import tensor, nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

__all__ = ["UNet"]

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, sn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        if sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act_fn = nn.ReLU(inplace=True)

        self.sequence = nn.Sequential(
                self.conv1,
                self.bn1,
                self.act_fn,
                self.conv2,
                self.bn2,
                self.act_fn
        )

    def forward(self, x):
        return self.sequence(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub=None, sn=False):
        super().__init__()
        self.conv1 = BaseConv(in_channels, out_channels, sn)
        self.pool = nn.MaxPool2d(2)

        self.upconv = nn.ConvTranspose2d(out_channels, out_channels // 2, 2, stride=2)
        self.conv2 = BaseConv(out_channels // 2 + in_channels, in_channels, sn)
        self.sub = sub

        if sub is not None:
            self.sequence = nn.Sequential(
                    self.pool,
                    self.conv1,
                    self.sub,
                    self.upconv
            )
        else:
            self.sequence = nn.Sequential(
                    self.pool,
                    self.conv1,
                    self.upconv
            )

    def forward(self, x):
        x1 = self.sequence(x)

        diffH = x.size()[-2] - x1.size()[-2]
        diffW = x.size()[-1] - x1.size()[-1]
        x1 = F.pad(x1, (diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2))

        x2 = torch.cat([x, x1], dim=1)
        return self.conv2(x2)


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, sn=False):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv0 = BaseConv(input_channels, 64, sn=sn)
        self.block4 = UNetBlock(512, 1024, sn=sn)
        self.block3 = UNetBlock(256, 512, sub=self.block4, sn=sn)
        self.block2 = UNetBlock(128, 256, sub=self.block3, sn=sn)
        self.block1 = UNetBlock(64, 128, sub=self.block2, sn=sn)

        self.out_conv = nn.Conv2d(64, output_channels, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.block1(x)
        x = self.out_conv(x)
        return x


