"""Small U-Net used as the first Task 1 segmentation baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """Two-convolution block with batch normalization."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """A compact U-Net for CPU smoke tests and baseline plumbing.

    Upsampling uses explicit interpolation to the encoder feature size, so the
    network preserves arbitrary input height/width without requiring divisibility
    by powers of two.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.bridge = ConvBlock(c2, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bridge = self.bridge(self.pool(enc2))

        up2 = F.interpolate(
            bridge,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = F.interpolate(
            dec2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.out(dec1)
