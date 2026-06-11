"""Supervised 3D residual U-Net baseline for cardiac CT artifact correction."""

from __future__ import annotations

import torch
from monai.networks.nets import BasicUNet
from torch import nn


class ResidualUNet3D(nn.Module):
    """Predict a voxel-space correction residual from a corrupted 3D CT volume.

    The default forward pass returns the corrected volume:

    ```text
    corrected = corrupted + unet(corrupted)
    ```

    This keeps the supervised baseline aligned with the artifact-correction
    task: most anatomy should be preserved and the network focuses on residual
    motion artifacts.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
        residual: bool = True,
        clamp_output: bool = False,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.clamp_output = clamp_output
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.backbone = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

    def forward(
        self,
        corrupted: torch.Tensor,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        residual = self.backbone(corrupted)
        corrected = corrupted + residual if self.residual else residual
        if self.clamp_output:
            corrected = corrected.clamp(self.clamp_min, self.clamp_max)
        if return_residual:
            return corrected, residual
        return corrected
