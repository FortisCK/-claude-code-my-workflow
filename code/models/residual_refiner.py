"""Residual refiner models for U-Net-initialized artifact correction."""

from __future__ import annotations

import torch
from monai.networks.nets import BasicUNet
from torch import nn


class ResidualRefinerUNet3D(nn.Module):
    """Predict the residual left by a deterministic correction initializer.

    The model consumes a three-channel condition by default:

    ```text
    [corrupted, initial_prediction, corrupted - initial_prediction]
    ```

    and predicts `clean - initial_prediction`. The returned corrected image is:

    ```text
    corrected = initial_prediction + predicted_residual
    ```
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
        clamp_output: bool = False,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("ResidualRefinerUNet3D currently expects exactly 3 input channels")
        if out_channels != 1:
            raise ValueError("ResidualRefinerUNet3D currently expects exactly 1 output channel")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.clamp_output = bool(clamp_output)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.backbone = BasicUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
        )

    @staticmethod
    def make_condition(corrupted: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
        """Build `[corrupted, initial, corrupted - initial]` condition tensor."""
        if corrupted.shape != initial.shape:
            raise ValueError(f"corrupted shape {tuple(corrupted.shape)} != initial {tuple(initial.shape)}")
        if corrupted.dim() != 5 or corrupted.shape[1] != 1:
            raise ValueError(f"expected `(B, 1, D, H, W)`, got {tuple(corrupted.shape)}")
        return torch.cat([corrupted, initial, corrupted - initial], dim=1)

    def forward(
        self,
        corrupted_or_condition: torch.Tensor,
        initial: torch.Tensor | None = None,
        return_residual: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if initial is None:
            condition = corrupted_or_condition
            if condition.dim() != 5 or condition.shape[1] != 3:
                raise ValueError(
                    "when `initial` is omitted, input must be a 3-channel condition tensor"
                )
            initial_pred = condition[:, 1:2]
        else:
            condition = self.make_condition(corrupted_or_condition, initial)
            initial_pred = initial

        residual = self.backbone(condition)
        corrected = initial_pred + residual
        if self.clamp_output:
            corrected = corrected.clamp(self.clamp_min, self.clamp_max)
        if return_residual:
            return corrected, residual
        return corrected
