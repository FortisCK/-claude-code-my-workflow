"""Reliability gate for posterior residual diffusion refinement."""

from __future__ import annotations

import torch
from monai.networks.nets import BasicUNet
from torch import nn


class ResidualGateNet3D(nn.Module):
    """Predict a local reliability gate for residual-diffusion proposals.

    The default feature tensor is:

    ```text
    [corrupted, unet_initial, corrupted - unet_initial,
     posterior_mean_residual, posterior_std_residual, |grad unet_initial|]
    ```

    The final correction is formed outside or inside the module as:

    ```text
    corrected = unet_initial + gate * posterior_mean_residual
    gate = g_max * sigmoid(logits + init_bias)
    ```

    A negative `init_bias` makes the gate conservative at initialization.
    """

    def __init__(
        self,
        in_channels: int = 6,
        features: tuple[int, int, int, int, int, int] = (16, 16, 32, 64, 128, 16),
        g_max: float = 0.25,
        init_bias: float = -4.0,
        clamp_output: bool = False,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if g_max <= 0:
            raise ValueError(f"g_max must be positive, got {g_max}")
        self.in_channels = int(in_channels)
        self.g_max = float(g_max)
        self.init_bias = float(init_bias)
        self.clamp_output = bool(clamp_output)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.backbone = BasicUNet(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=1,
            features=features,
        )

    def gate_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return the bounded gate map shaped `(B, 1, D, H, W)`."""
        if features.dim() != 5:
            raise ValueError(f"expected 5D feature tensor, got {tuple(features.shape)}")
        if features.shape[1] != self.in_channels:
            raise ValueError(f"expected C={self.in_channels}, got C={features.shape[1]}")
        logits = self.backbone(features)
        return self.g_max * torch.sigmoid(logits + self.init_bias)

    def forward(
        self,
        features: torch.Tensor,
        initial: torch.Tensor | None = None,
        residual_mean: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return gate or corrected image when `initial` and residual are given."""
        gate = self.gate_from_features(features)
        if initial is None or residual_mean is None:
            return gate
        if initial.shape != residual_mean.shape:
            raise ValueError(
                f"initial shape {tuple(initial.shape)} != residual_mean {tuple(residual_mean.shape)}"
            )
        if initial.dim() != 5 or initial.shape[1] != 1:
            raise ValueError(f"expected initial `(B, 1, D, H, W)`, got {tuple(initial.shape)}")
        corrected = initial + gate * residual_mean
        if self.clamp_output:
            corrected = corrected.clamp(self.clamp_min, self.clamp_max)
        if return_gate:
            return corrected, gate
        return corrected
