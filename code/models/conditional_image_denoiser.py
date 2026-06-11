"""Generic 3D diffusion denoiser for voxel-space conditional models."""

from __future__ import annotations

import torch
from monai.networks.nets import DiffusionModelUNet

DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128, 128)
DEFAULT_NUM_RES_BLOCKS: tuple[int, ...] = (2, 2, 2, 2)
DEFAULT_ATTENTION_LEVELS: tuple[bool, ...] = (False, False, True, True)
DEFAULT_NUM_HEAD_CHANNELS: tuple[int, ...] = (0, 0, 32, 32)
DEFAULT_NORM_NUM_GROUPS: int = 32


class ConditionalImageDenoiser(DiffusionModelUNet):
    """3D UNet denoiser with arbitrary target and condition channel counts.

    This is the voxel-space counterpart of `ConditionalDenoiser`. It keeps the
    same concat-conditioning convention but does not assume the target and
    condition have the same number of channels.
    """

    def __init__(
        self,
        target_channels: int = 1,
        condition_channels: int = 3,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        num_res_blocks: tuple[int, ...] = DEFAULT_NUM_RES_BLOCKS,
        attention_levels: tuple[bool, ...] = DEFAULT_ATTENTION_LEVELS,
        num_head_channels: tuple[int, ...] = DEFAULT_NUM_HEAD_CHANNELS,
        norm_num_groups: int = DEFAULT_NORM_NUM_GROUPS,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=target_channels + condition_channels,
            out_channels=target_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            with_conditioning=False,
        )
        self.target_channels = int(target_channels)
        self.condition_channels = int(condition_channels)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Denoise concat-conditioned voxel-space input."""
        return super().forward(x=x, timesteps=timesteps, context=None)
