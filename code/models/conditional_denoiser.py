"""conditional_denoiser.py — 3D UNet that denoises latent z_t conditioned on z_cond (concat).

Wraps `monai.networks.nets.DiffusionModelUNet` with cardiac-CT defaults and
sets `in_channels = 2 * latent_channels` so that we can concatenate the
noisy clean latent (z_t) with the corrupted-conditioning latent (z_cond)
along the channel axis. Output channels equal `latent_channels` (predicting
the denoised z_clean).

The forward signature matches the EDM wrapper's expectation:

    forward(x: (B, 2*latent_ch, 24,24,24), timesteps: (B,)) → (B, latent_ch, 24,24,24)

EDM wraps this and feeds it `c_in(σ) * z_input` and `c_noise(σ)` per Karras 2022.

Adapted from: MONAI 3D LDM tutorial DiffusionModelUNet constructor.

Per `.claude/rules/python-code-conventions.md`:
    - module-level constants (no magic numbers in __init__)
    - type hints on public methods
"""

from __future__ import annotations

import torch
from monai.networks.nets import DiffusionModelUNet

# ---- Latent-space defaults (24³ × 4 from CardiacVAE) -----------------------
DEFAULT_CHANNELS: tuple[int, ...] = (64, 128, 256, 256)
DEFAULT_NUM_RES_BLOCKS: tuple[int, ...] = (2, 2, 2, 2)
DEFAULT_ATTENTION_LEVELS: tuple[bool, ...] = (False, False, True, True)
DEFAULT_NUM_HEAD_CHANNELS: tuple[int, ...] = (0, 0, 64, 64)
DEFAULT_NORM_NUM_GROUPS: int = 32


class ConditionalDenoiser(DiffusionModelUNet):
    """3D UNet denoiser with channel-concat conditioning.

    Parameters
    ----------
    latent_channels : int
        Channels of the VAE latent. The UNet's `in_channels` is set to
        `2 * latent_channels` (z_t || z_cond) and `out_channels` to
        `latent_channels` (predicting denoised z_clean).
    """

    def __init__(
        self,
        latent_channels: int = 4,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        num_res_blocks: tuple[int, ...] = DEFAULT_NUM_RES_BLOCKS,
        attention_levels: tuple[bool, ...] = DEFAULT_ATTENTION_LEVELS,
        num_head_channels: tuple[int, ...] = DEFAULT_NUM_HEAD_CHANNELS,
        norm_num_groups: int = DEFAULT_NORM_NUM_GROUPS,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=2 * latent_channels,
            out_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            with_conditioning=False,  # concat-only; no cross-attn
        )
        self.latent_channels = latent_channels

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Denoise the concat-conditioned latent.

        Args:
            x: (B, 2 * latent_channels, D, H, W) — cat([z_t, z_cond], dim=1)
            timesteps: (B,) — float "time" embedding (EDM's c_noise(σ))
        """
        return super().forward(x=x, timesteps=timesteps, context=None)
