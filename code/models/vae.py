"""vae.py — 3D KL-VAE for cardiac-CT latent encoding.

Thin wrapper around `monai.networks.nets.AutoencoderKL` with cardiac-CT
defaults. Encodes a (1, 192³) HU-normalized volume into a 4-channel × 24³
latent (8× spatial downsample, MONAI default with 4 levels).

Usage:
    from code.models.vae import CardiacVAE
    vae = CardiacVAE()                       # defaults
    z_mu, z_sigma = vae.encode(v)            # MONAI returns (mu, sigma)
    z = vae.encode_to_latent(v)              # deterministic mu (no sampling)
    v_rec = vae.decode(z)
    v_rec, z_mu, z_sigma = vae(v)            # full VAE forward (recon + posterior)

Adapted from: external/GenerativeModels/tutorials/generative/3d_autoencoderkl/
              3d_autoencoderkl_tutorial.py constructor.

License: MONAI is Apache-2.0; this wrapper inherits.

Per `.claude/rules/python-code-conventions.md`:
    - type hints on public methods
    - module-level constants (no magic numbers in __init__)
    - immutable defaults
"""

from __future__ import annotations

import torch
from monai.networks.nets import AutoencoderKL

# ---- Cardiac-CT defaults ---------------------------------------------------
# 4-stage encoder (channels grows; downsample once per stage except last).
# 192 → 96 → 48 → 24 spatial; latent at 24³ × 4 channels (≈110 KB / volume).
DEFAULT_CHANNELS: tuple[int, ...] = (32, 64, 128, 128)
DEFAULT_NUM_RES_BLOCKS: tuple[int, ...] = (2, 2, 2, 2)
DEFAULT_ATTENTION_LEVELS: tuple[bool, ...] = (False, False, False, True)
DEFAULT_LATENT_CHANNELS: int = 4
DEFAULT_NORM_NUM_GROUPS: int = 32


class CardiacVAE(AutoencoderKL):
    """3D KL-VAE specialized for cardiac CT.

    Defaults produce a 4-channel × 24³ latent from a (1, 192, 192, 192) input.

    Parameters
    ----------
    in_channels, out_channels : int
        Both default to 1 (single-channel HU volume).
    channels : tuple[int, ...]
        Encoder/decoder feature widths per stage.
    latent_channels : int
        Latent depth.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        num_res_blocks: tuple[int, ...] = DEFAULT_NUM_RES_BLOCKS,
        attention_levels: tuple[bool, ...] = DEFAULT_ATTENTION_LEVELS,
        latent_channels: int = DEFAULT_LATENT_CHANNELS,
        norm_num_groups: int = DEFAULT_NORM_NUM_GROUPS,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            attention_levels=attention_levels,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            use_checkpoint=use_checkpoint,
        )

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic encoding — return posterior mean only (no sampling).

        Used at diffusion-Stage-2 training time when the VAE is frozen and
        we want a stable z_clean / z_cond.
        """
        z_mu, _ = self.encode(x)
        return z_mu

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to volume space."""
        return self.decode(z)
