"""posterior_sample.py — N-sample posterior sampling for conditional EDM.

Given a frozen VAE and a frozen conditional EDM denoiser, draw N parallel
Heun chains starting from independent noise, all conditioned on the same
corrupted volume. Decode each to volume space; aggregate to (mean, std).

The mean is the point estimate (V_corrected); the std is the per-voxel
uncertainty map — a native byproduct of generative posterior sampling
(brief §07 novelty 2).

Usage:
    from code.inference.posterior_sample import posterior_sample
    v_mean, v_std = posterior_sample(
        v_corrupted, vae, edm,
        n_samples=16, num_steps=50, device='cuda'
    )

Per `.claude/rules/python-code-conventions.md`:
    - torch.no_grad() / inference_mode for hot paths
    - decollate / SlidingWindowInferer not used here (we operate on a single
      already-cropped 192³ ROI)
    - type hints on public functions
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from code.models.edm import EDM
from code.models.vae import CardiacVAE

log = logging.getLogger(__name__)


@torch.inference_mode()
def posterior_sample(
    v_corrupted: torch.Tensor,
    vae: CardiacVAE,
    edm: EDM,
    n_samples: int = 16,
    num_steps: int = 50,
    deterministic: bool = False,
    seed: Optional[int] = None,
    return_samples: bool = False,
) -> dict[str, torch.Tensor]:
    """Run N posterior chains for one (or a small batch of) corrupted volume(s).

    Args:
        v_corrupted: (B, 1, D, H, W) HU-normalized corrupted volume(s).
        vae: frozen `CardiacVAE` (eval mode).
        edm: frozen `EDM` wrapping a `ConditionalDenoiser`.
        n_samples: posterior chains per condition (each gets independent noise).
        num_steps: Heun-2 sampler steps (50 = paper default).
        deterministic: if True, S_churn=0 (deterministic Heun).
        seed: optional initial-noise seed for reproducibility.
        return_samples: if True, also return the full (B*N, 1, D, H, W) tensor.

    Returns:
        dict with keys:
            "mean": (B, 1, D, H, W) posterior mean (V_corrected)
            "std":  (B, 1, D, H, W) per-voxel std (V_uncertainty)
            "samples": (B*N, 1, D, H, W) [only if return_samples]
    """
    vae.eval()
    edm.eval()

    # Encode the corrupted volume once → z_cond (B, latent_ch, ...)
    z_cond = vae.encode_to_latent(v_corrupted)
    b = z_cond.shape[0]

    # Sample N latents per condition; sample() internally repeat_interleaves
    # z_cond and returns shape (B*N, latent_ch, ...).
    z_pred = edm.sample(
        z_cond=z_cond,
        num_steps=num_steps,
        n_samples=n_samples,
        deterministic=deterministic,
        seed=seed,
    )

    # Decode each sample to volume space.
    v_pred = vae.decode_from_latent(z_pred)         # (B*N, 1, D, H, W)

    # Reshape to (B, N, 1, D, H, W) for aggregation.
    v_per_cond = v_pred.view(b, n_samples, *v_pred.shape[1:])
    v_mean = v_per_cond.mean(dim=1)                 # (B, 1, D, H, W)
    v_std = v_per_cond.std(dim=1, unbiased=False)

    out: dict[str, torch.Tensor] = {"mean": v_mean, "std": v_std}
    if return_samples:
        out["samples"] = v_pred
    log.info(
        "[posterior] B=%d N=%d steps=%d  mean range [%.3f, %.3f]  std mean=%.3f",
        b, n_samples, num_steps, float(v_mean.min()), float(v_mean.max()),
        float(v_std.mean()),
    )
    return out
