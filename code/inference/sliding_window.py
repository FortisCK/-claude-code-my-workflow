"""Full-volume sliding-window posterior inference for latent cardiac diffusion.

The trained Diffusion v1 model operates on 128^3 patches because direct 192^3
VAE encoding is too memory-heavy. This module wraps the patch posterior sampler
with MONAI's `SlidingWindowInferer` so a full 192^3 corrupted volume can be
corrected by overlapping patch inference and Gaussian blending.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from monai.inferers import SlidingWindowInferer

from code.inference.posterior_sample import posterior_sample
from code.models.edm import EDM
from code.models.vae import CardiacVAE

log = logging.getLogger(__name__)

DEFAULT_ROI_SIZE: tuple[int, int, int] = (128, 128, 128)
DEFAULT_OVERLAP: float = 0.5
DEFAULT_SIGMA_SCALE: float = 0.125


def _as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    out = tuple(int(v) for v in value)
    if len(out) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {value!r}")
    return out


@torch.inference_mode()
def sliding_window_posterior_sample(
    v_corrupted: torch.Tensor,
    vae: CardiacVAE,
    edm: EDM,
    roi_size: int | Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sw_batch_size: int = 1,
    n_samples: int = 1,
    num_steps: int = 50,
    deterministic: bool = True,
    sw_device: torch.device | str = "cuda",
    output_device: torch.device | str = "cpu",
    blend_mode: str = "gaussian",
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    progress: bool = False,
) -> dict[str, torch.Tensor]:
    """Correct a full volume by patch posterior sampling and weighted stitching.

    Args:
        v_corrupted: `(B, 1, D, H, W)` corrupted volume, typically on CPU.
        vae: frozen VAE used to encode/decode every patch.
        edm: frozen conditional EDM.
        roi_size: sliding-window patch size. Diffusion v1 was trained on 128^3.
        overlap: fractional overlap between adjacent windows.
        sw_batch_size: number of windows per predictor call. Keep at 1 on A6000
            for 128^3 VAE+EDM inference.
        n_samples: posterior chains per window.
        num_steps: EDM Heun steps per chain.
        deterministic: if true, disable EDM churn for deterministic sampling.
        sw_device: device for patch model execution.
        output_device: device used by MONAI for stitched outputs.
        blend_mode: MONAI blend mode, usually `"gaussian"` for patch edges.
        sigma_scale: Gaussian blend standard deviation coefficient.
        progress: show MONAI sliding-window progress bar.

    Returns:
        Dict with `"mean"` and `"std"` tensors on `output_device`, both shaped
        `(B, 1, D, H, W)`.
    """
    if v_corrupted.dim() != 5:
        raise ValueError(f"expected 5D `(B, C, D, H, W)` tensor, got {tuple(v_corrupted.shape)}")
    if v_corrupted.shape[1] != 1:
        raise ValueError(f"expected single-channel input, got C={v_corrupted.shape[1]}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if sw_batch_size < 1:
        raise ValueError(f"sw_batch_size must be >= 1, got {sw_batch_size}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    roi = _as_3tuple(roi_size, "roi_size")
    spatial = tuple(int(v) for v in v_corrupted.shape[-3:])
    log.info(
        "[sliding] shape=%s roi=%s overlap=%.2f mode=%s n_samples=%d steps=%d deterministic=%s",
        spatial,
        roi,
        overlap,
        blend_mode,
        n_samples,
        num_steps,
        deterministic,
    )

    vae.eval()
    edm.eval()
    sw_dev = torch.device(sw_device)
    out_dev = torch.device(output_device)
    inferer = SlidingWindowInferer(
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=blend_mode,
        sigma_scale=sigma_scale,
        sw_device=sw_dev,
        device=out_dev,
        progress=progress,
        cache_roi_weight_map=True,
    )

    def predictor(patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = posterior_sample(
            patches.to(sw_dev, non_blocking=True),
            vae,
            edm,
            n_samples=n_samples,
            num_steps=num_steps,
            deterministic=deterministic,
            seed=None,
        )
        return out["mean"].float(), out["std"].float()

    mean, std = inferer(v_corrupted.float(), predictor)
    return {"mean": mean.float(), "std": std.float()}
