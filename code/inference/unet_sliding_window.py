"""Full-volume sliding-window inference for supervised residual U-Net baselines."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from monai.inferers import SlidingWindowInferer

from code.models.residual_unet import ResidualUNet3D

log = logging.getLogger(__name__)


def _as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    out = tuple(int(v) for v in value)
    if len(out) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {value!r}")
    return out


@torch.inference_mode()
def sliding_window_unet_correct(
    v_corrupted: torch.Tensor,
    model: ResidualUNet3D,
    roi_size: int | Sequence[int] = (128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sw_device: torch.device | str = "cuda",
    output_device: torch.device | str = "cpu",
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    progress: bool = False,
) -> torch.Tensor:
    """Correct a full volume by overlapping U-Net patch inference."""
    if v_corrupted.dim() != 5:
        raise ValueError(f"expected 5D `(B, C, D, H, W)` tensor, got {tuple(v_corrupted.shape)}")
    if v_corrupted.shape[1] != 1:
        raise ValueError(f"expected single-channel input, got C={v_corrupted.shape[1]}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if sw_batch_size < 1:
        raise ValueError(f"sw_batch_size must be >= 1, got {sw_batch_size}")

    roi = _as_3tuple(roi_size, "roi_size")
    spatial = tuple(int(v) for v in v_corrupted.shape[-3:])
    log.info(
        "[unet sliding] shape=%s roi=%s overlap=%.2f mode=%s",
        spatial,
        roi,
        overlap,
        blend_mode,
    )

    model.eval()
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

    def predictor(patches: torch.Tensor) -> torch.Tensor:
        return model(patches.to(sw_dev, non_blocking=True)).float()

    corrected = inferer(v_corrupted.float(), predictor)
    return corrected.float()
