"""metrics.py — image-quality + downstream-relevant metrics for cardiac CT.

All functions accept torch tensors in volume space (B, 1, D, H, W) on any
device. They are stateless and are meant to be composed in `run_eval.py`.

PSNR / SSIM / NRMSE: image-quality metrics on the [-1, 1] HU-normalized
domain (consistent with VAE input/output range).

Masked and boundary metrics support paper-oriented evaluation: heart-region
errors, heart-boundary errors, and edge/gradient preservation.

dice_lumen: a *stub* downstream-task metric — Dice between two binarized
masks at HU > threshold. Not yet calibrated for TAVI; placeholder until
the lumen-segmentation pipeline lands (Week 9+, see brief §07 novelty 3).

Per `.claude/rules/python-code-conventions.md`:
    - no float == ; uses torch.allclose where applicable
    - probabilities clamped before log (none used here)
    - type hints on public functions
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from monai.metrics import SSIMMetric


def _as_bool_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Return `mask` as a 5D bool tensor broadcast-compatible with `ref`."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        mask = mask.unsqueeze(1)
    if mask.dim() != ref.dim():
        raise ValueError(f"mask dim {mask.dim()} does not match ref dim {ref.dim()}")
    if mask.shape[0] not in {1, ref.shape[0]}:
        raise ValueError(f"mask batch {mask.shape[0]} incompatible with ref batch {ref.shape[0]}")
    if mask.shape[1] not in {1, ref.shape[1]}:
        raise ValueError(f"mask channels {mask.shape[1]} incompatible with ref channels {ref.shape[1]}")
    if tuple(mask.shape[-3:]) != tuple(ref.shape[-3:]):
        raise ValueError(f"mask spatial shape {tuple(mask.shape[-3:])} != {tuple(ref.shape[-3:])}")
    return mask.to(device=ref.device, dtype=torch.bool)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample mean of `values` inside `mask`.

    Args:
        values: `(B, C, D, H, W)` tensor.
        mask: bool-like `(B, 1, D, H, W)`, `(1, 1, D, H, W)`, or `(D, H, W)`.
    """
    mask_bool = _as_bool_mask(mask, values)
    mask_f = mask_bool.to(dtype=values.dtype)
    if mask_f.shape[0] == 1 and values.shape[0] > 1:
        mask_f = mask_f.expand(values.shape[0], -1, -1, -1, -1)
    if mask_f.shape[1] == 1 and values.shape[1] > 1:
        mask_f = mask_f.expand(-1, values.shape[1], -1, -1, -1)
    denom = mask_f.flatten(1).sum(dim=1)
    if torch.any(denom <= 0):
        raise ValueError("masked metric received an empty mask")
    num = (values * mask_f).flatten(1).sum(dim=1)
    return num / denom.clamp_min(eps)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample MAE inside `mask`."""
    return masked_mean((pred - target).abs(), mask)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample RMSE inside `mask`."""
    return masked_mean((pred - target).pow(2), mask).sqrt()


def make_boundary_band(mask: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Return a dilate-minus-erode boundary band for a 3D binary mask.

    Args:
        mask: bool-like `(B, 1, D, H, W)`, `(1, 1, D, H, W)`, or `(D, H, W)`.
        radius: band radius in voxels. With 1 mm preprocessed spacing, radius 3
            corresponds to a roughly 3 mm neighborhood on each side.
    """
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if mask.dim() == 3:
        ref = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        ref = mask.unsqueeze(1)
    else:
        ref = mask
    mask_bool = _as_bool_mask(mask, ref)
    mask_f = mask_bool.float()
    kernel = 2 * radius + 1

    dilated = F.max_pool3d(mask_f, kernel_size=kernel, stride=1, padding=radius) > 0.5

    inv = (~mask_bool).float()
    padded_inv = F.pad(inv, (radius, radius, radius, radius, radius, radius), value=1.0)
    eroded = (1.0 - F.max_pool3d(padded_inv, kernel_size=kernel, stride=1)) > 0.5
    return torch.logical_and(dilated, torch.logical_not(eroded))


def gradient_magnitude(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Forward-difference 3D gradient magnitude with original spatial shape."""
    dz = torch.zeros_like(volume)
    dy = torch.zeros_like(volume)
    dx = torch.zeros_like(volume)
    dz[..., :-1, :, :] = volume[..., 1:, :, :] - volume[..., :-1, :, :]
    dy[..., :, :-1, :] = volume[..., :, 1:, :] - volume[..., :, :-1, :]
    dx[..., :, :, :-1] = volume[..., :, :, 1:] - volume[..., :, :, :-1]
    return torch.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + eps)


def masked_gradient_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample L1 error between gradient magnitudes inside `mask`."""
    return masked_mean((gradient_magnitude(pred) - gradient_magnitude(target)).abs(), mask)


# ------------------------------------------------------------------
# Coronary-lumen metrics (real, GT-mask based — replaces dice_lumen stub
# for the downstream-fidelity leg; see plan 2026-06-16_coronary-lumen-*).
# Normalization matches preprocessing.py: HU [-1024, 3071] -> [-1, 1].
# ------------------------------------------------------------------
_HU_CLIP_LO: float = -1024.0
_HU_CLIP_HI: float = 3071.0
LUMEN_CONTRAST_HU: float = 200.0  # contrast-enhanced vessel threshold for lumen Dice


def hu_to_norm(hu: float) -> float:
    """Map an HU value to the preprocessing [-1, 1] normalized domain."""
    return (hu - _HU_CLIP_LO) / (_HU_CLIP_HI - _HU_CLIP_LO) * 2.0 - 1.0


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation of a 3D binary mask by `radius` voxels."""
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    if mask.dim() == 3:
        ref = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 4:
        ref = mask.unsqueeze(1)
    else:
        ref = mask
    mask_bool = _as_bool_mask(mask, ref)
    kernel = 2 * radius + 1
    return F.max_pool3d(mask_bool.float(), kernel_size=kernel, stride=1, padding=radius) > 0.5


def masked_sharpness(volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean gradient magnitude inside `mask` — an absolute sharpness measure.

    A blurry (regression-to-mean) restoration has lower in-lumen gradient energy;
    a sharp restoration recovers the clean volume's edge content.
    """
    return masked_mean(gradient_magnitude(volume), mask)


def lumen_contrast(
    volume: torch.Tensor, lumen_mask: torch.Tensor, ring_mask: torch.Tensor
) -> torch.Tensor:
    """Lumen-to-surround contrast: mean(volume|lumen) - mean(volume|ring).

    `ring_mask` is the surrounding (e.g. dilated-minus-lumen) tissue band.
    Higher contrast = better-preserved vessel against background.
    """
    return masked_mean(volume, lumen_mask) - masked_mean(volume, ring_mask)


def dice_lumen_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    region_mask: torch.Tensor,
    hu_threshold: float = LUMEN_CONTRAST_HU,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Dice of high-contrast voxels (HU > threshold) within `region_mask`.

    Unlike `dice_lumen` (global, arbitrary 0.5 threshold), this restricts the
    comparison to a coronary-lumen neighborhood and uses a contrast-level HU
    threshold, so it measures geometric recovery of the bright vessel relative
    to the clean target — not background HU agreement.
    """
    thr = hu_to_norm(hu_threshold)
    region = _as_bool_mask(region_mask, pred)
    p_bin = ((pred > thr) & region).float()
    t_bin = ((target > thr) & region).float()
    intersect = (p_bin * t_bin).flatten(1).sum(dim=1)
    union = p_bin.flatten(1).sum(dim=1) + t_bin.flatten(1).sum(dim=1)
    return (2.0 * intersect + eps) / (union + eps)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Peak-signal-to-noise ratio (dB), elementwise then per-sample mean.

    Default `data_range=2.0` for the [-1, 1] HU-normalized domain.
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(data_range ** 2 / mse)


def ssim_3d(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """SSIM-3D via MONAI; returns per-sample SSIM in [0, 1]."""
    metric = SSIMMetric(spatial_dims=3, data_range=data_range, reduction="none")
    out = metric(pred, target)
    # MONAI returns (B, 1) tensor of SSIMs.
    return out.flatten()


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized RMSE: sqrt(MSE) / (max(target) - min(target)). Per-sample."""
    diff_sq = (pred - target).pow(2).flatten(1).mean(dim=1)
    rmse_ = diff_sq.sqrt()
    target_range = (target.flatten(1).max(dim=1).values
                    - target.flatten(1).min(dim=1).values).clamp_min(1e-8)
    return rmse_ / target_range


def dice_lumen(
    pred: torch.Tensor,
    target: torch.Tensor,
    hu_threshold_norm: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Stub Dice over high-intensity (≈ contrast-filled) voxels.

    NOT a calibrated lumen segmentation — uses a fixed HU-normalized threshold
    as a placeholder for the proper TAVI lumen pipeline (Week 9+).

    Args:
        pred, target: (B, 1, D, H, W), values in [-1, 1].
        hu_threshold_norm: threshold in normalized HU. 0.5 corresponds
            roughly to HU ≈ 1023 with our [-1024, 3071] clip.
    """
    p_bin = (pred > hu_threshold_norm).float()
    t_bin = (target > hu_threshold_norm).float()
    intersect = (p_bin * t_bin).flatten(1).sum(dim=1)
    union = p_bin.flatten(1).sum(dim=1) + t_bin.flatten(1).sum(dim=1)
    return (2.0 * intersect + eps) / (union + eps)


def all_metrics(
    pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0
) -> dict[str, torch.Tensor]:
    """Compute the full per-sample metric bundle."""
    return {
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim_3d(pred, target, data_range=data_range),
        "nrmse": nrmse(pred, target),
        "dice_lumen_stub": dice_lumen(pred, target),
    }
