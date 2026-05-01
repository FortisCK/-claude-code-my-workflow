"""metrics.py — image-quality + downstream-relevant metrics for cardiac CT.

All functions accept torch tensors in volume space (B, 1, D, H, W) on any
device. They are stateless and are meant to be composed in `run_eval.py`.

PSNR / SSIM / NRMSE: image-quality metrics on the [-1, 1] HU-normalized
domain (consistent with VAE input/output range).

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
