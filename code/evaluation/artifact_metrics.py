"""Artifact-aware evaluation helpers for cardiac CT correction.

These helpers keep paper-oriented metrics shared across diffusion, supervised
U-Net, and future v2 evaluations: global image metrics, heart-region errors,
heart-boundary errors, boundary-gradient errors, and severity summaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from code.evaluation.metrics import (
    all_metrics,
    dice_lumen_masked,
    dilate_mask,
    lumen_contrast,
    make_boundary_band,
    masked_gradient_l1,
    masked_mae,
    masked_rmse,
    masked_sharpness,
)

LUMEN_RING_RADIUS: int = 3  # voxels (~3 mm at 1 mm spacing) for lumen contrast/Dice band

HU_SCALE: float = 2047.5  # [-1, 1] maps to [-1024, 3071].


def load_heart_mask(
    case_id: str,
    processed_dir: Path,
    ref_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Load heart mask as `(1, 1, D, H, W)` bool tensor."""
    npz_path = processed_dir / f"case_{case_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        mask = torch.from_numpy(data["heart_mask"]).bool()
    if tuple(mask.shape) != tuple(ref_shape):
        raise ValueError(
            f"heart mask shape {tuple(mask.shape)} does not match volume shape {ref_shape} "
            f"for case {case_id}"
        )
    return mask.unsqueeze(0).unsqueeze(0)


def load_lumen_mask(
    case_id: str,
    processed_dir: Path,
    ref_shape: tuple[int, int, int],
) -> torch.Tensor | None:
    """Load the aligned coronary lumen mask as `(1, 1, D, H, W)` bool, or None.

    Non-destructive sidecar produced by scripts/python/generate_lumen_masks.py at
    `<processed_dir>/../lumen_masks/lumen_mask_<id>.npy`. Returns None if absent,
    so evaluations on caches/cases without lumen masks degrade gracefully.
    """
    mask_path = processed_dir.parent / "lumen_masks" / f"lumen_mask_{case_id}.npy"
    if not mask_path.exists():
        return None
    mask = torch.from_numpy(np.load(mask_path)).bool()
    if tuple(mask.shape) != tuple(ref_shape):
        raise ValueError(
            f"lumen mask shape {tuple(mask.shape)} does not match volume shape {ref_shape} "
            f"for case {case_id}"
        )
    return mask.unsqueeze(0).unsqueeze(0)


def _scalar(value: torch.Tensor) -> float:
    return float(value.flatten()[0].item())


def metric_row(
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    heart_mask: torch.Tensor | None = None,
    lumen_mask: torch.Tensor | None = None,
    boundary_radius: int = 3,
) -> dict[str, float]:
    """Return global plus optional heart/boundary/lumen metrics for one prediction."""
    metrics = all_metrics(pred.float(), target.float())
    abs_err = (pred.float() - target.float()).abs()
    out = {
        f"{prefix}_mae_norm": float(abs_err.mean().item()),
        f"{prefix}_mae_hu": float(abs_err.mean().item() * HU_SCALE),
        f"{prefix}_psnr": float(metrics["psnr"].item()),
        f"{prefix}_ssim": float(metrics["ssim"].item()),
        f"{prefix}_nrmse": float(metrics["nrmse"].item()),
        f"{prefix}_dice_lumen_stub": float(metrics["dice_lumen_stub"].item()),
    }
    if heart_mask is not None:
        mask = heart_mask.to(device=pred.device)
        boundary = make_boundary_band(mask, radius=boundary_radius).to(device=pred.device)
        heart_mae = _scalar(masked_mae(pred.float(), target.float(), mask))
        heart_rmse = _scalar(masked_rmse(pred.float(), target.float(), mask))
        boundary_mae = _scalar(masked_mae(pred.float(), target.float(), boundary))
        boundary_rmse = _scalar(masked_rmse(pred.float(), target.float(), boundary))
        boundary_grad_l1 = _scalar(masked_gradient_l1(pred.float(), target.float(), boundary))
        out.update(
            {
                f"{prefix}_heart_mae_norm": heart_mae,
                f"{prefix}_heart_mae_hu": heart_mae * HU_SCALE,
                f"{prefix}_heart_rmse_norm": heart_rmse,
                f"{prefix}_heart_rmse_hu": heart_rmse * HU_SCALE,
                f"{prefix}_boundary_mae_norm": boundary_mae,
                f"{prefix}_boundary_mae_hu": boundary_mae * HU_SCALE,
                f"{prefix}_boundary_rmse_norm": boundary_rmse,
                f"{prefix}_boundary_rmse_hu": boundary_rmse * HU_SCALE,
                f"{prefix}_boundary_gradient_l1_norm": boundary_grad_l1,
                f"{prefix}_boundary_gradient_l1_hu": boundary_grad_l1 * HU_SCALE,
                f"{prefix}_boundary_voxels": float(boundary.sum().item()),
            }
        )

    if lumen_mask is not None:
        lm = lumen_mask.to(device=pred.device)
        # Surrounding tissue ring (epicardial fat / myocardium) for contrast.
        ring = (dilate_mask(lm, LUMEN_RING_RADIUS) & ~lm.bool()).to(device=pred.device)
        region = dilate_mask(lm, LUMEN_RING_RADIUS).to(device=pred.device)
        lumen_mae = _scalar(masked_mae(pred.float(), target.float(), lm))
        lumen_rmse = _scalar(masked_rmse(pred.float(), target.float(), lm))
        lumen_grad_l1 = _scalar(masked_gradient_l1(pred.float(), target.float(), lm))
        lumen_sharp = _scalar(masked_sharpness(pred.float(), lm))
        lumen_sharp_ref = _scalar(masked_sharpness(target.float(), lm))
        lumen_cnr = _scalar(lumen_contrast(pred.float(), lm, ring))
        lumen_cnr_ref = _scalar(lumen_contrast(target.float(), lm, ring))
        lumen_dice = _scalar(dice_lumen_masked(pred.float(), target.float(), region))
        out.update(
            {
                f"{prefix}_lumen_mae_norm": lumen_mae,
                f"{prefix}_lumen_mae_hu": lumen_mae * HU_SCALE,
                f"{prefix}_lumen_rmse_hu": lumen_rmse * HU_SCALE,
                f"{prefix}_lumen_gradient_l1_hu": lumen_grad_l1 * HU_SCALE,
                f"{prefix}_lumen_sharpness_hu": lumen_sharp * HU_SCALE,
                f"{prefix}_lumen_sharpness_ref_hu": lumen_sharp_ref * HU_SCALE,
                f"{prefix}_lumen_cnr_hu": lumen_cnr * HU_SCALE,
                f"{prefix}_lumen_cnr_ref_hu": lumen_cnr_ref * HU_SCALE,
                f"{prefix}_lumen_dice": lumen_dice,
                f"{prefix}_lumen_voxels": float(lm.sum().item()),
            }
        )

    return out


def _summarize_numeric(rows: list[dict[str, object]], numeric_keys: list[str]) -> dict[str, object]:
    block: dict[str, object] = {"n_cases": len(rows)}
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        block[f"{key}_mean"] = float(vals.mean())
        block[f"{key}_std"] = float(vals.std(ddof=0))
    return block


def assign_artifact_severity(
    rows: list[dict[str, object]],
    score_key: str = "corrupted_heart_mae_hu",
) -> None:
    """Assign mild/medium/severe labels by corrupted-input error rank."""
    if not rows or score_key not in rows[0]:
        return
    order = sorted(range(len(rows)), key=lambda i: float(rows[i][score_key]))
    n = len(order)
    for rank, row_idx in enumerate(order):
        frac = (rank + 0.5) / max(1, n)
        if frac < 1.0 / 3.0:
            label = "mild"
        elif frac < 2.0 / 3.0:
            label = "medium"
        else:
            label = "severe"
        rows[row_idx]["artifact_severity"] = label
        rows[row_idx]["artifact_severity_rank"] = float(rank + 1)
        rows[row_idx]["artifact_severity_score_key"] = score_key


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Summarize numeric fields overall, by split, and by artifact severity."""
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    out: dict[str, object] = {"n_cases": len(rows), "by_split": {}, "by_artifact_severity": {}}
    for split_name in sorted({str(r["split"]) for r in rows}):
        split_rows = [r for r in rows if r["split"] == split_name]
        out["by_split"][split_name] = _summarize_numeric(split_rows, numeric_keys)

    if "artifact_severity" in rows[0]:
        for severity in ("mild", "medium", "severe"):
            severity_rows = [r for r in rows if r.get("artifact_severity") == severity]
            if severity_rows:
                out["by_artifact_severity"][severity] = _summarize_numeric(
                    severity_rows, numeric_keys
                )

    out["all"] = _summarize_numeric(rows, numeric_keys)
    return out
