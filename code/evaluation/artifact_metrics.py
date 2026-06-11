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
    make_boundary_band,
    masked_gradient_l1,
    masked_mae,
    masked_rmse,
)

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


def _scalar(value: torch.Tensor) -> float:
    return float(value.flatten()[0].item())


def metric_row(
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    heart_mask: torch.Tensor | None = None,
    boundary_radius: int = 3,
) -> dict[str, float]:
    """Return global plus optional heart/boundary metrics for one prediction."""
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
    if heart_mask is None:
        return out

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
