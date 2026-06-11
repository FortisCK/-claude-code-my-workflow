#!/usr/bin/env python
"""Gate ablations for posterior residual diffusion v2.

This script freezes the current U-Net initializer and residual diffusion model,
samples a posterior residual mean/std map once per case, then evaluates several
release strategies:

* fixed scalar gates
* uncertainty shrinkage gates
* oracle gates that use the clean target as an upper-bound diagnostic

The oracle rows are not valid deployable models. They answer whether the sampled
residual contains useful directions if a reliability gate could choose where and
how strongly to release it.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_TMP = REPO_ROOT / "experiments" / "runs" / "_tmp"
RUN_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(RUN_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(RUN_TMP / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(RUN_TMP / "xdg_cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.evaluation.artifact_metrics import (  # noqa: E402
    HU_SCALE,
    load_heart_mask,
    metric_row,
)
from code.evaluation.metrics import make_boundary_band, masked_mean  # noqa: E402
from code.inference.residual_diffusion_sliding_window import (  # noqa: E402
    sliding_window_residual_diffusion_correct,
)
from code.training.train_residual_refiner import load_initializer  # noqa: E402
from scripts.python.evaluate_residual_diffusion_full_volume import (  # noqa: E402
    DEFAULT_OVERLAP,
    DEFAULT_ROI_SIZE,
    DEFAULT_SIGMA_SCALE,
    git_sha,
    load_pair,
    load_residual_diffusion,
    save_panel,
    select_cases,
    uncertainty_row,
)

log = logging.getLogger("evaluate_residual_diffusion_gates")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate residual diffusion release gates.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/diffusion_v2_residual.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--val-cases", type=int, default=0)
    p.add_argument("--test-cases", type=int, default=5)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--sample-sigma-min", type=float, default=None)
    p.add_argument("--sample-sigma-max", type=float, default=None)
    p.add_argument("--stochastic-churn", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--fixed-alphas", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50])
    p.add_argument("--gate-mask-modes", choices=["none", "heart", "boundary"], nargs="+", default=["none", "heart"])
    p.add_argument("--shrinkage-alphas", type=float, nargs="+", default=[1.0])
    p.add_argument("--shrinkage-taus", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05, 0.10])
    p.add_argument("--shrinkage-power", type=float, default=2.0)
    p.add_argument(
        "--oracle-alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.025, 0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 1.0],
    )
    p.add_argument("--oracle-block-size", type=int, default=32)
    p.add_argument("--skip-oracle-voxel", action="store_true")
    p.add_argument("--skip-oracle-block", action="store_true")
    p.add_argument("--clip-output", action="store_true")
    p.add_argument("--save-posterior-arrays", action="store_true")
    p.add_argument("--no-print-summary", action="store_true")
    p.add_argument("--figure-cases", type=int, default=3)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/diffusion_v2_residual/gate_ablation"),
    )
    return p


def _fmt_float(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _mask_for_mode(mode: str, heart_mask: torch.Tensor, boundary_mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mode == "none":
        return torch.ones_like(ref)
    if mode == "heart":
        return heart_mask.to(device=ref.device, dtype=ref.dtype)
    if mode == "boundary":
        return boundary_mask.to(device=ref.device, dtype=ref.dtype)
    raise ValueError(f"unknown gate mask mode: {mode}")


def _maybe_clip(pred: torch.Tensor, clip_output: bool) -> torch.Tensor:
    return pred.clamp(-1.0, 1.0) if clip_output else pred


def compose_prediction(
    initial: torch.Tensor,
    residual_mean: torch.Tensor,
    gate: torch.Tensor | float,
    clip_output: bool,
) -> torch.Tensor:
    pred = initial + gate * residual_mean
    return _maybe_clip(pred, clip_output)


def _scalar_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(masked_mean(values.float(), mask).mean().item())


def add_gate_stats(
    row: dict[str, object],
    gate: torch.Tensor | None,
    heart_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> None:
    if gate is None:
        return
    gate_f = gate.float()
    row["gate_mean"] = float(gate_f.mean().item())
    row["gate_abs_mean"] = float(gate_f.abs().mean().item())
    row["gate_nonzero_frac"] = float((gate_f.abs() > 1e-6).float().mean().item())
    row["gate_heart_mean"] = _scalar_masked_mean(gate_f, heart_mask)
    row["gate_boundary_mean"] = _scalar_masked_mean(gate_f, boundary_mask)


def evaluate_prediction(
    *,
    split_name: str,
    case_id: str,
    strategy: str,
    strategy_group: str,
    pred: torch.Tensor,
    clean: torch.Tensor,
    heart_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    boundary_radius: int,
    unet_metrics: dict[str, float],
    gate: torch.Tensor | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "split": split_name,
        "case_id": case_id,
        "strategy": strategy,
        "strategy_group": strategy_group,
    }
    row.update(metric_row("pred", pred, clean, heart_mask=heart_mask, boundary_radius=boundary_radius))
    row["delta_vs_unet_mae_hu"] = float(row["pred_mae_hu"]) - float(unet_metrics["pred_mae_hu"])
    row["delta_vs_unet_heart_mae_hu"] = (
        float(row["pred_heart_mae_hu"]) - float(unet_metrics["pred_heart_mae_hu"])
    )
    row["delta_vs_unet_boundary_mae_hu"] = (
        float(row["pred_boundary_mae_hu"]) - float(unet_metrics["pred_boundary_mae_hu"])
    )
    row["delta_vs_unet_boundary_gradient_l1_hu"] = (
        float(row["pred_boundary_gradient_l1_hu"])
        - float(unet_metrics["pred_boundary_gradient_l1_hu"])
    )
    add_gate_stats(row, gate, heart_mask, boundary_mask)
    if extra:
        row.update(extra)
    return row


def best_scalar_alpha_prediction(
    initial: torch.Tensor,
    clean: torch.Tensor,
    residual_mean: torch.Tensor,
    alphas: list[float],
    gate_mask: torch.Tensor,
    optimize_mask: torch.Tensor | None,
    clip_output: bool,
) -> tuple[torch.Tensor, float]:
    best_alpha = float(alphas[0])
    best_pred = compose_prediction(initial, residual_mean, best_alpha * gate_mask, clip_output)
    best_err = (best_pred - clean).abs()
    if optimize_mask is not None:
        best_score = _scalar_masked_mean(best_err, optimize_mask)
    else:
        best_score = float(best_err.mean().item())

    for alpha in alphas[1:]:
        pred = compose_prediction(initial, residual_mean, float(alpha) * gate_mask, clip_output)
        err = (pred - clean).abs()
        score = _scalar_masked_mean(err, optimize_mask) if optimize_mask is not None else float(err.mean().item())
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred = pred
    return best_pred, best_alpha


def oracle_voxel_prediction(
    initial: torch.Tensor,
    clean: torch.Tensor,
    residual_mean: torch.Tensor,
    alphas: list[float],
    gate_mask: torch.Tensor,
    clip_output: bool,
) -> torch.Tensor:
    best_pred = initial.clone()
    best_err = (best_pred - clean).abs()
    valid = gate_mask > 0.5
    for alpha in alphas:
        pred = compose_prediction(initial, residual_mean, float(alpha) * gate_mask, clip_output)
        err = (pred - clean).abs()
        better = torch.logical_and(valid, err < best_err)
        best_pred = torch.where(better, pred, best_pred)
        best_err = torch.where(better, err, best_err)
    return best_pred


def _spatial_block_slices(shape: tuple[int, int, int], block_size: int) -> Iterable[tuple[slice, slice, slice]]:
    d_size, h_size, w_size = shape
    for d0 in range(0, d_size, block_size):
        for h0 in range(0, h_size, block_size):
            for w0 in range(0, w_size, block_size):
                yield (
                    slice(d0, min(d0 + block_size, d_size)),
                    slice(h0, min(h0 + block_size, h_size)),
                    slice(w0, min(w0 + block_size, w_size)),
                )


def oracle_block_prediction(
    initial: torch.Tensor,
    clean: torch.Tensor,
    residual_mean: torch.Tensor,
    alphas: list[float],
    gate_mask: torch.Tensor,
    block_size: int,
    clip_output: bool,
) -> torch.Tensor:
    if block_size < 1:
        raise ValueError(f"oracle block size must be >= 1, got {block_size}")
    best = initial.clone()
    spatial = tuple(int(v) for v in initial.shape[-3:])
    for z_slice, y_slice, x_slice in _spatial_block_slices(spatial, block_size):
        block = (slice(None), slice(None), z_slice, y_slice, x_slice)
        init_b = initial[block]
        clean_b = clean[block]
        residual_b = residual_mean[block]
        gate_b = gate_mask[block]
        valid = gate_b > 0.5
        if not bool(valid.any().item()):
            continue
        best_alpha = 0.0
        best_score = float("inf")
        best_block = init_b
        for alpha in alphas:
            pred_b = compose_prediction(init_b, residual_b, float(alpha) * gate_b, clip_output)
            score = float((pred_b - clean_b).abs()[valid].mean().item())
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
                best_block = pred_b
        _ = best_alpha
        best[block] = best_block
    return best


def _numeric_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        }
    )
    out: dict[str, object] = {"n_rows": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if isinstance(row.get(key), (float, int))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(arr.std(ddof=0))
    return out


def _group_summary(rows: list[dict[str, object]], group_key: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for value in sorted({str(row[group_key]) for row in rows}):
        group_rows = [row for row in rows if str(row[group_key]) == value]
        out[value] = _numeric_summary(group_rows)
    return out


def _strategy_ranking(rows: list[dict[str, object]], metric_key: str) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for strategy, block in _group_summary(rows, "strategy").items():
        metric_mean = block.get(f"{metric_key}_mean")
        if metric_mean is None:
            continue
        ranked.append({"strategy": strategy, f"{metric_key}_mean": metric_mean})
    return sorted(ranked, key=lambda row: float(row[f"{metric_key}_mean"]))


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    preferred = [
        "split",
        "case_id",
        "strategy_group",
        "strategy",
        "pred_mae_hu",
        "delta_vs_unet_mae_hu",
        "pred_heart_mae_hu",
        "delta_vs_unet_heart_mae_hu",
        "pred_boundary_mae_hu",
        "delta_vs_unet_boundary_mae_hu",
        "pred_boundary_gradient_l1_hu",
        "delta_vs_unet_boundary_gradient_l1_hu",
    ]
    keys = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in keys]
    fieldnames.extend(sorted(keys.difference(fieldnames)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; gate evaluation is GPU-only by default.")
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")
    if args.n_samples < 2 and args.shrinkage_taus:
        log.warning("n_samples < 2; uncertainty shrinkage rows will be skipped.")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected")
    log.info("selected %d cases: %s", len(selected), selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = args.out_dir / "posterior_arrays"
    if args.save_posterior_arrays:
        arrays_dir.mkdir(parents=True, exist_ok=True)

    initializer = load_initializer(cfg.initializer, device)
    engine, model_meta = load_residual_diffusion(args.config, args.ckpt, args.weights, device)
    if args.sample_sigma_min is not None:
        engine.sigma_min = float(args.sample_sigma_min)
        model_meta["sample_sigma_min_override"] = float(args.sample_sigma_min)
    if args.sample_sigma_max is not None:
        engine.sigma_max = float(args.sample_sigma_max)
        model_meta["sample_sigma_max_override"] = float(args.sample_sigma_max)
    num_steps = int(args.num_steps or model_meta["engine_cfg"].get("num_sample_steps_eval", 32))

    fixed_alphas = _as_float_list(args.fixed_alphas)
    shrinkage_alphas = _as_float_list(args.shrinkage_alphas)
    shrinkage_taus = _as_float_list(args.shrinkage_taus)
    oracle_alphas = _as_float_list(args.oracle_alphas)
    rows: list[dict[str, object]] = []

    for idx, (split_name, case_id) in enumerate(selected):
        clean, corrupted = load_pair(case_id, processed_dir)
        heart_mask = load_heart_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
        boundary_mask = make_boundary_band(heart_mask, radius=args.boundary_radius)

        outputs = sliding_window_residual_diffusion_correct(
            corrupted,
            initializer,
            engine,
            roi_size=args.roi_size,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            sw_device=device,
            output_device="cpu",
            blend_mode=args.blend_mode,
            sigma_scale=args.sigma_scale,
            num_steps=num_steps,
            n_samples=args.n_samples,
            residual_scale=1.0,
            deterministic=not args.stochastic_churn,
            progress=args.progress,
            return_initial=True,
            return_uncertainty=args.n_samples > 1,
        )
        if args.n_samples > 1:
            posterior_mean_pred, initial, residual_std = outputs
        else:
            posterior_mean_pred, initial = outputs
            residual_std = None
        residual_mean = posterior_mean_pred - initial

        if args.save_posterior_arrays:
            np.savez_compressed(
                arrays_dir / f"{split_name}_case_{case_id}.npz",
                initial=initial[0, 0].detach().float().cpu().numpy(),
                residual_mean=residual_mean[0, 0].detach().float().cpu().numpy(),
                residual_std=(
                    residual_std[0, 0].detach().float().cpu().numpy()
                    if residual_std is not None
                    else np.zeros(tuple(initial.shape[-3:]), dtype=np.float32)
                ),
            )

        unet_metrics = metric_row(
            "pred",
            initial,
            clean,
            heart_mask=heart_mask,
            boundary_radius=args.boundary_radius,
        )
        rows.append(
            evaluate_prediction(
                split_name=split_name,
                case_id=case_id,
                strategy="corrupted",
                strategy_group="baseline",
                pred=corrupted,
                clean=clean,
                heart_mask=heart_mask,
                boundary_mask=boundary_mask,
                boundary_radius=args.boundary_radius,
                unet_metrics=unet_metrics,
            )
        )
        rows.append(
            evaluate_prediction(
                split_name=split_name,
                case_id=case_id,
                strategy="unet_init",
                strategy_group="baseline",
                pred=initial,
                clean=clean,
                heart_mask=heart_mask,
                boundary_mask=boundary_mask,
                boundary_radius=args.boundary_radius,
                unet_metrics=unet_metrics,
            )
        )

        posterior_gate = torch.ones_like(residual_mean)
        posterior_pred = _maybe_clip(posterior_mean_pred, args.clip_output)
        posterior_metrics = metric_row(
            "pred",
            posterior_pred,
            clean,
            heart_mask=heart_mask,
            boundary_radius=args.boundary_radius,
        )
        posterior_extra = {
            "residual_mean_abs_hu": float(residual_mean.abs().mean().item() * HU_SCALE),
            "residual_heart_abs_hu": _scalar_masked_mean(residual_mean.abs(), heart_mask) * HU_SCALE,
            "residual_boundary_abs_hu": _scalar_masked_mean(residual_mean.abs(), boundary_mask) * HU_SCALE,
        }
        if residual_std is not None:
            posterior_extra.update(
                {
                    "residual_std_mean_hu": float(residual_std.mean().item() * HU_SCALE),
                    "residual_std_heart_mean_hu": _scalar_masked_mean(residual_std, heart_mask) * HU_SCALE,
                    "residual_std_boundary_mean_hu": _scalar_masked_mean(residual_std, boundary_mask) * HU_SCALE,
                }
            )
        rows.append(
            evaluate_prediction(
                split_name=split_name,
                case_id=case_id,
                strategy="posterior_mean",
                strategy_group="posterior",
                pred=posterior_pred,
                clean=clean,
                heart_mask=heart_mask,
                boundary_mask=boundary_mask,
                boundary_radius=args.boundary_radius,
                unet_metrics=unet_metrics,
                gate=posterior_gate,
                extra=posterior_extra,
            )
        )

        if residual_std is not None:
            rows[-1].update(
                uncertainty_row(
                    posterior_pred,
                    clean,
                    residual_std,
                    heart_mask=heart_mask,
                    boundary_radius=args.boundary_radius,
                )
            )

        for mask_mode in args.gate_mask_modes:
            mask_gate = _mask_for_mode(mask_mode, heart_mask, boundary_mask, residual_mean)
            for alpha in fixed_alphas:
                gate = float(alpha) * mask_gate
                strategy = f"fixed_alpha_{_fmt_float(alpha)}_{mask_mode}"
                rows.append(
                    evaluate_prediction(
                        split_name=split_name,
                        case_id=case_id,
                        strategy=strategy,
                        strategy_group="fixed_alpha",
                        pred=compose_prediction(initial, residual_mean, gate, args.clip_output),
                        clean=clean,
                        heart_mask=heart_mask,
                        boundary_mask=boundary_mask,
                        boundary_radius=args.boundary_radius,
                        unet_metrics=unet_metrics,
                        gate=gate,
                        extra={"alpha": float(alpha), "gate_mask_mode": mask_mode},
                    )
                )

        if residual_std is not None:
            for mask_mode in args.gate_mask_modes:
                mask_gate = _mask_for_mode(mask_mode, heart_mask, boundary_mask, residual_mean)
                for alpha in shrinkage_alphas:
                    for tau in shrinkage_taus:
                        tau_t = torch.as_tensor(float(tau), dtype=residual_std.dtype, device=residual_std.device)
                        shrink = 1.0 / (1.0 + (residual_std / tau_t.clamp_min(1e-8)).pow(float(args.shrinkage_power)))
                        gate = float(alpha) * shrink * mask_gate
                        strategy = (
                            f"shrink_a{_fmt_float(alpha)}_tau{_fmt_float(tau)}_"
                            f"p{_fmt_float(args.shrinkage_power)}_{mask_mode}"
                        )
                        rows.append(
                            evaluate_prediction(
                                split_name=split_name,
                                case_id=case_id,
                                strategy=strategy,
                                strategy_group="uncertainty_shrinkage",
                                pred=compose_prediction(initial, residual_mean, gate, args.clip_output),
                                clean=clean,
                                heart_mask=heart_mask,
                                boundary_mask=boundary_mask,
                                boundary_radius=args.boundary_radius,
                                unet_metrics=unet_metrics,
                                gate=gate,
                                extra={
                                    "alpha": float(alpha),
                                    "tau_norm": float(tau),
                                    "tau_hu": float(tau) * HU_SCALE,
                                    "gate_mask_mode": mask_mode,
                                },
                            )
                        )

        global_gate = torch.ones_like(residual_mean)
        pred_global, best_alpha = best_scalar_alpha_prediction(
            initial,
            clean,
            residual_mean,
            oracle_alphas,
            global_gate,
            optimize_mask=None,
            clip_output=args.clip_output,
        )
        rows.append(
            evaluate_prediction(
                split_name=split_name,
                case_id=case_id,
                strategy="oracle_global_alpha",
                strategy_group="oracle_scalar",
                pred=pred_global,
                clean=clean,
                heart_mask=heart_mask,
                boundary_mask=boundary_mask,
                boundary_radius=args.boundary_radius,
                unet_metrics=unet_metrics,
                gate=best_alpha * global_gate,
                extra={"oracle_best_alpha": best_alpha, "oracle_optimize_region": "global"},
            )
        )

        for mask_mode, optimize_mask in (("heart", heart_mask), ("boundary", boundary_mask)):
            mask_gate = _mask_for_mode(mask_mode, heart_mask, boundary_mask, residual_mean)
            pred_scalar, best_alpha = best_scalar_alpha_prediction(
                initial,
                clean,
                residual_mean,
                oracle_alphas,
                mask_gate,
                optimize_mask=optimize_mask,
                clip_output=args.clip_output,
            )
            rows.append(
                evaluate_prediction(
                    split_name=split_name,
                    case_id=case_id,
                    strategy=f"oracle_{mask_mode}_alpha",
                    strategy_group="oracle_scalar",
                    pred=pred_scalar,
                    clean=clean,
                    heart_mask=heart_mask,
                    boundary_mask=boundary_mask,
                    boundary_radius=args.boundary_radius,
                    unet_metrics=unet_metrics,
                    gate=best_alpha * mask_gate,
                    extra={"oracle_best_alpha": best_alpha, "oracle_optimize_region": mask_mode},
                )
            )

        if not args.skip_oracle_voxel:
            for mask_mode in ("none", "heart", "boundary"):
                mask_gate = _mask_for_mode(mask_mode, heart_mask, boundary_mask, residual_mean)
                pred_voxel = oracle_voxel_prediction(
                    initial,
                    clean,
                    residual_mean,
                    oracle_alphas,
                    mask_gate,
                    clip_output=args.clip_output,
                )
                rows.append(
                    evaluate_prediction(
                        split_name=split_name,
                        case_id=case_id,
                        strategy=f"oracle_voxel_{mask_mode}",
                        strategy_group="oracle_voxel",
                        pred=pred_voxel,
                        clean=clean,
                        heart_mask=heart_mask,
                        boundary_mask=boundary_mask,
                        boundary_radius=args.boundary_radius,
                        unet_metrics=unet_metrics,
                        extra={"oracle_mask_mode": mask_mode},
                    )
                )

        if not args.skip_oracle_block:
            for mask_mode in ("none", "heart", "boundary"):
                mask_gate = _mask_for_mode(mask_mode, heart_mask, boundary_mask, residual_mean)
                pred_block = oracle_block_prediction(
                    initial,
                    clean,
                    residual_mean,
                    oracle_alphas,
                    mask_gate,
                    block_size=int(args.oracle_block_size),
                    clip_output=args.clip_output,
                )
                rows.append(
                    evaluate_prediction(
                        split_name=split_name,
                        case_id=case_id,
                        strategy=f"oracle_block{args.oracle_block_size}_{mask_mode}",
                        strategy_group="oracle_block",
                        pred=pred_block,
                        clean=clean,
                        heart_mask=heart_mask,
                        boundary_mask=boundary_mask,
                        boundary_radius=args.boundary_radius,
                        unet_metrics=unet_metrics,
                        extra={"oracle_mask_mode": mask_mode, "oracle_block_size": float(args.oracle_block_size)},
                    )
                )

        if idx < args.figure_cases:
            save_panel(
                figures_dir / f"{split_name}_case_{case_id}_posterior_mean.png",
                clean,
                corrupted,
                initial,
                posterior_pred,
                residual_std,
            )

        log.info(
            "%s case %s | unet %.2f posterior_mean %.2f rows=%d",
            split_name,
            case_id,
            unet_metrics["pred_mae_hu"],
            posterior_metrics["pred_mae_hu"],
            len(rows),
        )

    metrics_path = args.out_dir / "gate_metrics.csv"
    _write_rows_csv(metrics_path, rows)

    summary = {
        "git_sha": git_sha(),
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "initializer": OmegaConf.to_container(cfg.initializer, resolve=True),
        "model_meta": model_meta,
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "num_steps": num_steps,
        "n_samples": args.n_samples,
        "sample_sigma_min": engine.sigma_min,
        "sample_sigma_max": engine.sigma_max,
        "stochastic_churn": bool(args.stochastic_churn),
        "boundary_radius": args.boundary_radius,
        "fixed_alphas": fixed_alphas,
        "gate_mask_modes": list(args.gate_mask_modes),
        "shrinkage_alphas": shrinkage_alphas,
        "shrinkage_taus": shrinkage_taus,
        "shrinkage_power": float(args.shrinkage_power),
        "oracle_alphas": oracle_alphas,
        "oracle_block_size": int(args.oracle_block_size),
        "clip_output": bool(args.clip_output),
        "case_selection": {
            "val_cases": args.val_cases,
            "test_cases": args.test_cases,
            "case_ids": args.case_ids,
            "selected_cases": [{"split": split_name, "case_id": case_id} for split_name, case_id in selected],
        },
        "summary": {
            "n_rows": len(rows),
            "by_strategy": _group_summary(rows, "strategy"),
            "by_strategy_group": _group_summary(rows, "strategy_group"),
            "rank_pred_mae_hu": _strategy_ranking(rows, "pred_mae_hu")[:20],
            "rank_pred_heart_mae_hu": _strategy_ranking(rows, "pred_heart_mae_hu")[:20],
            "rank_pred_boundary_gradient_l1_hu": _strategy_ranking(rows, "pred_boundary_gradient_l1_hu")[:20],
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", metrics_path)
    log.info("wrote %s", summary_path)
    if not args.no_print_summary:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
