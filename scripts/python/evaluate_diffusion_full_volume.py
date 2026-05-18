#!/usr/bin/env python
"""Full-volume evaluation for Diffusion v1 sliding-window correction.

This evaluates trained patch-level conditional EDM checkpoints on full 192^3
ImageCAS volumes by applying 128^3 overlapping sliding-window inference.

Usage:
    python scripts/python/evaluate_diffusion_full_volume.py \
        --case-ids 9 21 \
        --eval-mode both \
        --num-steps 50 \
        --stochastic-samples 4 \
        --out-dir experiments/runs/diffusion_v1/eval_epoch200_full_volume

Outputs:
    - `metrics.csv`: per-case corrupted, deterministic, and/or stochastic metrics
    - `summary.json`: aggregate metrics and run metadata
    - `figures/*.png`: center-slice qualitative panels for the first K cases
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
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

from code.data import paths as path_registry  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.evaluation.metrics import all_metrics  # noqa: E402
from code.evaluation.run_eval import load_frozen_edm, load_pair  # noqa: E402
from code.inference.sliding_window import sliding_window_posterior_sample  # noqa: E402
from code.training.train_diffusion import load_frozen_vae  # noqa: E402

log = logging.getLogger("evaluate_diffusion_full_volume")

HU_SCALE = 2047.5  # [-1, 1] maps to [-1024, 3071].
DEFAULT_ROI_SIZE = 128
DEFAULT_OVERLAP = 0.5
DEFAULT_SIGMA_SCALE = 0.125


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Diffusion v1 on full 192^3 volumes.")
    p.add_argument("--vae-config", type=Path, default=Path("code/training/configs/vae_v2_128.yaml"))
    p.add_argument("--vae-ckpt", type=Path, default=Path("experiments/checkpoints/vae_v2/best_val.pt"))
    p.add_argument("--diff-config", type=Path, default=Path("code/training/configs/diffusion_v1.yaml"))
    p.add_argument("--diff-ckpt", type=Path, default=Path("experiments/checkpoints/diffusion_v1/epoch_200.pt"))
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--val-cases", type=int, default=1)
    p.add_argument("--test-cases", type=int, default=1)
    p.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Explicit case IDs. When set, all are labeled split='manual'.",
    )
    p.add_argument("--eval-mode", choices=["deterministic", "stochastic", "both"], default="both")
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--stochastic-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/diffusion_v1/eval_epoch200_full_volume"),
    )
    p.add_argument("--figure-cases", type=int, default=5)
    p.add_argument("--progress", action="store_true", help="Show MONAI sliding-window progress bars.")
    return p


def select_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.case_ids:
        return [("manual", str(cid)) for cid in args.case_ids]
    split = load_split(args.split_file)
    selected: list[tuple[str, str]] = []
    selected.extend(("val", str(cid)) for cid in split.val[: args.val_cases])
    selected.extend(("test", str(cid)) for cid in split.test[: args.test_cases])
    return selected


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def metric_row(prefix: str, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = all_metrics(pred.float(), target.float())
    abs_err = (pred.float() - target.float()).abs()
    return {
        f"{prefix}_mae_norm": float(abs_err.mean().item()),
        f"{prefix}_mae_hu": float(abs_err.mean().item() * HU_SCALE),
        f"{prefix}_psnr": float(metrics["psnr"].item()),
        f"{prefix}_ssim": float(metrics["ssim"].item()),
        f"{prefix}_nrmse": float(metrics["nrmse"].item()),
        f"{prefix}_dice_lumen_stub": float(metrics["dice_lumen_stub"].item()),
    }


def save_panel(
    out_path: Path,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    predictions: dict[str, torch.Tensor],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_v = clean[0, 0].detach().float().cpu()
    corr_v = corrupted[0, 0].detach().float().cpu()
    z = clean_v.shape[0] // 2

    panels: list[tuple[str, torch.Tensor, str, float, Optional[float]]] = [
        ("clean", clean_v[z], "gray", -1.0, 1.0),
        ("corrupted", corr_v[z], "gray", -1.0, 1.0),
    ]
    if "det" in predictions:
        det_v = predictions["det"][0, 0].detach().float().cpu()
        panels.append(("det", det_v[z], "gray", -1.0, 1.0))
    if "stoch_mean" in predictions:
        stoch_v = predictions["stoch_mean"][0, 0].detach().float().cpu()
        panels.append(("stoch_mean", stoch_v[z], "gray", -1.0, 1.0))

    err_source = predictions["det"] if "det" in predictions else predictions.get("stoch_mean")
    if err_source is not None:
        err_v = (err_source - clean).abs()[0, 0].detach().float().cpu()
        panels.append(("abs_err", err_v[z], "magma", 0.0, 0.4))
    if "stoch_std" in predictions:
        std_v = predictions["stoch_std"][0, 0].detach().float().cpu()
        panels.append(("posterior_std", std_v[z], "hot", 0.0, None))

    n_cols = 3
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = np.asarray(axs).reshape(-1)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    out: dict[str, object] = {"n_cases": len(rows), "by_split": {}}
    for split_name in sorted({str(r["split"]) for r in rows}):
        split_rows = [r for r in rows if r["split"] == split_name]
        block: dict[str, object] = {"n_cases": len(split_rows)}
        for key in numeric_keys:
            vals = np.asarray([float(r[key]) for r in split_rows], dtype=np.float64)
            block[f"{key}_mean"] = float(vals.mean())
            block[f"{key}_std"] = float(vals.std(ddof=0))
        out["by_split"][split_name] = block

    all_block: dict[str, object] = {"n_cases": len(rows)}
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        all_block[f"{key}_mean"] = float(vals.mean())
        all_block[f"{key}_std"] = float(vals.std(ddof=0))
    out["all"] = all_block
    return out


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; full-volume diffusion eval is GPU-only.")
    device = torch.device(args.device)
    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected")
    log.info("selected %d cases: %s", len(selected), selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    vae = load_frozen_vae(args.vae_config, args.vae_ckpt, device)
    edm, _ = load_frozen_edm(args.diff_config, args.diff_ckpt, device)

    run_det = args.eval_mode in {"deterministic", "both"}
    run_stoch = args.eval_mode in {"stochastic", "both"}
    det_prefix = f"det{args.num_steps}_sw"
    stoch_prefix = f"stoch{args.stochastic_samples}_mean{args.num_steps}_sw"

    rows: list[dict[str, object]] = []
    for idx, (split_name, case_id) in enumerate(selected):
        clean, corrupted = load_pair(case_id, processed_dir, pair_mode="precomputed")
        row: dict[str, object] = {
            "split": split_name,
            "case_id": case_id,
            "shape": "x".join(str(v) for v in clean.shape[-3:]),
        }
        row.update(metric_row("corrupted", corrupted, clean))
        predictions: dict[str, torch.Tensor] = {}

        if run_det:
            det_out = sliding_window_posterior_sample(
                corrupted,
                vae,
                edm,
                roi_size=args.roi_size,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size,
                n_samples=1,
                num_steps=args.num_steps,
                deterministic=True,
                sw_device=device,
                output_device="cpu",
                blend_mode=args.blend_mode,
                sigma_scale=args.sigma_scale,
                progress=args.progress,
            )
            det = det_out["mean"].float()
            predictions["det"] = det
            row.update(metric_row(det_prefix, det, clean))
            row[f"{det_prefix}_delta_mae_norm"] = (
                float(row[f"{det_prefix}_mae_norm"]) - float(row["corrupted_mae_norm"])
            )
            row[f"{det_prefix}_min"] = float(det.min().item())
            row[f"{det_prefix}_max"] = float(det.max().item())
            log.info(
                "%s case %s | corr MAE %.5f det %.5f",
                split_name,
                case_id,
                row["corrupted_mae_norm"],
                row[f"{det_prefix}_mae_norm"],
            )

        if run_stoch:
            stoch_out = sliding_window_posterior_sample(
                corrupted,
                vae,
                edm,
                roi_size=args.roi_size,
                overlap=args.overlap,
                sw_batch_size=args.sw_batch_size,
                n_samples=args.stochastic_samples,
                num_steps=args.num_steps,
                deterministic=False,
                sw_device=device,
                output_device="cpu",
                blend_mode=args.blend_mode,
                sigma_scale=args.sigma_scale,
                progress=args.progress,
            )
            stoch_mean = stoch_out["mean"].float()
            stoch_std = stoch_out["std"].float()
            predictions["stoch_mean"] = stoch_mean
            predictions["stoch_std"] = stoch_std
            row.update(metric_row(stoch_prefix, stoch_mean, clean))
            row[f"{stoch_prefix}_delta_mae_norm"] = (
                float(row[f"{stoch_prefix}_mae_norm"]) - float(row["corrupted_mae_norm"])
            )
            row[f"stoch{args.stochastic_samples}_sw_std_mean"] = float(stoch_std.mean().item())
            row[f"{stoch_prefix}_min"] = float(stoch_mean.min().item())
            row[f"{stoch_prefix}_max"] = float(stoch_mean.max().item())
            log.info(
                "%s case %s | corr MAE %.5f stoch %.5f std %.5f",
                split_name,
                case_id,
                row["corrupted_mae_norm"],
                row[f"{stoch_prefix}_mae_norm"],
                row[f"stoch{args.stochastic_samples}_sw_std_mean"],
            )

        rows.append(row)
        if idx < args.figure_cases:
            save_panel(figures_dir / f"{split_name}_case_{case_id}.png", clean, corrupted, predictions)

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "git_sha": git_sha(),
        "checkpoint": str(args.diff_ckpt),
        "vae_checkpoint": str(args.vae_ckpt),
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "case_selection": {
            "val_cases": args.val_cases,
            "test_cases": args.test_cases,
            "case_ids": args.case_ids,
        },
        "sampling": {
            "eval_mode": args.eval_mode,
            "num_steps": args.num_steps,
            "stochastic_samples": args.stochastic_samples,
            "seed": args.seed,
        },
        "summary": summarize(rows),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("wrote %s", metrics_path)
    log.info("wrote %s", summary_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
