#!/usr/bin/env python
"""Patch-level evaluation for a trained conditional latent EDM.

This evaluates the 128^3 patch regime used by Diffusion v1 before attempting
full-volume sliding-window inference. For each selected case it compares:

    clean patch vs corrupted patch
    clean patch vs deterministic EDM prediction
    clean patch vs stochastic posterior mean

Outputs CSV/JSON summaries plus center-slice visual panels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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
from code.inference.posterior_sample import posterior_sample  # noqa: E402
from code.training.train_diffusion import load_frozen_vae  # noqa: E402

log = logging.getLogger("evaluate_diffusion_patches")

HU_SCALE = 2047.5  # [-1, 1] maps to [-1024, 3071].


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Diffusion v1 on 128^3 patches.")
    p.add_argument("--vae-config", type=Path, default=Path("code/training/configs/vae_v2_128.yaml"))
    p.add_argument("--vae-ckpt", type=Path, default=Path("experiments/checkpoints/vae_v2/best_val.pt"))
    p.add_argument("--diff-config", type=Path, default=Path("code/training/configs/diffusion_v1.yaml"))
    p.add_argument("--diff-ckpt", type=Path, default=Path("experiments/checkpoints/diffusion_v1/epoch_200.pt"))
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--val-cases", type=int, default=5)
    p.add_argument("--test-cases", type=int, default=5)
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Explicit case IDs. When set, all are labeled split='manual'.")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--stochastic-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiments/runs/diffusion_v1/eval_epoch200_patches"))
    p.add_argument("--figure-cases", type=int, default=10)
    return p


def center_crop(v: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop a `(B, C, D, H, W)` tensor to `size^3`."""
    _, _, d, h, w = v.shape
    if min(d, h, w) < size:
        raise ValueError(f"cannot crop {tuple(v.shape)} to {size}^3")
    z0 = (d - size) // 2
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return v[:, :, z0:z0 + size, y0:y0 + size, x0:x0 + size].contiguous()


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
    det: torch.Tensor,
    stoch_mean: torch.Tensor,
    stoch_std: torch.Tensor,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_v = clean[0, 0].detach().float().cpu()
    corr_v = corrupted[0, 0].detach().float().cpu()
    det_v = det[0, 0].detach().float().cpu()
    stoch_v = stoch_mean[0, 0].detach().float().cpu()
    err_v = (det - clean).abs()[0, 0].detach().float().cpu()
    std_v = stoch_std[0, 0].detach().float().cpu()

    z = clean_v.shape[0] // 2
    panels = [
        ("clean", clean_v[z], "gray", -1.0, 1.0),
        ("corrupted", corr_v[z], "gray", -1.0, 1.0),
        ("det50", det_v[z], "gray", -1.0, 1.0),
        ("stoch_mean", stoch_v[z], "gray", -1.0, 1.0),
        ("abs_err_det", err_v[z], "magma", 0.0, 0.4),
        ("posterior_std", std_v[z], "hot", 0.0, None),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(axs.ravel(), panels):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def select_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.case_ids:
        return [("manual", str(cid)) for cid in args.case_ids]
    split = load_split(args.split_file)
    selected: list[tuple[str, str]] = []
    selected.extend(("val", str(cid)) for cid in split.val[: args.val_cases])
    selected.extend(("test", str(cid)) for cid in split.test[: args.test_cases])
    return selected


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
        raise RuntimeError("CUDA requested but unavailable; patch diffusion eval is GPU-only.")
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

    rows: list[dict[str, object]] = []
    for idx, (split_name, case_id) in enumerate(selected):
        clean_full, corr_full = load_pair(case_id, processed_dir, pair_mode="precomputed")
        clean = center_crop(clean_full, args.patch_size).to(device)
        corrupted = center_crop(corr_full, args.patch_size).to(device)

        det_out = posterior_sample(
            corrupted, vae, edm,
            n_samples=1, num_steps=args.num_steps,
            deterministic=True, seed=args.seed,
        )
        det = det_out["mean"].float()
        stoch_out = posterior_sample(
            corrupted, vae, edm,
            n_samples=args.stochastic_samples, num_steps=args.num_steps,
            deterministic=False, seed=args.seed,
        )
        stoch_mean = stoch_out["mean"].float()
        stoch_std = stoch_out["std"].float()

        row: dict[str, object] = {"split": split_name, "case_id": case_id}
        row.update(metric_row("corrupted", corrupted, clean))
        row.update(metric_row(f"det{args.num_steps}", det, clean))
        row.update(metric_row(f"stoch{args.stochastic_samples}_mean{args.num_steps}", stoch_mean, clean))

        det_prefix = f"det{args.num_steps}"
        stoch_prefix = f"stoch{args.stochastic_samples}_mean{args.num_steps}"
        row[f"{det_prefix}_delta_mae_norm"] = (
            float(row[f"{det_prefix}_mae_norm"]) - float(row["corrupted_mae_norm"])
        )
        row[f"{stoch_prefix}_delta_mae_norm"] = (
            float(row[f"{stoch_prefix}_mae_norm"]) - float(row["corrupted_mae_norm"])
        )
        row[f"stoch{args.stochastic_samples}_std_mean"] = float(stoch_std.mean().item())
        row[f"{det_prefix}_min"] = float(det.min().item())
        row[f"{det_prefix}_max"] = float(det.max().item())
        row[f"{stoch_prefix}_min"] = float(stoch_mean.min().item())
        row[f"{stoch_prefix}_max"] = float(stoch_mean.max().item())
        rows.append(row)

        log.info(
            "%s case %s | corr MAE %.5f det %.5f stoch %.5f std %.5f",
            split_name,
            case_id,
            row["corrupted_mae_norm"],
            row[f"{det_prefix}_mae_norm"],
            row[f"{stoch_prefix}_mae_norm"],
            row[f"stoch{args.stochastic_samples}_std_mean"],
        )
        if idx < args.figure_cases:
            save_panel(
                figures_dir / f"{split_name}_case_{case_id}.png",
                clean, corrupted, det, stoch_mean, stoch_std,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.diff_ckpt),
        "vae_checkpoint": str(args.vae_ckpt),
        "patch_size": args.patch_size,
        "case_selection": {
            "val_cases": args.val_cases,
            "test_cases": args.test_cases,
            "case_ids": args.case_ids,
        },
        "sampling": {
            "num_steps": args.num_steps,
            "stochastic_samples": args.stochastic_samples,
            "deterministic_seed": args.seed,
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
