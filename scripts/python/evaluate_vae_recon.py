#!/usr/bin/env python
"""Evaluate a frozen cardiac VAE reconstruction checkpoint.

This is a lightweight verification entry point for Stage-1 VAE checkpoints.
It computes per-case reconstruction metrics on fixed split cases and writes
CSV/JSON summaries plus optional clean/recon/error slice figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers.inferer import SlidingWindowInferer
from omegaconf import OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.data.imagecas_dataset import ImageCASCleanDataset  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.data.transforms import vae_v2_val_transforms  # noqa: E402
from code.evaluation.metrics import all_metrics  # noqa: E402
from code.models.vae import CardiacVAE  # noqa: E402
from code.training.train_vae_v2 import dynamic_infer, kl_loss  # noqa: E402

log = logging.getLogger("evaluate_vae_recon")

HU_SCALE = 2047.5  # [-1, 1] maps to [-1024, 3071].


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate VAE reconstruction quality.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/vae_v2_128.yaml"))
    p.add_argument("--ckpt", type=Path, default=Path("experiments/checkpoints/vae_v2/best_val.pt"))
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--max-cases-per-split", type=int, default=8)
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Explicit case IDs. When set, --splits is ignored.")
    p.add_argument("--out-dir", type=Path, default=Path("experiments/runs/vae_v2/verify_2026-05-17"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast during inference.")
    p.add_argument("--clamp", action="store_true",
                   help="Clamp reconstructions to [-1, 1] before metrics and figures.")
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--figure-cases", type=int, default=4)
    return p


def _load_vae(cfg_path: Path, ckpt_path: Path, device: torch.device) -> tuple[CardiacVAE, dict]:
    cfg = OmegaConf.load(cfg_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    model_cfg = payload.get("model_cfg") or OmegaConf.to_container(cfg.model, resolve=True)
    vae = CardiacVAE(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=tuple(model_cfg["channels"]),
        num_res_blocks=tuple(model_cfg["num_res_blocks"]),
        attention_levels=tuple(model_cfg["attention_levels"]),
        latent_channels=int(model_cfg["latent_channels"]),
        norm_num_groups=int(model_cfg["norm_num_groups"]),
        use_checkpoint=bool(model_cfg.get("use_checkpoint", False)),
    )
    vae.load_state_dict(payload["model"])
    vae.to(device)
    vae.eval()
    return vae, {
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_val_recon": payload.get("val_recon"),
        "checkpoint_val_metrics": payload.get("val_metrics"),
        "model_cfg": model_cfg,
    }


def _select_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.case_ids:
        return [("manual", str(cid)) for cid in args.case_ids]
    split = load_split(args.split_file)
    selected: list[tuple[str, str]] = []
    for split_name in args.splits:
        ids = list(getattr(split, split_name))
        if args.max_cases_per_split > 0:
            ids = ids[: args.max_cases_per_split]
        selected.extend((split_name, str(cid)) for cid in ids)
    return selected


def _volume_center_slices(volume: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = volume.detach().float().cpu().numpy()
    z, y, x = (d // 2 for d in arr.shape)
    return arr[z, :, :], arr[:, y, :], arr[:, :, x]


def _save_figure(
    case_id: str,
    split_name: str,
    clean: torch.Tensor,
    recon: torch.Tensor,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional visual dependency
        log.warning("matplotlib unavailable, skipping figures: %s", exc)
        return

    clean_slices = _volume_center_slices(clean)
    recon_slices = _volume_center_slices(recon)
    err_slices = _volume_center_slices((recon - clean).abs())

    fig, axes = plt.subplots(3, 3, figsize=(9, 8), constrained_layout=True)
    titles = ["axial", "coronal", "sagittal"]
    for col, title in enumerate(titles):
        axes[0, col].imshow(clean_slices[col], cmap="gray", vmin=-1.0, vmax=1.0)
        axes[0, col].set_title(f"clean {title}")
        axes[1, col].imshow(recon_slices[col], cmap="gray", vmin=-1.0, vmax=1.0)
        axes[1, col].set_title(f"recon {title}")
        axes[2, col].imshow(err_slices[col], cmap="magma", vmin=0.0, vmax=0.2)
        axes[2, col].set_title(f"abs err {title}")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{split_name} case {case_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_keys = [
        "mae_norm",
        "mae_hu",
        "heart_mae_norm",
        "heart_mae_hu",
        "psnr",
        "ssim",
        "nrmse",
        "dice_lumen_stub",
        "kl",
    ]
    summary: dict[str, object] = {"n_cases": len(rows), "by_split": {}}
    for split_name in sorted({str(r["split"]) for r in rows}):
        split_rows = [r for r in rows if r["split"] == split_name]
        block: dict[str, float] = {"n_cases": float(len(split_rows))}
        for key in metric_keys:
            vals = np.asarray([float(r[key]) for r in split_rows], dtype=np.float64)
            block[f"{key}_mean"] = float(vals.mean())
            block[f"{key}_std"] = float(vals.std(ddof=0))
        summary["by_split"][split_name] = block
    return summary


@torch.no_grad()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    selected = _select_cases(args)
    if not selected:
        raise ValueError("no cases selected")

    vae, ckpt_info = _load_vae(args.config, args.ckpt, device)
    cfg = OmegaConf.load(args.config)
    roi_size = tuple(int(v) for v in cfg.val.sliding_window_patch_size)
    inferer = SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        mode="constant",
    )
    use_amp = bool(args.amp and device.type == "cuda")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"

    rows: list[dict[str, object]] = []
    transform = vae_v2_val_transforms()
    for idx, (split_name, case_id) in enumerate(selected):
        ds = ImageCASCleanDataset([case_id], processed_dir=processed_dir, transform=transform)
        batch = ds[0]
        clean = batch["volume"].unsqueeze(0).to(device)
        heart_mask = batch["heart_mask"].unsqueeze(0).to(device).bool()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            recon, z_mu, z_sigma = dynamic_infer(inferer, vae, clean)
        if args.clamp:
            recon = recon.clamp(-1.0, 1.0)

        metric_bundle = all_metrics(recon.float(), clean.float())
        abs_err = (recon.float() - clean.float()).abs()
        heart_abs_err = abs_err[heart_mask]
        row = {
            "split": split_name,
            "case_id": case_id,
            "mae_norm": float(abs_err.mean().item()),
            "mae_hu": float(abs_err.mean().item() * HU_SCALE),
            "heart_mae_norm": float(heart_abs_err.mean().item()) if heart_abs_err.numel() else math.nan,
            "heart_mae_hu": float(heart_abs_err.mean().item() * HU_SCALE) if heart_abs_err.numel() else math.nan,
            "psnr": float(metric_bundle["psnr"].mean().item()),
            "ssim": float(metric_bundle["ssim"].mean().item()),
            "nrmse": float(metric_bundle["nrmse"].mean().item()),
            "dice_lumen_stub": float(metric_bundle["dice_lumen_stub"].mean().item()),
            "kl": float(kl_loss(z_mu.float(), z_sigma.float()).item()),
        }
        rows.append(row)
        log.info(
            "%s case %s | MAE=%.5f (%.1f HU) heart=%.5f (%.1f HU) PSNR=%.2f SSIM=%.4f",
            split_name,
            case_id,
            row["mae_norm"],
            row["mae_hu"],
            row["heart_mae_norm"],
            row["heart_mae_hu"],
            row["psnr"],
            row["ssim"],
        )

        if idx < args.figure_cases:
            _save_figure(
                case_id,
                split_name,
                clean[0, 0],
                recon[0, 0],
                figures_dir / f"{split_name}_case_{case_id}.png",
            )

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": str(args.config),
        "checkpoint": str(args.ckpt),
        "checkpoint_info": ckpt_info,
        "processed_dir": str(processed_dir),
        "split_file": str(args.split_file),
        "roi_size": roi_size,
        "amp": use_amp,
        "clamp": bool(args.clamp),
        **_summarize(rows),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    log.info("wrote %s and %s", metrics_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
