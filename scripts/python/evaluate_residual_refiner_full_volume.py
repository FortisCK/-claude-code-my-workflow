#!/usr/bin/env python
"""Full-volume evaluation for U-Net-initialized residual refiners."""

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
from omegaconf import OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.evaluation.artifact_metrics import (  # noqa: E402
    assign_artifact_severity,
    load_heart_mask,
    metric_row,
    summarize,
)
from code.inference.residual_refiner_sliding_window import sliding_window_refiner_correct  # noqa: E402
from code.models.residual_refiner import ResidualRefinerUNet3D  # noqa: E402
from code.training.train_residual_refiner import load_initializer  # noqa: E402

log = logging.getLogger("evaluate_residual_refiner_full_volume")

DEFAULT_ROI_SIZE = 128
DEFAULT_OVERLAP = 0.5
DEFAULT_SIGMA_SCALE = 0.125


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate deterministic residual refiner on full 192^3 volumes.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/residual_refiner_v1.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--val-cases", type=int, default=1)
    p.add_argument("--test-cases", type=int, default=1)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--figure-cases", type=int, default=5)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/residual_refiner_v1/eval_full_volume"),
    )
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


def load_pair(case_id: str, processed_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = sorted(processed_dir.glob(f"case_{case_id}__pair_*.npz"))
    if not pairs:
        raise FileNotFoundError(f"no precomputed pair for case {case_id} in {processed_dir}")
    with np.load(pairs[0], allow_pickle=True) as data:
        clean = torch.from_numpy(data["volume"]).float()
        corrupted = torch.from_numpy(data["corrupted"]).float()
    return clean.unsqueeze(0).unsqueeze(0), corrupted.unsqueeze(0).unsqueeze(0)


def load_refiner(config_path: Path, ckpt_path: Path, weights: str, device: torch.device) -> ResidualRefinerUNet3D:
    cfg = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = state.get("model_cfg") or OmegaConf.to_container(cfg.model, resolve=True)
    model = ResidualRefinerUNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=tuple(model_cfg["features"]),
        clamp_output=model_cfg.get("clamp_output", False),
    ).to(device)
    if weights == "ema" and state.get("ema_model") is not None:
        model.load_state_dict(state["ema_model"])
        log.info("[refiner] loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
        log.info("[refiner] loaded online weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("[refiner] loaded %s @ epoch %s", ckpt_path, state.get("epoch", "unknown"))
    return model


def save_panel(
    out_path: Path,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    initial: torch.Tensor,
    pred: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_v = clean[0, 0].detach().float().cpu()
    corr_v = corrupted[0, 0].detach().float().cpu()
    init_v = initial[0, 0].detach().float().cpu()
    pred_v = pred[0, 0].detach().float().cpu()
    init_err_v = (initial - clean).abs()[0, 0].detach().float().cpu()
    pred_err_v = (pred - clean).abs()[0, 0].detach().float().cpu()
    z = clean_v.shape[0] // 2
    panels = [
        ("clean", clean_v[z], "gray", -1.0, 1.0),
        ("corrupted", corr_v[z], "gray", -1.0, 1.0),
        ("unet init", init_v[z], "gray", -1.0, 1.0),
        ("refiner", pred_v[z], "gray", -1.0, 1.0),
        ("unet abs err", init_err_v[z], "magma", 0.0, 0.4),
        ("refiner abs err", pred_err_v[z], "magma", 0.0, 0.4),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(np.asarray(axs).reshape(-1), panels):
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; full-volume refiner eval is GPU-only.")
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

    initializer = load_initializer(cfg.initializer, device)
    refiner = load_refiner(args.config, args.ckpt, args.weights, device)
    rows: list[dict[str, object]] = []

    for idx, (split_name, case_id) in enumerate(selected):
        clean, corrupted = load_pair(case_id, processed_dir)
        heart_mask = load_heart_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
        row: dict[str, object] = {
            "split": split_name,
            "case_id": case_id,
            "shape": "x".join(str(v) for v in clean.shape[-3:]),
            "heart_voxels": float(heart_mask.sum().item()),
        }
        row.update(
            metric_row(
                "corrupted",
                corrupted,
                clean,
                heart_mask=heart_mask,
                boundary_radius=args.boundary_radius,
            )
        )

        pred, initial = sliding_window_refiner_correct(
            corrupted,
            initializer,
            refiner,
            roi_size=args.roi_size,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            sw_device=device,
            output_device="cpu",
            blend_mode=args.blend_mode,
            sigma_scale=args.sigma_scale,
            progress=args.progress,
            return_initial=True,
        )
        row.update(
            metric_row(
                "unet_init",
                initial,
                clean,
                heart_mask=heart_mask,
                boundary_radius=args.boundary_radius,
            )
        )
        row.update(
            metric_row(
                "refiner_sw",
                pred,
                clean,
                heart_mask=heart_mask,
                boundary_radius=args.boundary_radius,
            )
        )
        row["refiner_delta_vs_unet_mae_hu"] = (
            float(row["refiner_sw_mae_hu"]) - float(row["unet_init_mae_hu"])
        )
        row["refiner_delta_vs_unet_heart_mae_hu"] = (
            float(row["refiner_sw_heart_mae_hu"]) - float(row["unet_init_heart_mae_hu"])
        )
        row["refiner_delta_vs_unet_boundary_mae_hu"] = (
            float(row["refiner_sw_boundary_mae_hu"]) - float(row["unet_init_boundary_mae_hu"])
        )
        row["refiner_sw_min"] = float(pred.min().item())
        row["refiner_sw_max"] = float(pred.max().item())
        rows.append(row)
        log.info(
            "%s case %s | unet MAE %.2f refiner %.2f delta %.2f",
            split_name,
            case_id,
            row["unet_init_mae_hu"],
            row["refiner_sw_mae_hu"],
            row["refiner_delta_vs_unet_mae_hu"],
        )

        if idx < args.figure_cases:
            save_panel(figures_dir / f"{split_name}_case_{case_id}.png", clean, corrupted, initial, pred)

    assign_artifact_severity(rows)

    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "git_sha": git_sha(),
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "initializer": OmegaConf.to_container(cfg.initializer, resolve=True),
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "boundary_radius": args.boundary_radius,
        "case_selection": {
            "val_cases": args.val_cases,
            "test_cases": args.test_cases,
            "case_ids": args.case_ids,
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
