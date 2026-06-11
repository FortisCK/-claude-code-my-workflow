#!/usr/bin/env python
"""Evaluate learned residual-diffusion reliability gates from cached features."""

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
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.evaluation.artifact_metrics import (  # noqa: E402
    assign_artifact_severity,
    metric_row,
    summarize,
)
from code.evaluation.metrics import make_boundary_band, masked_mean  # noqa: E402
from code.models.residual_gate import ResidualGateNet3D  # noqa: E402
from code.training.train_residual_gate import build_gate, make_gate_features  # noqa: E402
from scripts.python.evaluate_residual_diffusion_gates import (  # noqa: E402
    oracle_block_prediction,
    oracle_voxel_prediction,
)

log = logging.getLogger("evaluate_residual_gate_full_volume")

DEFAULT_ROI_SIZE = 128
DEFAULT_OVERLAP = 0.5
DEFAULT_SIGMA_SCALE = 0.125


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate learned GateNet on cached full-volume features.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/residual_gate_v1.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--oracle-alphas", type=float, nargs="+", default=[0.0, 0.025, 0.05, 0.10, 0.15, 0.25, 0.50])
    p.add_argument("--oracle-block-size", type=int, default=32)
    p.add_argument("--fixed-alphas", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--figure-cases", type=int, default=5)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--no-print-summary", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/residual_gate_v1/eval_full_volume"),
    )
    return p


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


def discover_cache_files(
    cache_dir: Path,
    splits: list[str],
    case_ids: Optional[list[str]],
    max_cases: Optional[int],
) -> list[tuple[str, str, Path]]:
    allowed = {str(cid) for cid in case_ids} if case_ids else None
    selected: list[tuple[str, str, Path]] = []
    for split_name in splits:
        for path in sorted((cache_dir / split_name).glob("case_*.npz")):
            case_id = path.stem.removeprefix("case_")
            if allowed is not None and case_id not in allowed:
                continue
            selected.append((split_name, case_id, path))
            if max_cases is not None and len(selected) >= int(max_cases):
                return selected
    return selected


def load_cache(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        out = {
            key: torch.from_numpy(np.asarray(data[key])).unsqueeze(0).unsqueeze(0).float()
            for key in ("clean", "corrupted", "initial", "residual_mean", "residual_std")
        }
        out["heart_mask"] = torch.from_numpy(np.asarray(data["heart_mask"])).unsqueeze(0).unsqueeze(0).bool()
    return out


def load_gate(config_path: Path, ckpt_path: Path, weights: str, device: torch.device) -> tuple[ResidualGateNet3D, dict[str, object]]:
    cfg = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = OmegaConf.create(state.get("model_cfg") or OmegaConf.to_container(cfg.model, resolve=True))
    model = build_gate(model_cfg).to(device)
    if weights == "ema" and state.get("ema_model") is not None:
        model.load_state_dict(state["ema_model"])
        log.info("[gate] loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
        log.info("[gate] loaded online weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    meta = {
        "checkpoint_epoch": state.get("epoch", "unknown"),
        "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        "cache_dir_train": state.get("cache_dir"),
    }
    log.info("[gate] loaded %s @ epoch %s", ckpt_path, meta["checkpoint_epoch"])
    return model, meta


@torch.inference_mode()
def sliding_window_gate(
    features: torch.Tensor,
    model: ResidualGateNet3D,
    roi_size: int,
    overlap: float,
    sw_batch_size: int,
    sw_device: torch.device,
    output_device: torch.device | str,
    blend_mode: str,
    sigma_scale: float,
    progress: bool,
) -> torch.Tensor:
    inferer = SlidingWindowInferer(
        roi_size=(roi_size, roi_size, roi_size),
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=blend_mode,
        sigma_scale=sigma_scale,
        sw_device=sw_device,
        device=torch.device(output_device),
        progress=progress,
        cache_roi_weight_map=True,
    )

    def predictor(patches: torch.Tensor) -> torch.Tensor:
        return model(patches.to(sw_device, non_blocking=True)).float()

    return inferer(features.float(), predictor).float()


def _scalar_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    return float(masked_mean(values.float(), mask).mean().item())


def add_deltas(row: dict[str, object], prefix: str, baseline_prefix: str = "unet_init") -> None:
    for key in ("mae_hu", "heart_mae_hu", "boundary_mae_hu", "boundary_gradient_l1_hu"):
        metric_key = f"{prefix}_{key}"
        baseline_key = f"{baseline_prefix}_{key}"
        if metric_key in row and baseline_key in row:
            row[f"{prefix}_delta_vs_{baseline_prefix}_{key}"] = float(row[metric_key]) - float(row[baseline_key])


def save_panel(
    out_path: Path,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    initial: torch.Tensor,
    posterior: torch.Tensor,
    learned: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = clean.shape[-3] // 2
    panels = [
        ("clean", clean[0, 0, z].float().cpu(), "gray", -1.0, 1.0),
        ("corrupted", corrupted[0, 0, z].float().cpu(), "gray", -1.0, 1.0),
        ("unet init", initial[0, 0, z].float().cpu(), "gray", -1.0, 1.0),
        ("posterior mean", posterior[0, 0, z].float().cpu(), "gray", -1.0, 1.0),
        ("learned gate", learned[0, 0, z].float().cpu(), "gray", -1.0, 1.0),
        ("gate", gate[0, 0, z].float().cpu(), "viridis", 0.0, max(float(gate.max().item()), 1e-6)),
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
        raise RuntimeError("CUDA requested but unavailable; gate eval needs GPU for normal runs.")

    cfg = OmegaConf.load(args.config)
    cache_dir = args.cache_dir or Path(cfg.data.cache_dir)
    selected = discover_cache_files(cache_dir, list(args.splits), args.case_ids, args.max_cases)
    if not selected:
        raise FileNotFoundError(f"no cache files found in {cache_dir} for splits={args.splits}")
    log.info("selected %d cached cases: %s", len(selected), [(s, c) for s, c, _ in selected])

    device = torch.device(args.device)
    model, model_meta = load_gate(args.config, args.ckpt, args.weights, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    oracle_alphas = [float(v) for v in args.oracle_alphas]
    for idx, (split_name, case_id, path) in enumerate(selected):
        data = load_cache(path)
        clean = data["clean"]
        corrupted = data["corrupted"]
        initial = data["initial"]
        residual_mean = data["residual_mean"]
        residual_std = data["residual_std"]
        heart_mask = data["heart_mask"]
        boundary_mask = make_boundary_band(heart_mask, radius=args.boundary_radius)
        posterior = initial + residual_mean

        feature_4d = make_gate_features(
            corrupted[0],
            initial[0],
            residual_mean[0],
            residual_std[0],
        )
        features = feature_4d.unsqueeze(0)
        gate = sliding_window_gate(
            features,
            model,
            roi_size=args.roi_size,
            overlap=args.overlap,
            sw_batch_size=args.sw_batch_size,
            sw_device=device,
            output_device="cpu",
            blend_mode=args.blend_mode,
            sigma_scale=args.sigma_scale,
            progress=args.progress,
        )
        learned = initial + gate * residual_mean

        row: dict[str, object] = {
            "split": split_name,
            "case_id": case_id,
            "cache_path": str(path),
            "shape": "x".join(str(v) for v in clean.shape[-3:]),
            "heart_voxels": float(heart_mask.sum().item()),
        }
        for prefix, pred in (
            ("corrupted", corrupted),
            ("unet_init", initial),
            ("posterior_mean", posterior),
            ("learned_gate", learned),
        ):
            row.update(metric_row(prefix, pred, clean, heart_mask=heart_mask, boundary_radius=args.boundary_radius))
            if prefix != "unet_init":
                add_deltas(row, prefix)

        row["gate_mean"] = float(gate.mean().item())
        row["gate_max"] = float(gate.max().item())
        row["gate_heart_mean"] = _scalar_masked_mean(gate, heart_mask)
        row["gate_boundary_mean"] = _scalar_masked_mean(gate, boundary_mask)
        row["residual_mean_abs_hu"] = float(residual_mean.abs().mean().item() * 2047.5)
        row["residual_std_mean_hu"] = float(residual_std.mean().item() * 2047.5)

        for alpha in args.fixed_alphas:
            prefix = f"fixed_alpha_{str(alpha).replace('.', 'p')}"
            pred = initial + float(alpha) * residual_mean
            row.update(metric_row(prefix, pred, clean, heart_mask=heart_mask, boundary_radius=args.boundary_radius))
            add_deltas(row, prefix)

        ones = torch.ones_like(residual_mean)
        oracle_voxel = oracle_voxel_prediction(
            initial,
            clean,
            residual_mean,
            oracle_alphas,
            ones,
            clip_output=False,
        )
        row.update(metric_row("oracle_voxel", oracle_voxel, clean, heart_mask=heart_mask, boundary_radius=args.boundary_radius))
        add_deltas(row, "oracle_voxel")

        oracle_block = oracle_block_prediction(
            initial,
            clean,
            residual_mean,
            oracle_alphas,
            ones,
            block_size=int(args.oracle_block_size),
            clip_output=False,
        )
        row.update(metric_row("oracle_block", oracle_block, clean, heart_mask=heart_mask, boundary_radius=args.boundary_radius))
        add_deltas(row, "oracle_block")

        rows.append(row)
        log.info(
            "%s case %s | unet %.2f learned %.2f delta %.2f gate %.4f",
            split_name,
            case_id,
            row["unet_init_mae_hu"],
            row["learned_gate_mae_hu"],
            row["learned_gate_delta_vs_unet_init_mae_hu"],
            row["gate_mean"],
        )
        if idx < args.figure_cases:
            save_panel(
                figures_dir / f"{split_name}_case_{case_id}.png",
                clean,
                corrupted,
                initial,
                posterior,
                learned,
                gate,
            )

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
        "model_meta": model_meta,
        "cache_dir": str(cache_dir),
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "oracle_alphas": oracle_alphas,
        "oracle_block_size": int(args.oracle_block_size),
        "fixed_alphas": [float(v) for v in args.fixed_alphas],
        "case_selection": {
            "splits": list(args.splits),
            "case_ids": args.case_ids,
            "max_cases": args.max_cases,
            "selected_cases": [{"split": split_name, "case_id": case_id} for split_name, case_id, _ in selected],
        },
        "summary": summarize(rows),
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
