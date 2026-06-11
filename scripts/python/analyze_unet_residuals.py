#!/usr/bin/env python
"""Diagnose residual errors left by the supervised U-Net baseline.

This script answers whether the frozen deterministic U-Net leaves structured,
localized residuals that are worth modeling with a posterior residual diffusion
model. It re-runs full-volume sliding-window U-Net inference, then measures the
signed residual

    r = clean - unet(corrupted)

globally, inside the heart mask, around the heart boundary, and in top-error
voxels.
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
from omegaconf import OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.evaluation.artifact_metrics import (  # noqa: E402
    HU_SCALE,
    assign_artifact_severity,
    load_heart_mask,
    summarize,
)
from code.evaluation.metrics import make_boundary_band  # noqa: E402
from code.inference.unet_sliding_window import sliding_window_unet_correct  # noqa: E402
from code.models.residual_unet import ResidualUNet3D  # noqa: E402

log = logging.getLogger("analyze_unet_residuals")

DEFAULT_CKPT = Path("experiments/checkpoints/unet_v1/epoch_200.pt")
DEFAULT_ROI_SIZE = 128
DEFAULT_OVERLAP = 0.5
DEFAULT_SIGMA_SCALE = 0.125


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze residual errors after frozen U-Net correction.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/unet_v1.yaml"))
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--val-cases", type=int, default=0)
    p.add_argument("--test-cases", type=int, default=10)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--top-fracs", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05])
    p.add_argument("--figure-cases", type=int, default=5)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/unet_v1/residual_diagnosis"),
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


def load_model(config_path: Path, ckpt_path: Path, weights: str, device: torch.device) -> ResidualUNet3D:
    cfg = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = state.get("model_cfg") or OmegaConf.to_container(cfg.model, resolve=True)
    model = ResidualUNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=tuple(model_cfg["features"]),
        residual=model_cfg["residual"],
        clamp_output=model_cfg.get("clamp_output", False),
    ).to(device)
    if weights == "ema" and state.get("ema_model") is not None:
        model.load_state_dict(state["ema_model"])
        log.info("[model] loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
        log.info("[model] loaded online weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("[model] loaded %s @ epoch %s", ckpt_path, state.get("epoch", "unknown"))
    return model


def _frac_label(frac: float) -> str:
    pct = frac * 100.0
    if pct >= 1.0:
        text = f"{pct:.0f}pct" if abs(pct - round(pct)) < 1e-8 else f"{pct:.1f}pct"
    else:
        text = f"{pct:.2f}pct"
    return "top_" + text.replace(".", "p")


def _masked_flat(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    flat = values.detach().float().flatten()
    if mask is None:
        return flat
    return flat[mask.detach().bool().flatten()]


def _region_residual_stats(
    prefix: str,
    residual_norm: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, float]:
    vals_hu = _masked_flat(residual_norm, mask) * HU_SCALE
    abs_hu = vals_hu.abs()
    if vals_hu.numel() == 0:
        raise ValueError(f"empty region for {prefix}")
    qs = torch.quantile(abs_hu, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=abs_hu.dtype))
    return {
        f"{prefix}_signed_mean_hu": float(vals_hu.mean().item()),
        f"{prefix}_signed_std_hu": float(vals_hu.std(unbiased=False).item()),
        f"{prefix}_mae_hu": float(abs_hu.mean().item()),
        f"{prefix}_rmse_hu": float(vals_hu.pow(2).mean().sqrt().item()),
        f"{prefix}_abs_p50_hu": float(qs[0].item()),
        f"{prefix}_abs_p90_hu": float(qs[1].item()),
        f"{prefix}_abs_p95_hu": float(qs[2].item()),
        f"{prefix}_abs_p99_hu": float(qs[3].item()),
        f"{prefix}_abs_max_hu": float(abs_hu.max().item()),
        f"{prefix}_positive_fraction": float((vals_hu > 0).float().mean().item()),
    }


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = a_centered.norm() * b_centered.norm()
    if float(denom.item()) <= 0.0:
        return float("nan")
    return float((a_centered * b_centered).sum().div(denom).item())


def _concentration_stats(
    abs_residual_hu: torch.Tensor,
    abs_corrupted_error_hu: torch.Tensor,
    heart_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
    top_fracs: list[float],
) -> dict[str, float]:
    abs_res_flat = abs_residual_hu.detach().float().flatten()
    abs_corr_flat = abs_corrupted_error_hu.detach().float().flatten()
    heart_flat = heart_mask.detach().bool().flatten()
    boundary_flat = boundary_mask.detach().bool().flatten()
    total_error = abs_res_flat.sum().clamp_min(1e-8)
    n_vox = int(abs_res_flat.numel())

    heart_error = abs_res_flat[heart_flat].sum()
    boundary_error = abs_res_flat[boundary_flat].sum()
    heart_volume_fraction = float(heart_flat.float().mean().item())
    boundary_volume_fraction = float(boundary_flat.float().mean().item())
    heart_error_fraction = float((heart_error / total_error).item())
    boundary_error_fraction = float((boundary_error / total_error).item())
    out = {
        "heart_volume_fraction": heart_volume_fraction,
        "boundary_volume_fraction": boundary_volume_fraction,
        "heart_abs_error_fraction": heart_error_fraction,
        "boundary_abs_error_fraction": boundary_error_fraction,
        "heart_error_enrichment": heart_error_fraction / max(heart_volume_fraction, 1e-8),
        "boundary_error_enrichment": boundary_error_fraction / max(boundary_volume_fraction, 1e-8),
        "abs_residual_vs_abs_corrupted_error_pearson": _pearson(abs_res_flat, abs_corr_flat),
    }

    for frac in top_fracs:
        if not 0.0 < frac < 1.0:
            raise ValueError(f"top fraction must be in (0, 1), got {frac}")
        k = max(1, int(round(n_vox * frac)))
        label = _frac_label(frac)
        top_res = torch.topk(abs_res_flat, k=k, largest=True, sorted=False)
        top_corr_idx = torch.topk(abs_corr_flat, k=k, largest=True, sorted=False).indices
        top_corr_mask = torch.zeros(n_vox, dtype=torch.bool)
        top_corr_mask[top_corr_idx] = True
        out.update(
            {
                f"{label}_residual_error_fraction": float((top_res.values.sum() / total_error).item()),
                f"{label}_residual_voxels_in_heart_fraction": float(
                    heart_flat[top_res.indices].float().mean().item()
                ),
                f"{label}_residual_voxels_in_boundary_fraction": float(
                    boundary_flat[top_res.indices].float().mean().item()
                ),
                f"{label}_residual_voxels_in_top_corrupted_error_fraction": float(
                    top_corr_mask[top_res.indices].float().mean().item()
                ),
                f"{label}_residual_threshold_hu": float(top_res.values.min().item()),
            }
        )
    return out


def _best_residual_slice(abs_residual_hu: torch.Tensor, boundary_mask: torch.Tensor) -> int:
    values = abs_residual_hu[0, 0].float()
    boundary = boundary_mask[0, 0].bool()
    if bool(boundary.any().item()):
        score = (values * boundary.float()).flatten(1).sum(dim=1)
    else:
        score = values.flatten(1).sum(dim=1)
    return int(score.argmax().item())


def save_residual_panel(
    out_path: Path,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    pred: torch.Tensor,
    heart_mask: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_v = clean[0, 0].detach().float().cpu()
    corr_v = corrupted[0, 0].detach().float().cpu()
    pred_v = pred[0, 0].detach().float().cpu()
    signed_res_hu = (clean - pred)[0, 0].detach().float().cpu() * HU_SCALE
    abs_res_hu = signed_res_hu.abs()
    corr_abs_hu = (corrupted - clean).abs()[0, 0].detach().float().cpu() * HU_SCALE
    heart_v = heart_mask[0, 0].detach().float().cpu()
    boundary_v = boundary_mask[0, 0].detach().float().cpu()
    z = _best_residual_slice(abs_res_hu.unsqueeze(0).unsqueeze(0), boundary_mask.cpu())

    panels = [
        ("clean", clean_v[z], "gray", -1.0, 1.0),
        ("corrupted", corr_v[z], "gray", -1.0, 1.0),
        ("unet", pred_v[z], "gray", -1.0, 1.0),
        ("abs residual HU", abs_res_hu[z], "magma", 0.0, 200.0),
        ("signed residual HU", signed_res_hu[z], "coolwarm", -150.0, 150.0),
        ("corrupted abs err HU", corr_abs_hu[z], "magma", 0.0, 600.0),
        ("heart mask", heart_v[z], "gray", 0.0, 1.0),
        ("boundary band", boundary_v[z], "gray", 0.0, 1.0),
    ]

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (title, arr, cmap, vmin, vmax) in zip(np.asarray(axs).reshape(-1), panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        if "HU" in title:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; full-volume U-Net diagnosis is GPU-only.")
    device = torch.device(args.device)
    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info("selected %d cases: %s", len(selected), selected)
    model = load_model(args.config, args.ckpt, args.weights, device)
    rows: list[dict[str, object]] = []

    for idx, (split_name, case_id) in enumerate(selected):
        clean, corrupted = load_pair(case_id, processed_dir)
        heart_mask = load_heart_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
        boundary_mask = make_boundary_band(heart_mask, radius=args.boundary_radius)
        pred = sliding_window_unet_correct(
            corrupted,
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

        residual = clean.float() - pred.float()
        abs_residual_hu = residual.abs() * HU_SCALE
        abs_corrupted_error_hu = (corrupted.float() - clean.float()).abs() * HU_SCALE
        row: dict[str, object] = {
            "split": split_name,
            "case_id": case_id,
            "shape": "x".join(str(v) for v in clean.shape[-3:]),
            "voxels": float(residual.numel()),
            "heart_voxels": float(heart_mask.sum().item()),
            "boundary_voxels": float(boundary_mask.sum().item()),
            "corrupted_global_mae_hu": float(abs_corrupted_error_hu.mean().item()),
        }
        row.update(_region_residual_stats("residual_global", residual, None))
        row.update(_region_residual_stats("residual_heart", residual, heart_mask))
        row.update(_region_residual_stats("residual_boundary", residual, boundary_mask))
        row.update(
            _concentration_stats(
                abs_residual_hu,
                abs_corrupted_error_hu,
                heart_mask,
                boundary_mask,
                [float(v) for v in args.top_fracs],
            )
        )
        rows.append(row)
        log.info(
            "%s case %s | residual MAE %.2f HU heart %.2f HU boundary %.2f HU",
            split_name,
            case_id,
            row["residual_global_mae_hu"],
            row["residual_heart_mae_hu"],
            row["residual_boundary_mae_hu"],
        )

        if idx < args.figure_cases:
            save_residual_panel(
                figures_dir / f"{split_name}_case_{case_id}_residual.png",
                clean,
                corrupted,
                pred,
                heart_mask,
                boundary_mask,
            )

    # Keep severity labels consistent with the existing metric suite.
    assign_artifact_severity(rows, score_key="corrupted_global_mae_hu")

    metrics_path = args.out_dir / "residual_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "git_sha": git_sha(),
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "boundary_radius": args.boundary_radius,
        "top_fracs": [float(v) for v in args.top_fracs],
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
