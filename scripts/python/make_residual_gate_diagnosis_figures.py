#!/usr/bin/env python
"""Create qualitative diagnosis panels for residual-gate refinement runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import OrderedDict
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.evaluation.metrics import make_boundary_band  # noqa: E402
from code.training.train_residual_gate import make_gate_features  # noqa: E402
from scripts.python.evaluate_residual_gate_full_volume import (  # noqa: E402
    DEFAULT_OVERLAP,
    DEFAULT_ROI_SIZE,
    DEFAULT_SIGMA_SCALE,
    gate_feature_kwargs,
    load_cache,
    load_gate,
    sliding_window_gate,
)

log = logging.getLogger("make_residual_gate_diagnosis_figures")

HU_SCALE = 2047.5


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create qualitative residual-gate diagnosis panels.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/residual_gate_v1.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--metrics-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--cases-per-group", type=int, default=5)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", action="store_true")
    return p


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _add_case(
    selected: OrderedDict[str, set[str]],
    row: dict[str, str],
    group: str,
) -> None:
    cid = str(row["case_id"])
    selected.setdefault(cid, set()).add(group)


def select_cases(rows: list[dict[str, str]], case_ids: Optional[list[str]], cases_per_group: int) -> OrderedDict[str, set[str]]:
    by_case = {str(row["case_id"]): row for row in rows}
    selected: OrderedDict[str, set[str]] = OrderedDict()
    if case_ids:
        for cid in case_ids:
            if str(cid) not in by_case:
                raise KeyError(f"case_id {cid} not found in metrics csv")
            _add_case(selected, by_case[str(cid)], "manual")
        return selected

    n = max(1, int(cases_per_group))
    groups = [
        (
            "best_global_delta",
            sorted(rows, key=lambda r: _float(r, "learned_gate_delta_vs_unet_init_mae_hu"))[:n],
        ),
        (
            "worst_global_delta",
            sorted(rows, key=lambda r: _float(r, "learned_gate_delta_vs_unet_init_mae_hu"), reverse=True)[:n],
        ),
        (
            "highest_unet_mae",
            sorted(rows, key=lambda r: _float(r, "unet_init_mae_hu"), reverse=True)[:n],
        ),
        (
            "highest_boundary_mae",
            sorted(rows, key=lambda r: _float(r, "unet_init_boundary_mae_hu"), reverse=True)[:n],
        ),
    ]
    for group, group_rows in groups:
        for row in group_rows:
            _add_case(selected, row, group)

    for cid in ("21", "32", "41", "50", "51"):
        if cid in by_case:
            _add_case(selected, by_case[cid], "original_test5")
    return selected


def _to_hu(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy() * HU_SCALE


def _center_slice(gate: torch.Tensor, heart_mask: torch.Tensor, boundary_mask: torch.Tensor) -> int:
    gate_3d = gate[0, 0].detach().float().cpu()
    boundary_3d = boundary_mask[0, 0].detach().float().cpu()
    heart_3d = heart_mask[0, 0].detach().float().cpu()
    score = (gate_3d * (boundary_3d + 0.25 * heart_3d)).flatten(1).sum(dim=1)
    if float(score.max().item()) <= 0.0:
        score = heart_3d.flatten(1).sum(dim=1)
    if float(score.max().item()) <= 0.0:
        return int(gate_3d.shape[0] // 2)
    return int(torch.argmax(score).item())


def _imshow(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = False,
) -> None:
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


def _overlay_contours(ax: plt.Axes, heart_slice: np.ndarray, boundary_slice: np.ndarray) -> None:
    if np.any(heart_slice):
        ax.contour(heart_slice.astype(float), levels=[0.5], colors=["lime"], linewidths=0.45)
    if np.any(boundary_slice):
        ax.contour(boundary_slice.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.45)


def save_case_panel(
    out_path: Path,
    row: dict[str, str],
    groups: set[str],
    tensors: dict[str, torch.Tensor],
    gate: torch.Tensor,
    boundary_mask: torch.Tensor,
) -> None:
    clean = tensors["clean"]
    corrupted = tensors["corrupted"]
    initial = tensors["initial"]
    residual_mean = tensors["residual_mean"]
    residual_std = tensors["residual_std"]
    heart_mask = tensors["heart_mask"]
    posterior = initial + residual_mean
    learned = initial + gate * residual_mean
    applied = gate * residual_mean

    z = _center_slice(gate, heart_mask, boundary_mask)
    clean_hu = _to_hu(clean[0, 0, z])
    corrupted_hu = _to_hu(corrupted[0, 0, z])
    initial_hu = _to_hu(initial[0, 0, z])
    posterior_hu = _to_hu(posterior[0, 0, z])
    learned_hu = _to_hu(learned[0, 0, z])
    residual_hu = _to_hu(residual_mean[0, 0, z])
    residual_std_hu = _to_hu(residual_std[0, 0, z])
    applied_hu = _to_hu(applied[0, 0, z])
    gate_np = gate[0, 0, z].detach().float().cpu().numpy()
    heart_np = heart_mask[0, 0, z].detach().cpu().numpy().astype(bool)
    boundary_np = boundary_mask[0, 0, z].detach().cpu().numpy().astype(bool)

    corr_err = np.abs(corrupted_hu - clean_hu)
    unet_err = np.abs(initial_hu - clean_hu)
    learned_err = np.abs(learned_hu - clean_hu)
    error_gain = unet_err - learned_err
    snr = np.abs(residual_hu) / (residual_std_hu + 1e-3)

    fig, axs = plt.subplots(3, 5, figsize=(17, 10.2), constrained_layout=True)
    axes = np.asarray(axs).reshape(3, 5)
    image_window = (-200.0, 800.0)
    err_window = (0.0, 160.0)

    _imshow(axes[0, 0], clean_hu, "clean", vmin=image_window[0], vmax=image_window[1])
    _imshow(axes[0, 1], corrupted_hu, "corrupted", vmin=image_window[0], vmax=image_window[1])
    _imshow(axes[0, 2], initial_hu, "U-Net init", vmin=image_window[0], vmax=image_window[1])
    _imshow(axes[0, 3], learned_hu, "gated refinement", vmin=image_window[0], vmax=image_window[1])
    _imshow(axes[0, 4], posterior_hu, "posterior mean", vmin=image_window[0], vmax=image_window[1])

    _imshow(axes[1, 0], corr_err, "|corrupted-clean| HU", cmap="magma", vmin=err_window[0], vmax=err_window[1], cbar=True)
    _imshow(axes[1, 1], unet_err, "|U-Net-clean| HU", cmap="magma", vmin=err_window[0], vmax=err_window[1], cbar=True)
    _imshow(axes[1, 2], learned_err, "|gated-clean| HU", cmap="magma", vmin=err_window[0], vmax=err_window[1], cbar=True)
    _imshow(axes[1, 3], error_gain, "error gain HU", cmap="coolwarm", vmin=-20.0, vmax=20.0, cbar=True)
    _imshow(axes[1, 4], gate_np, "gate", cmap="viridis", vmin=0.0, vmax=max(float(gate.max().item()), 1e-6), cbar=True)

    _imshow(axes[2, 0], residual_hu, "residual mean HU", cmap="coolwarm", vmin=-100.0, vmax=100.0, cbar=True)
    _imshow(axes[2, 1], residual_std_hu, "residual std HU", cmap="magma", vmin=0.0, vmax=100.0, cbar=True)
    _imshow(axes[2, 2], applied_hu, "gate * residual HU", cmap="coolwarm", vmin=-30.0, vmax=30.0, cbar=True)
    _imshow(axes[2, 3], snr, "|mean| / std", cmap="plasma", vmin=0.0, vmax=2.0, cbar=True)
    _imshow(axes[2, 4], clean_hu, "heart/boundary overlay", vmin=image_window[0], vmax=image_window[1])

    for ax in axes.reshape(-1):
        _overlay_contours(ax, heart_np, boundary_np)

    title = (
        f"case {row['case_id']} | z={z} | groups={','.join(sorted(groups))}\n"
        f"MAE {float(row['unet_init_mae_hu']):.2f}->{float(row['learned_gate_mae_hu']):.2f} "
        f"(d={float(row['learned_gate_delta_vs_unet_init_mae_hu']):.3f} HU), "
        f"heart d={float(row['learned_gate_delta_vs_unet_init_heart_mae_hu']):.3f}, "
        f"boundary d={float(row['learned_gate_delta_vs_unet_init_boundary_mae_hu']):.3f}, "
        f"gate mean={float(row['gate_mean']):.4f}"
    )
    fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = _read_rows(args.metrics_csv)
    by_case = {str(row["case_id"]): row for row in rows}
    selected = select_cases(rows, args.case_ids, args.cases_per_group)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    model, model_meta = load_gate(args.config, args.ckpt, args.weights, device)
    feature_kwargs = gate_feature_kwargs(cfg, model_meta, boundary_radius=int(args.boundary_radius))
    selection_rows: list[dict[str, object]] = []
    log.info("selected %d unique cases: %s", len(selected), list(selected.keys()))

    for idx, (case_id, groups) in enumerate(selected.items(), start=1):
        cache_path = args.cache_dir / "test" / f"case_{case_id}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        data = load_cache(cache_path)
        clean = data["clean"]
        corrupted = data["corrupted"]
        initial = data["initial"]
        residual_mean = data["residual_mean"]
        residual_std = data["residual_std"]
        heart_mask = data["heart_mask"]
        boundary_mask = make_boundary_band(heart_mask, radius=args.boundary_radius)
        feature_4d = make_gate_features(
            corrupted[0],
            initial[0],
            residual_mean[0],
            residual_std[0],
            heart_mask[0],
            **feature_kwargs,
        )
        gate = sliding_window_gate(
            feature_4d.unsqueeze(0),
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
        row = by_case[case_id]
        out_path = figures_dir / f"case_{case_id}_{'_'.join(sorted(groups))}.png"
        save_case_panel(
            out_path,
            row,
            groups,
            {
                "clean": clean,
                "corrupted": corrupted,
                "initial": initial,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "heart_mask": heart_mask,
            },
            gate,
            boundary_mask,
        )
        selection_rows.append(
            {
                "case_id": case_id,
                "groups": ";".join(sorted(groups)),
                "figure": str(out_path),
                "unet_mae_hu": row["unet_init_mae_hu"],
                "learned_gate_mae_hu": row["learned_gate_mae_hu"],
                "delta_mae_hu": row["learned_gate_delta_vs_unet_init_mae_hu"],
                "delta_heart_mae_hu": row["learned_gate_delta_vs_unet_init_heart_mae_hu"],
                "delta_boundary_mae_hu": row["learned_gate_delta_vs_unet_init_boundary_mae_hu"],
                "delta_boundary_gradient_l1_hu": row["learned_gate_delta_vs_unet_init_boundary_gradient_l1_hu"],
                "gate_mean": row["gate_mean"],
                "gate_heart_mean": row["gate_heart_mean"],
                "gate_boundary_mean": row["gate_boundary_mean"],
                "artifact_severity": row.get("artifact_severity", ""),
            }
        )
        log.info("[%d/%d] wrote %s", idx, len(selected), out_path)

    selection_path = args.out_dir / "selected_cases.csv"
    with selection_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selection_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(selection_rows)

    meta = {
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "model_meta": model_meta,
        "feature_kwargs": feature_kwargs,
        "cache_dir": str(args.cache_dir),
        "metrics_csv": str(args.metrics_csv),
        "out_dir": str(args.out_dir),
        "selected_cases": selection_rows,
    }
    (args.out_dir / "diagnosis_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("wrote %s", selection_path)
    log.info("wrote %s", args.out_dir / "diagnosis_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
