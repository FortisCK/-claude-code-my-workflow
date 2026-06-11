#!/usr/bin/env python
"""Analyze voxel-level calibration of residual-gate refinement.

This script reruns only GateNet inference on cached posterior residual features.
It does not rerun diffusion sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

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

from code.evaluation.metrics import make_boundary_band  # noqa: E402
from code.training.train_residual_gate import make_gate_features  # noqa: E402
from scripts.python.evaluate_residual_gate_full_volume import (  # noqa: E402
    DEFAULT_OVERLAP,
    DEFAULT_ROI_SIZE,
    DEFAULT_SIGMA_SCALE,
    load_cache,
    load_gate,
    sliding_window_gate,
)

log = logging.getLogger("analyze_residual_gate_calibration")

HU_SCALE = 2047.5

STAT_KEYS = (
    "improvement_hu",
    "positive",
    "init_abs_error_hu",
    "learned_abs_error_hu",
    "posterior_abs_error_hu",
    "posterior_improvement_hu",
    "posterior_positive",
    "gate",
    "residual_abs_hu",
    "residual_std_hu",
    "residual_snr",
    "applied_abs_hu",
)

BIN_SPECS = {
    "gate": [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200],
    "residual_std_hu": [10.0, 20.0, 40.0, 60.0, 80.0, 120.0],
    "residual_snr": [0.10, 0.25, 0.50, 1.00, 2.00, 4.00],
    "residual_abs_hu": [5.0, 10.0, 20.0, 40.0, 80.0, 120.0],
    "applied_abs_hu": [0.05, 0.10, 0.25, 0.50, 1.00, 2.00, 5.00],
}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze residual gate calibration from cached test features.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/residual_gate_v1.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def _read_metric_rows(path: Optional[Path]) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as f:
        return {str(row["case_id"]): row for row in csv.DictReader(f)}


def _discover_cache_files(
    cache_dir: Path,
    splits: Iterable[str],
    case_ids: Optional[list[str]],
    max_cases: Optional[int],
) -> list[tuple[str, str, Path]]:
    allowed = {str(cid) for cid in case_ids} if case_ids else None
    selected: list[tuple[str, str, Path]] = []
    for split in splits:
        for path in sorted((cache_dir / split).glob("case_*.npz")):
            case_id = path.stem.removeprefix("case_")
            if allowed is not None and case_id not in allowed:
                continue
            selected.append((split, case_id, path))
            if max_cases is not None and len(selected) >= int(max_cases):
                return selected
    if not selected:
        raise FileNotFoundError(f"no cache files found under {cache_dir} for splits={list(splits)}")
    return selected


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    return float(values[mask].mean().item())


def _masked_positive_fraction(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return float("nan")
    return float((values[mask] > 0).float().mean().item())


def _label_for_bin(thresholds: list[float], idx: int) -> tuple[str, float, str]:
    if idx == 0:
        return f"<{thresholds[0]:g}", 0.0, f"{thresholds[0]:g}"
    if idx == len(thresholds):
        return f">={thresholds[-1]:g}", float(thresholds[-1]), "inf"
    return f"[{thresholds[idx - 1]:g},{thresholds[idx]:g})", float(thresholds[idx - 1]), f"{thresholds[idx]:g}"


def _bincount(idx: torch.Tensor, weights: torch.Tensor, n_bins: int) -> np.ndarray:
    return torch.bincount(idx, weights=weights.double(), minlength=n_bins).cpu().numpy()[:n_bins]


def _accumulate(
    accum: dict[tuple[str, str, int], dict[str, float]],
    bin_type: str,
    region: str,
    bin_values: torch.Tensor,
    stats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    thresholds: list[float],
) -> None:
    if not bool(mask.any().item()):
        return
    flat_mask = mask.reshape(-1)
    values = bin_values.reshape(-1)[flat_mask]
    if values.numel() == 0:
        return
    threshold_tensor = torch.tensor(thresholds, dtype=values.dtype, device=values.device)
    bin_idx = torch.bucketize(values, threshold_tensor, right=False).to(dtype=torch.long)
    n_bins = len(thresholds) + 1
    counts = torch.bincount(bin_idx, minlength=n_bins).cpu().numpy()[:n_bins]
    for bin_i, count in enumerate(counts):
        if int(count) == 0:
            continue
        key = (bin_type, region, int(bin_i))
        target = accum[key]
        target["count"] += float(count)
    for stat_name, stat_tensor in stats.items():
        stat_values = stat_tensor.reshape(-1)[flat_mask]
        sums = _bincount(bin_idx, stat_values, n_bins)
        for bin_i, value in enumerate(sums):
            if int(counts[bin_i]) == 0:
                continue
            accum[(bin_type, region, int(bin_i))][f"sum_{stat_name}"] += float(value)


def _accumulate_region(
    accum: dict[str, dict[str, float]],
    region: str,
    stats: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> None:
    if not bool(mask.any().item()):
        return
    flat_mask = mask.reshape(-1)
    count = int(flat_mask.sum().item())
    target = accum[region]
    target["count"] += float(count)
    for stat_name, stat_tensor in stats.items():
        target[f"sum_{stat_name}"] += float(stat_tensor.reshape(-1)[flat_mask].double().sum().item())


def _row_from_sums(prefix: dict[str, float]) -> dict[str, float]:
    count = max(float(prefix.get("count", 0.0)), 1.0)
    row = {"voxels": float(prefix.get("count", 0.0))}
    for stat_name in STAT_KEYS:
        row[f"mean_{stat_name}"] = float(prefix.get(f"sum_{stat_name}", 0.0)) / count
    row["mean_error_reduction_ratio"] = (
        row["mean_improvement_hu"] / max(row["mean_init_abs_error_hu"], 1e-8)
    )
    return row


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    selected = _discover_cache_files(args.cache_dir, args.splits, args.case_ids, args.max_cases)
    metric_rows = _read_metric_rows(args.metrics_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, model_meta = load_gate(args.config, args.ckpt, args.weights, device)

    region_accum: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    bin_accum: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    case_rows: list[dict[str, object]] = []
    log.info("selected %d cached cases", len(selected))

    for idx, (split, case_id, path) in enumerate(selected, start=1):
        data = load_cache(path)
        clean = data["clean"]
        corrupted = data["corrupted"]
        initial = data["initial"]
        residual_mean = data["residual_mean"]
        residual_std = data["residual_std"]
        heart_mask = data["heart_mask"]
        boundary_mask = make_boundary_band(heart_mask, radius=int(args.boundary_radius))
        heart_interior = torch.logical_and(heart_mask, torch.logical_not(boundary_mask))
        non_heart = torch.logical_not(heart_mask)

        features = make_gate_features(corrupted[0], initial[0], residual_mean[0], residual_std[0]).unsqueeze(0)
        gate = sliding_window_gate(
            features,
            model,
            roi_size=int(args.roi_size),
            overlap=float(args.overlap),
            sw_batch_size=int(args.sw_batch_size),
            sw_device=device,
            output_device="cpu",
            blend_mode=args.blend_mode,
            sigma_scale=float(args.sigma_scale),
            progress=args.progress,
        )
        learned = initial + gate * residual_mean
        posterior = initial + residual_mean
        init_abs_error_hu = (initial - clean).abs() * HU_SCALE
        learned_abs_error_hu = (learned - clean).abs() * HU_SCALE
        posterior_abs_error_hu = (posterior - clean).abs() * HU_SCALE
        improvement_hu = init_abs_error_hu - learned_abs_error_hu
        posterior_improvement_hu = init_abs_error_hu - posterior_abs_error_hu
        residual_abs_hu = residual_mean.abs() * HU_SCALE
        residual_std_hu = residual_std * HU_SCALE
        residual_snr = residual_abs_hu / (residual_std_hu + 1e-3)
        applied_abs_hu = (gate * residual_mean).abs() * HU_SCALE
        stats = {
            "improvement_hu": improvement_hu,
            "positive": (improvement_hu > 0).float(),
            "init_abs_error_hu": init_abs_error_hu,
            "learned_abs_error_hu": learned_abs_error_hu,
            "posterior_abs_error_hu": posterior_abs_error_hu,
            "posterior_improvement_hu": posterior_improvement_hu,
            "posterior_positive": (posterior_improvement_hu > 0).float(),
            "gate": gate,
            "residual_abs_hu": residual_abs_hu,
            "residual_std_hu": residual_std_hu,
            "residual_snr": residual_snr,
            "applied_abs_hu": applied_abs_hu,
        }
        regions = {
            "all": torch.ones_like(heart_mask, dtype=torch.bool),
            "heart": heart_mask,
            "boundary": boundary_mask,
            "heart_interior": heart_interior,
            "non_heart": non_heart,
        }
        bin_values = {
            "gate": gate,
            "residual_std_hu": residual_std_hu,
            "residual_snr": residual_snr,
            "residual_abs_hu": residual_abs_hu,
            "applied_abs_hu": applied_abs_hu,
        }
        for region_name, mask in regions.items():
            _accumulate_region(region_accum, region_name, stats, mask)
            for bin_type, thresholds in BIN_SPECS.items():
                _accumulate(bin_accum, bin_type, region_name, bin_values[bin_type], stats, mask, thresholds)

        metric_row = metric_rows.get(str(case_id), {})
        case_rows.append(
            {
                "split": split,
                "case_id": case_id,
                "cache_path": str(path),
                "artifact_severity": metric_row.get("artifact_severity", ""),
                "global_mean_improvement_hu": _masked_mean(improvement_hu, regions["all"]),
                "global_win_fraction": _masked_positive_fraction(improvement_hu, regions["all"]),
                "heart_mean_improvement_hu": _masked_mean(improvement_hu, heart_mask),
                "heart_win_fraction": _masked_positive_fraction(improvement_hu, heart_mask),
                "boundary_mean_improvement_hu": _masked_mean(improvement_hu, boundary_mask),
                "boundary_win_fraction": _masked_positive_fraction(improvement_hu, boundary_mask),
                "non_heart_mean_improvement_hu": _masked_mean(improvement_hu, non_heart),
                "gate_mean": _masked_mean(gate, regions["all"]),
                "gate_heart_mean": _masked_mean(gate, heart_mask),
                "gate_boundary_mean": _masked_mean(gate, boundary_mask),
                "residual_std_heart_mean_hu": _masked_mean(residual_std_hu, heart_mask),
                "residual_snr_heart_mean": _masked_mean(residual_snr, heart_mask),
            }
        )
        log.info(
            "[%d/%d] %s case %s | global %.4f HU heart %.4f HU boundary %.4f HU gate %.4f",
            idx,
            len(selected),
            split,
            case_id,
            case_rows[-1]["global_mean_improvement_hu"],
            case_rows[-1]["heart_mean_improvement_hu"],
            case_rows[-1]["boundary_mean_improvement_hu"],
            case_rows[-1]["gate_mean"],
        )

    region_rows: list[dict[str, object]] = []
    for region, sums in sorted(region_accum.items()):
        row: dict[str, object] = {"region": region}
        row.update(_row_from_sums(sums))
        region_rows.append(row)

    bin_rows: list[dict[str, object]] = []
    for (bin_type, region, bin_idx), sums in sorted(bin_accum.items()):
        thresholds = BIN_SPECS[bin_type]
        label, lower, upper = _label_for_bin(thresholds, bin_idx)
        row = {
            "bin_type": bin_type,
            "region": region,
            "bin_index": bin_idx,
            "bin_label": label,
            "lower": lower,
            "upper": upper,
        }
        row.update(_row_from_sums(sums))
        bin_rows.append(row)

    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(args.out_dir / "region_summary.csv", region_rows)
    _write_csv(args.out_dir / "bin_summary.csv", bin_rows)
    _write_csv(args.out_dir / "case_summary.csv", case_rows)

    manifest = {
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "model_meta": model_meta,
        "cache_dir": str(args.cache_dir),
        "metrics_csv": str(args.metrics_csv) if args.metrics_csv else None,
        "selected_cases": [{"split": split, "case_id": case_id, "path": str(path)} for split, case_id, path in selected],
        "bin_specs": BIN_SPECS,
        "outputs": {
            "region_summary": str(args.out_dir / "region_summary.csv"),
            "bin_summary": str(args.out_dir / "bin_summary.csv"),
            "case_summary": str(args.out_dir / "case_summary.csv"),
        },
    }
    (args.out_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
