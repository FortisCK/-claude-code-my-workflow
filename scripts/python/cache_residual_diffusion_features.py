#!/usr/bin/env python
"""Cache posterior residual-diffusion features for GateNet training.

The cache stores full-volume arrays per case:

```text
clean
corrupted
initial          = frozen U-Net output
residual_mean    = posterior mean residual
residual_std     = posterior residual std, or zeros when n_samples=1
heart_mask
```

GateNet training can then sample random patches cheaply without running
diffusion sampling in the training loop.
"""

from __future__ import annotations

import argparse
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
from omegaconf import OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.evaluation.artifact_metrics import load_heart_mask  # noqa: E402
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
)

log = logging.getLogger("cache_residual_diffusion_features")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cache residual-diffusion posterior features.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/diffusion_v2_residual.yaml"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights", choices=["ema", "online"], default="ema")
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--train-cases", type=int, default=0)
    p.add_argument("--val-cases", type=int, default=0)
    p.add_argument("--test-cases", type=int, default=0)
    p.add_argument("--case-ids", nargs="+", default=None)
    p.add_argument("--manual-split", default="manual")
    p.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--blend-mode", choices=["constant", "gaussian"], default="gaussian")
    p.add_argument("--sigma-scale", type=float, default=DEFAULT_SIGMA_SCALE)
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--sample-sigma-min", type=float, default=None)
    p.add_argument("--sample-sigma-max", type=float, default=None)
    p.add_argument("--stochastic-churn", action="store_true")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/cache/residual_diffusion_features/v1_small"),
    )
    return p


def select_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.case_ids:
        return [(str(args.manual_split), str(cid)) for cid in args.case_ids]
    split = load_split(args.split_file)
    selected: list[tuple[str, str]] = []
    selected.extend(("train", str(cid)) for cid in split.train[: args.train_cases])
    selected.extend(("val", str(cid)) for cid in split.val[: args.val_cases])
    selected.extend(("test", str(cid)) for cid in split.test[: args.test_cases])
    return selected


def _to_numpy_3d(volume: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    arr = volume[0, 0].detach().float().cpu().numpy()
    return arr.astype(dtype, copy=False)


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; feature caching needs GPU for normal runs.")
    if args.n_samples < 1:
        raise ValueError("--n-samples must be >= 1")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    selected = select_cases(args)
    if not selected:
        raise ValueError("no cases selected; set --train-cases/--val-cases/--test-cases or --case-ids")
    log.info("selected %d cases: %s", len(selected), selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, _case_id in selected:
        (args.out_dir / split_name).mkdir(parents=True, exist_ok=True)

    initializer = load_initializer(cfg.initializer, device)
    engine, model_meta = load_residual_diffusion(args.config, args.ckpt, args.weights, device)
    if args.sample_sigma_min is not None:
        engine.sigma_min = float(args.sample_sigma_min)
        model_meta["sample_sigma_min_override"] = float(args.sample_sigma_min)
    if args.sample_sigma_max is not None:
        engine.sigma_max = float(args.sample_sigma_max)
        model_meta["sample_sigma_max_override"] = float(args.sample_sigma_max)

    np_dtype = np.float16 if args.dtype == "float16" else np.float32
    manifest: dict[str, object] = {
        "git_sha": git_sha(),
        "checkpoint": str(args.ckpt),
        "weights": args.weights,
        "initializer": OmegaConf.to_container(cfg.initializer, resolve=True),
        "model_meta": model_meta,
        "num_steps": int(args.num_steps),
        "n_samples": int(args.n_samples),
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": float(args.overlap),
        "blend_mode": args.blend_mode,
        "sigma_scale": float(args.sigma_scale),
        "dtype": args.dtype,
        "selected_cases": [{"split": split_name, "case_id": case_id} for split_name, case_id in selected],
        "files": [],
    }

    for idx, (split_name, case_id) in enumerate(selected, start=1):
        out_path = args.out_dir / split_name / f"case_{case_id}.npz"
        if out_path.exists() and not args.overwrite:
            log.info("[%d/%d] skip existing %s", idx, len(selected), out_path)
            manifest["files"].append(str(out_path))
            continue

        clean, corrupted = load_pair(case_id, processed_dir)
        heart_mask = load_heart_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
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
            num_steps=int(args.num_steps),
            n_samples=int(args.n_samples),
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
            residual_std = torch.zeros_like(initial)
        residual_mean = posterior_mean_pred - initial

        np.savez_compressed(
            out_path,
            clean=_to_numpy_3d(clean, np_dtype),
            corrupted=_to_numpy_3d(corrupted, np_dtype),
            initial=_to_numpy_3d(initial, np_dtype),
            residual_mean=_to_numpy_3d(residual_mean, np_dtype),
            residual_std=_to_numpy_3d(residual_std, np_dtype),
            heart_mask=heart_mask[0, 0].detach().cpu().numpy().astype(np.uint8, copy=False),
            split=np.asarray(split_name),
            case_id=np.asarray(case_id),
        )
        manifest["files"].append(str(out_path))
        log.info(
            "[%d/%d] cached %s | residual_abs=%.2fHU std=%.2fHU",
            idx,
            len(selected),
            out_path,
            float(residual_mean.abs().mean().item() * 2047.5),
            float(residual_std.mean().item() * 2047.5),
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log.info("wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
