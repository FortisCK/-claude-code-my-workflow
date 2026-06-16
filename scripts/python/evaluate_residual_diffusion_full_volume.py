#!/usr/bin/env python
"""Full-volume evaluation for posterior residual diffusion v2."""

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
from omegaconf import DictConfig, OmegaConf

from code.data import paths as path_registry  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.evaluation.artifact_metrics import (  # noqa: E402
    assign_artifact_severity,
    load_heart_mask,
    load_lumen_mask,
    metric_row,
    summarize,
)
from code.evaluation.metrics import make_boundary_band, masked_mean  # noqa: E402
from code.inference.residual_diffusion_sliding_window import (  # noqa: E402
    sliding_window_residual_diffusion_correct,
)
from code.models.conditional_image_denoiser import ConditionalImageDenoiser  # noqa: E402
from code.models.residual_edm import ResidualEDM  # noqa: E402
from code.training.train_residual_refiner import load_initializer  # noqa: E402

log = logging.getLogger("evaluate_residual_diffusion_full_volume")

_HU_LO, _HU_HI = -1024.0, 3071.0  # matches preprocessing.py normalization


def _save_volume_nifti(tensor: torch.Tensor, path: "Path") -> None:
    """Save a normalized [-1,1] (1,1,D,H,W) volume as an int16 HU NIfTI (1mm iso)."""
    import SimpleITK as sitk  # local import: only when --save-volumes-dir is used

    arr = tensor.detach().float().cpu().numpy()[0, 0]  # (Z, Y, X), normalized
    hu = ((arr + 1.0) * 0.5 * (_HU_HI - _HU_LO) + _HU_LO).astype(np.int16)
    img = sitk.GetImageFromArray(hu)
    img.SetSpacing((1.0, 1.0, 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))

DEFAULT_ROI_SIZE = 128
DEFAULT_OVERLAP = 0.5
DEFAULT_SIGMA_SCALE = 0.125


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate posterior residual diffusion on full 192^3 volumes.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/diffusion_v2_residual.yaml"))
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
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument(
        "--aggregate",
        choices=["mean"],
        default="mean",
        help="How to aggregate multiple residual samples. Currently only posterior mean is supported.",
    )
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--save-volumes-dir", type=Path, default=None,
                   help="if set, export per-case NIfTI HU volumes "
                        "(clean/corrupted/unet/diff_mean/diff_sample) for downstream segmentation")
    p.add_argument("--sample-sigma-min", type=float, default=None)
    p.add_argument("--sample-sigma-max", type=float, default=None)
    p.add_argument("--stochastic-churn", action="store_true")
    p.add_argument(
        "--save-uncertainty",
        action="store_true",
        help="Save posterior mean residual and residual std maps as compressed npz files.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--boundary-radius", type=int, default=3)
    p.add_argument("--figure-cases", type=int, default=5)
    p.add_argument("--progress", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/runs/diffusion_v2_residual/eval_full_volume"),
    )
    return p


def _cfg_block(block: object) -> DictConfig:
    return OmegaConf.create(block)


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


def build_denoiser_from_cfg(cfg: DictConfig) -> ConditionalImageDenoiser:
    return ConditionalImageDenoiser(
        target_channels=int(cfg.target_channels),
        condition_channels=int(cfg.condition_channels),
        channels=tuple(int(v) for v in cfg.channels),
        num_res_blocks=tuple(int(v) for v in cfg.num_res_blocks),
        attention_levels=tuple(bool(v) for v in cfg.attention_levels),
        num_head_channels=tuple(int(v) for v in cfg.num_head_channels),
        norm_num_groups=int(cfg.norm_num_groups),
    )


def build_engine_from_cfg(denoiser: ConditionalImageDenoiser, denoiser_cfg: DictConfig, edm_cfg: DictConfig) -> ResidualEDM:
    return ResidualEDM(
        denoiser=denoiser,
        target_channels=int(denoiser_cfg.target_channels),
        condition_channels=int(denoiser_cfg.condition_channels),
        sigma_min=float(edm_cfg.sigma_min),
        sigma_max=float(edm_cfg.sigma_max),
        sigma_data=float(edm_cfg.sigma_data),
        rho=float(edm_cfg.rho),
        P_mean=float(edm_cfg.P_mean),
        P_std=float(edm_cfg.P_std),
        S_churn=float(edm_cfg.S_churn),
        S_tmin=float(edm_cfg.S_tmin),
        S_tmax=float(edm_cfg.S_tmax),
        S_noise=float(edm_cfg.S_noise),
        clip_pred=bool(edm_cfg.clip_pred),
        clip_value=float(edm_cfg.clip_value),
        final_l1_weight=float(edm_cfg.get("final_l1_weight", 0.0)),
        residual_l1_weight=float(edm_cfg.get("residual_l1_weight", 0.0)),
        gradient_weight=float(edm_cfg.get("gradient_weight", 0.0)),
    )


def load_residual_diffusion(
    config_path: Path,
    ckpt_path: Path,
    weights: str,
    device: torch.device,
) -> tuple[ResidualEDM, dict[str, object]]:
    cfg = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location=device)
    denoiser_cfg = _cfg_block(state.get("denoiser_cfg") or cfg.denoiser)
    edm_cfg = _cfg_block(state.get("engine_cfg") or cfg.edm)
    denoiser = build_denoiser_from_cfg(denoiser_cfg).to(device)
    if weights == "ema" and state.get("ema_denoiser") is not None:
        denoiser.load_state_dict(state["ema_denoiser"])
        log.info("[residual diffusion] loaded EMA denoiser")
    else:
        denoiser.load_state_dict(state["denoiser"])
        log.info("[residual diffusion] loaded online denoiser")
    engine = build_engine_from_cfg(denoiser, denoiser_cfg, edm_cfg).to(device)
    engine.eval()
    for p in engine.parameters():
        p.requires_grad_(False)
    meta = {
        "checkpoint_epoch": state.get("epoch", "unknown"),
        "denoiser_cfg": OmegaConf.to_container(denoiser_cfg, resolve=True),
        "engine_cfg": OmegaConf.to_container(edm_cfg, resolve=True),
        "initializer_cfg": state.get("initializer_cfg") or OmegaConf.to_container(cfg.initializer, resolve=True),
    }
    log.info("[residual diffusion] loaded %s @ epoch %s", ckpt_path, meta["checkpoint_epoch"])
    return engine, meta


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.flatten().float()
    bv = b.flatten().float()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = (av.pow(2).mean().sqrt() * bv.pow(2).mean().sqrt()).clamp_min(1e-8)
    return float((av * bv).mean().div(denom).item())


def uncertainty_row(
    pred: torch.Tensor,
    target: torch.Tensor,
    uncertainty: torch.Tensor,
    heart_mask: torch.Tensor,
    boundary_radius: int,
) -> dict[str, float]:
    abs_err = (pred.float() - target.float()).abs()
    mask = heart_mask.to(device=pred.device)
    boundary = make_boundary_band(mask, radius=boundary_radius).to(device=pred.device)
    return {
        "diffusion_v2_uncertainty_mean_norm": float(uncertainty.mean().item()),
        "diffusion_v2_uncertainty_heart_mean_norm": float(masked_mean(uncertainty, mask).item()),
        "diffusion_v2_uncertainty_boundary_mean_norm": float(masked_mean(uncertainty, boundary).item()),
        "diffusion_v2_uncertainty_error_corr": _pearson(uncertainty, abs_err),
    }


def save_panel(
    out_path: Path,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    initial: torch.Tensor,
    pred: torch.Tensor,
    uncertainty: torch.Tensor | None,
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
    if uncertainty is None:
        sixth_title = "diffusion abs err"
        sixth_arr = pred_err_v[z]
        sixth_cmap = "magma"
        sixth_vmin = 0.0
        sixth_vmax = 0.4
    else:
        unc_v = uncertainty[0, 0].detach().float().cpu()
        sixth_title = "posterior std"
        sixth_arr = unc_v[z]
        sixth_cmap = "viridis"
        sixth_vmin = 0.0
        sixth_vmax = max(float(unc_v.quantile(0.99).item()), 1e-6)
    panels = [
        ("clean", clean_v[z], "gray", -1.0, 1.0),
        ("corrupted", corr_v[z], "gray", -1.0, 1.0),
        ("unet init", init_v[z], "gray", -1.0, 1.0),
        ("diffusion v2", pred_v[z], "gray", -1.0, 1.0),
        ("unet abs err", init_err_v[z], "magma", 0.0, 0.4),
        (sixth_title, sixth_arr, sixth_cmap, sixth_vmin, sixth_vmax),
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
    if args.save_uncertainty and args.n_samples < 2:
        log.warning("--save-uncertainty requested with n_samples < 2; no residual std maps will be saved.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; full-volume diffusion eval is GPU-only.")
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
    uncertainty_dir = args.out_dir / "uncertainty"
    if args.save_uncertainty:
        uncertainty_dir.mkdir(parents=True, exist_ok=True)

    initializer = load_initializer(cfg.initializer, device)
    engine, model_meta = load_residual_diffusion(args.config, args.ckpt, args.weights, device)
    if args.sample_sigma_min is not None:
        engine.sigma_min = float(args.sample_sigma_min)
        model_meta["sample_sigma_min_override"] = float(args.sample_sigma_min)
    if args.sample_sigma_max is not None:
        engine.sigma_max = float(args.sample_sigma_max)
        model_meta["sample_sigma_max_override"] = float(args.sample_sigma_max)
    num_steps = int(args.num_steps or model_meta["engine_cfg"].get("num_sample_steps_eval", 32))
    rows: list[dict[str, object]] = []

    for idx, (split_name, case_id) in enumerate(selected):
        clean, corrupted = load_pair(case_id, processed_dir)
        heart_mask = load_heart_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
        lumen_mask = load_lumen_mask(case_id, processed_dir, tuple(clean.shape[-3:]))
        row: dict[str, object] = {
            "split": split_name,
            "case_id": case_id,
            "shape": "x".join(str(v) for v in clean.shape[-3:]),
            "heart_voxels": float(heart_mask.sum().item()),
            "lumen_voxels": float(lumen_mask.sum().item()) if lumen_mask is not None else 0.0,
        }
        row.update(
            metric_row(
                "corrupted",
                corrupted,
                clean,
                heart_mask=heart_mask,
                lumen_mask=lumen_mask,
                boundary_radius=args.boundary_radius,
            )
        )

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
            residual_scale=args.residual_scale,
            deterministic=not args.stochastic_churn,
            progress=args.progress,
            return_initial=True,
            return_uncertainty=args.n_samples > 1,
            return_single_sample=True,
        )
        # Return order: corrected, initial, [uncertainty], single_sample
        if args.n_samples > 1:
            pred, initial, uncertainty, single = outputs
        else:
            pred, initial, single = outputs
            uncertainty = None
        row.update(
            metric_row(
                "unet_init",
                initial,
                clean,
                heart_mask=heart_mask,
                lumen_mask=lumen_mask,
                boundary_radius=args.boundary_radius,
            )
        )
        row.update(
            metric_row(
                "diffusion_v2",
                pred,
                clean,
                heart_mask=heart_mask,
                lumen_mask=lumen_mask,
                boundary_radius=args.boundary_radius,
            )
        )
        row.update(
            metric_row(
                "diffusion_v2_sample",
                single,
                clean,
                heart_mask=heart_mask,
                lumen_mask=lumen_mask,
                boundary_radius=args.boundary_radius,
            )
        )
        row["diffusion_v2_sample_delta_vs_unet_mae_hu"] = (
            float(row["diffusion_v2_sample_mae_hu"]) - float(row["unet_init_mae_hu"])
        )
        row["diffusion_v2_delta_vs_unet_mae_hu"] = (
            float(row["diffusion_v2_mae_hu"]) - float(row["unet_init_mae_hu"])
        )
        row["diffusion_v2_delta_vs_unet_heart_mae_hu"] = (
            float(row["diffusion_v2_heart_mae_hu"]) - float(row["unet_init_heart_mae_hu"])
        )
        row["diffusion_v2_delta_vs_unet_boundary_mae_hu"] = (
            float(row["diffusion_v2_boundary_mae_hu"]) - float(row["unet_init_boundary_mae_hu"])
        )
        row["diffusion_v2_min"] = float(pred.min().item())
        row["diffusion_v2_max"] = float(pred.max().item())
        if uncertainty is not None:
            row.update(
                uncertainty_row(
                    pred,
                    clean,
                    uncertainty,
                    heart_mask=heart_mask,
                    boundary_radius=args.boundary_radius,
                )
            )
            if args.save_uncertainty:
                scale = float(args.residual_scale)
                if abs(scale) > 1e-8:
                    residual_mean = (pred - initial) / scale
                    residual_std = uncertainty / abs(scale)
                else:
                    residual_mean = torch.zeros_like(pred)
                    residual_std = uncertainty
                np.savez_compressed(
                    uncertainty_dir / f"{split_name}_case_{case_id}.npz",
                    residual_mean=residual_mean[0, 0].detach().float().cpu().numpy(),
                    residual_std=residual_std[0, 0].detach().float().cpu().numpy(),
                    corrected=pred[0, 0].detach().float().cpu().numpy(),
                    initial=initial[0, 0].detach().float().cpu().numpy(),
                )
        if args.save_volumes_dir is not None:
            vdir = args.save_volumes_dir
            _save_volume_nifti(clean, vdir / f"case_{case_id}__clean.nii.gz")
            _save_volume_nifti(corrupted, vdir / f"case_{case_id}__corrupted.nii.gz")
            _save_volume_nifti(initial, vdir / f"case_{case_id}__unet.nii.gz")
            _save_volume_nifti(pred, vdir / f"case_{case_id}__diff_mean.nii.gz")
            _save_volume_nifti(single, vdir / f"case_{case_id}__diff_sample.nii.gz")

        rows.append(row)
        log.info(
            "%s case %s | unet MAE %.2f diffusion_v2 %.2f delta %.2f",
            split_name,
            case_id,
            row["unet_init_mae_hu"],
            row["diffusion_v2_mae_hu"],
            row["diffusion_v2_delta_vs_unet_mae_hu"],
        )

        if idx < args.figure_cases:
            save_panel(
                figures_dir / f"{split_name}_case_{case_id}.png",
                clean,
                corrupted,
                initial,
                pred,
                uncertainty,
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
        "initializer": OmegaConf.to_container(cfg.initializer, resolve=True),
        "model_meta": model_meta,
        "roi_size": [args.roi_size, args.roi_size, args.roi_size],
        "overlap": args.overlap,
        "blend_mode": args.blend_mode,
        "sigma_scale": args.sigma_scale,
        "sw_batch_size": args.sw_batch_size,
        "num_steps": num_steps,
        "n_samples": args.n_samples,
        "aggregate": args.aggregate,
        "residual_scale": args.residual_scale,
        "save_uncertainty": bool(args.save_uncertainty),
        "sample_sigma_min": engine.sigma_min,
        "sample_sigma_max": engine.sigma_max,
        "stochastic_churn": bool(args.stochastic_churn),
        "boundary_radius": args.boundary_radius,
        "case_selection": {
            "val_cases": args.val_cases,
            "test_cases": args.test_cases,
            "case_ids": args.case_ids,
            "selected_cases": [{"split": split_name, "case_id": case_id} for split_name, case_id in selected],
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
