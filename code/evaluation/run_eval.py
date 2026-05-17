"""run_eval.py — end-to-end evaluation: posterior sampling + metrics + figures.

For each test case:
    1. Load V_clean and V_corrupted (paired cache OR online stub).
    2. Posterior-sample N chains, decode each, compute (mean, std).
    3. Score (mean) against V_clean: PSNR, SSIM, NRMSE, dice_lumen_stub.
    4. Save mid-axial PNG (clean / corrupted / mean / std) for the first K cases.
    5. Append per-case metrics to a CSV.

Usage:
    python -m code.evaluation.run_eval \\
        --vae-config code/training/configs/vae_v1.yaml \\
        --vae-ckpt experiments/checkpoints/vae_v1/epoch_100.pt \\
        --diff-config code/training/configs/diffusion_v1.yaml \\
        --diff-ckpt experiments/checkpoints/diffusion_v1/epoch_200.pt \\
        --case-ids 1 2 3 \\
        --n-samples 16 --num-steps 50 \\
        --out-dir experiments/runs/eval_v1/

Sanity smoke (uses smoke checkpoints, online motion stub):
    python -m code.evaluation.run_eval --smoke

Per `.claude/rules/python-code-conventions.md`:
    - set_determinism once at top
    - logging not print
    - relative paths via `code.data.paths`
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import paths as path_registry  # noqa: E402
from code.evaluation.metrics import all_metrics  # noqa: E402
from code.inference.posterior_sample import posterior_sample  # noqa: E402
from code.models.conditional_denoiser import ConditionalDenoiser  # noqa: E402
from code.models.edm import EDM  # noqa: E402
from code.models.vae import CardiacVAE  # noqa: E402
from code.training.train_diffusion import load_frozen_vae  # noqa: E402

log = logging.getLogger(__name__)


# ============================================================
# Loaders
# ============================================================


def load_frozen_edm(
    diff_cfg_path: Path, ckpt_path: Path, device: torch.device
) -> tuple[EDM, dict]:
    cfg = OmegaConf.load(diff_cfg_path)
    state = torch.load(ckpt_path, map_location=device)
    if "denoiser_cfg" in state:
        d = state["denoiser_cfg"]
        log.info("[edm] using denoiser_cfg embedded in checkpoint")
    else:
        d = cfg.denoiser
        log.info("[edm] using denoiser_cfg from %s", diff_cfg_path)

    denoiser = ConditionalDenoiser(
        latent_channels=d["latent_channels"],
        channels=tuple(d["channels"]),
        num_res_blocks=tuple(d["num_res_blocks"]),
        attention_levels=tuple(d["attention_levels"]),
        num_head_channels=tuple(d["num_head_channels"]),
        norm_num_groups=d["norm_num_groups"],
    ).to(device)
    if state.get("ema_denoiser") is not None:
        denoiser.load_state_dict(state["ema_denoiser"])
        log.info("[edm] loaded EMA weights")
    else:
        denoiser.load_state_dict(state["denoiser"])
        log.info("[edm] loaded online (non-EMA) weights")
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)

    edm = EDM(
        denoiser=denoiser,
        latent_channels=d["latent_channels"],
        sigma_min=cfg.edm.sigma_min,
        sigma_max=cfg.edm.sigma_max,
        sigma_data=cfg.edm.sigma_data,
        rho=cfg.edm.rho,
        P_mean=cfg.edm.P_mean,
        P_std=cfg.edm.P_std,
        S_churn=cfg.edm.S_churn,
        S_tmin=cfg.edm.S_tmin,
        S_tmax=cfg.edm.S_tmax,
        S_noise=cfg.edm.S_noise,
        clip_pred=cfg.edm.clip_pred,
        clip_value=cfg.edm.clip_value,
    ).to(device)
    edm.eval()
    return edm, OmegaConf.to_container(cfg, resolve=True)


def load_pair(
    case_id: str,
    processed_dir: Path,
    pair_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load (V_clean, V_corrupted) as (1, 1, D, H, W) tensors.

    Modes:
        "precomputed": picks the first available `case_<id>__pair_*.npz`.
        "online_stub": noise-corrupts V_clean (used in the smoke run).
    """
    npz_clean = processed_dir / f"case_{case_id}.npz"
    if not npz_clean.exists():
        raise FileNotFoundError(npz_clean)

    if pair_mode == "precomputed":
        pairs = sorted(processed_dir.glob(f"case_{case_id}__pair_*.npz"))
        if not pairs:
            raise FileNotFoundError(
                f"no pairs for case {case_id} in {processed_dir}; "
                f"use pair_mode='online_stub' for a noise-corrupted smoke."
            )
        with np.load(pairs[0], allow_pickle=True) as data:
            v_clean = torch.from_numpy(data["volume"]).float()
            v_corr = torch.from_numpy(data["corrupted"]).float()
    elif pair_mode == "online_stub":
        with np.load(npz_clean, allow_pickle=True) as data:
            v_clean = torch.from_numpy(data["volume"]).float()
        v_corr = v_clean + 0.05 * torch.randn_like(v_clean)
    else:
        raise ValueError(f"unknown pair_mode: {pair_mode}")

    return v_clean.unsqueeze(0).unsqueeze(0), v_corr.unsqueeze(0).unsqueeze(0)


# ============================================================
# Figure
# ============================================================


def save_quad_panel_png(
    v_clean: torch.Tensor, v_corr: torch.Tensor,
    v_mean: torch.Tensor, v_std: torch.Tensor,
    out_path: Path,
) -> None:
    """Mid-axial slice: clean / corrupted / posterior-mean / posterior-std."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = v_clean.shape[-3] // 2

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(v_clean[0, 0, z].cpu(), cmap="gray", vmin=-1, vmax=1)
    axs[0].set_title("clean")
    axs[1].imshow(v_corr[0, 0, z].cpu(), cmap="gray", vmin=-1, vmax=1)
    axs[1].set_title("corrupted")
    axs[2].imshow(v_mean[0, 0, z].cpu(), cmap="gray", vmin=-1, vmax=1)
    axs[2].set_title("posterior mean")
    axs[3].imshow(v_std[0, 0, z].cpu(), cmap="hot")
    axs[3].set_title("posterior std")
    for ax in axs:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Eval driver
# ============================================================


def run_eval(
    vae: CardiacVAE,
    edm: EDM,
    case_ids: list[str],
    processed_dir: Path,
    out_dir: Path,
    pair_mode: str,
    n_samples: int,
    num_steps: int,
    n_figs: int,
    device: torch.device,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"

    csv_path = out_dir / "eval_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "psnr", "ssim", "nrmse", "dice_lumen_stub", "uncertainty_mean"])

        for i, cid in enumerate(case_ids):
            v_clean, v_corr = load_pair(cid, processed_dir, pair_mode)
            v_clean = v_clean.to(device)
            v_corr = v_corr.to(device)

            out = posterior_sample(
                v_corr, vae, edm,
                n_samples=n_samples, num_steps=num_steps,
                deterministic=False, seed=42,
            )
            v_mean, v_std = out["mean"], out["std"]

            metrics = all_metrics(v_mean, v_clean)
            writer.writerow([
                cid,
                f"{metrics['psnr'].item():.4f}",
                f"{metrics['ssim'].item():.4f}",
                f"{metrics['nrmse'].item():.6f}",
                f"{metrics['dice_lumen_stub'].item():.4f}",
                f"{v_std.mean().item():.6f}",
            ])
            log.info(
                "case %s | PSNR=%.2f SSIM=%.3f NRMSE=%.4f dice_stub=%.3f",
                cid,
                metrics["psnr"].item(), metrics["ssim"].item(),
                metrics["nrmse"].item(), metrics["dice_lumen_stub"].item(),
            )

            if i < n_figs:
                save_quad_panel_png(
                    v_clean, v_corr, v_mean, v_std,
                    fig_dir / f"case_{cid}.png",
                )
    log.info("[eval] metrics → %s", csv_path)
    return csv_path


# ============================================================
# Entry-point
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate cardiac diffusion (Stage 2).")
    p.add_argument("--vae-config", type=Path, default=Path("code/training/configs/vae_v1.yaml"))
    p.add_argument("--vae-ckpt", type=Path,
                   default=Path("experiments/checkpoints/vae_v1/smoke/epoch_000.pt"))
    p.add_argument("--diff-config", type=Path,
                   default=Path("code/training/configs/diffusion_v1.yaml"))
    p.add_argument("--diff-ckpt", type=Path,
                   default=Path("experiments/checkpoints/diffusion_v1/smoke/epoch_000.pt"))
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--case-ids", nargs="+", default=None,
                   help="Explicit case ids. Overrides --split-file if both are given.")
    p.add_argument("--split-file", type=Path,
                   default=Path("data/imagecas/splits/v1.json"),
                   help="JSON split file; the `test` field is used when --case-ids is None.")
    p.add_argument("--pair-mode", choices=["precomputed", "online_stub"], default=None)
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--n-figs", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path("experiments/runs/eval_v1"))
    p.add_argument("--smoke", action="store_true",
                   help="smoke run: 1 case, 4 sampling steps, 2 samples, online stub.")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        device = torch.device("cpu")
        args.n_samples = 2
        args.num_steps = 4
        args.case_ids = args.case_ids or ["1"]
        args.pair_mode = args.pair_mode or "online_stub"
        log.info("[smoke] device=cpu, n_samples=2, num_steps=4, online_stub")

    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    case_ids = args.case_ids or []
    if not case_ids:
        # Prefer the canonical test split. Manuscript metrics MUST come from here.
        if args.split_file and args.split_file.exists():
            from code.data.splits import load_split
            split = load_split(args.split_file)
            case_ids = list(split.test)
            log.info("[eval] using %d test ids from %s", len(case_ids), args.split_file)
        else:
            # Last-resort fallback: first 5 cases on disk (smoke / debugging).
            case_ids = sorted(p.stem.removeprefix("case_")
                              for p in processed_dir.glob("case_*.npz")
                              if "__pair_" not in p.name)[:5]
            log.warning("[eval] no split file — using first 5 cases on disk (debug only)")
    pair_mode = args.pair_mode or "precomputed"

    vae = load_frozen_vae(args.vae_config, args.vae_ckpt, device)
    edm, _ = load_frozen_edm(args.diff_config, args.diff_ckpt, device)

    return int(run_eval(
        vae=vae, edm=edm,
        case_ids=case_ids,
        processed_dir=processed_dir,
        out_dir=args.out_dir,
        pair_mode=pair_mode,
        n_samples=args.n_samples,
        num_steps=args.num_steps,
        n_figs=args.n_figs,
        device=device,
    ).exists()) - 1  # 0 if file written


if __name__ == "__main__":
    sys.exit(main())
