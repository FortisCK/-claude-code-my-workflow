"""train_lumen_refiner.py — task-aware (coronary-lumen) residual refiner.

Run card: experiments/runs/2026-06-16_*_lumen-refiner-pilot.md
Plan: quality_reports/plans/2026-06-16_step2-task-aware-refiner.md

Trains a residual refiner on top of the FROZEN U-Net initializer with a
DIFFERENTIABLE lumen-aware objective (soft-Dice of a contrast response vs the GT
coronary lumen, restricted to a dilated band) + a small L1 fidelity leash. Goal:
capture downstream coronary segmentability the distortion-trained U-Net misses,
judged independently by frozen-segmenter Dice-RCA (NOT in the loss → no gaming).

Self-contained data loop (lumen-centered 128^3 patches; clean/corrupted/lumen
cropped identically) — deliberately does not touch the shared dataloader so the
GT lumen mask stays voxel-aligned to the patch.

Usage:
    python -m code.training.train_lumen_refiner --train-cases 100 --epochs 30 \
        --lumen-weight 1.0 --l1-weight 0.5 --ckpt-dir experiments/checkpoints/lumen_refiner_v1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from omegaconf import OmegaConf

from code.data import paths as path_registry
from code.data.splits import load_split
from code.evaluation.metrics import dilate_mask, gradient_magnitude, lumen_soft_dice_loss
from code.training.train_residual_refiner import build_refiner, load_initializer

log = logging.getLogger("train_lumen_refiner")

PATCH = 128
LUMEN_BAND_RADIUS = 3


def _crop_centered(arr: np.ndarray, center: tuple[int, int, int], size: int) -> np.ndarray:
    out_lo = [max(c - size // 2, 0) for c in center]
    out_lo = [min(lo, arr.shape[i] - size) for i, lo in enumerate(out_lo)]
    out_lo = [max(lo, 0) for lo in out_lo]
    sl = tuple(slice(lo, lo + size) for lo in out_lo)
    return arr[sl]


def _load_case(cid: str, processed_dir: Path, lumen_dir: Path):
    npz = processed_dir / f"case_{cid}__pair_0.npz"
    d = np.load(npz, allow_pickle=True)
    lumen = np.load(lumen_dir / f"lumen_mask_{cid}.npy").astype(np.uint8)
    return d["volume"].astype(np.float32), d["corrupted"].astype(np.float32), lumen


def _sample_patch(clean, corrupted, lumen, rng: np.random.Generator):
    coords = np.argwhere(lumen > 0)
    if coords.shape[0] > 0:
        center = tuple(int(c) for c in coords[rng.integers(coords.shape[0])])
    else:
        center = tuple(s // 2 for s in clean.shape)
    c = _crop_centered(clean, center, PATCH)
    x = _crop_centered(corrupted, center, PATCH)
    l = _crop_centered(lumen, center, PATCH)
    if rng.random() < 0.5:  # cheap axial flip augmentation (applied to all 3)
        ax = int(rng.integers(3))
        c, x, l = np.flip(c, ax).copy(), np.flip(x, ax).copy(), np.flip(l, ax).copy()
    return c, x, l


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train-cases", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lumen-weight", type=float, default=1.0)
    p.add_argument("--l1-weight", type=float, default=0.5)
    p.add_argument("--grad-weight", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ckpt-dir", type=Path, default=Path("experiments/checkpoints/lumen_refiner_v1"))
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    set_determinism(seed=args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    processed_dir = path_registry.get("IMAGECAS_PROCESSED")
    lumen_dir = processed_dir.parent / "lumen_masks"

    init_cfg = OmegaConf.create({
        "config": "code/training/configs/unet_v1.yaml",
        "ckpt": "experiments/checkpoints/unet_v1/epoch_200.pt",
        "weights": "ema",
    })
    initializer = load_initializer(init_cfg, device)
    initializer.eval()
    model_cfg = OmegaConf.create({
        "in_channels": 3, "out_channels": 1,
        "features": [16, 16, 32, 64, 128, 16] if args.smoke else [32, 32, 64, 128, 256, 32],
        "clamp_output": False,
    })
    model = build_refiner(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[model] refiner params=%.2fM device=%s", n_params / 1e6, device)

    split = load_split(path_registry.get("IMAGECAS_SPLITS") / "v1.json")
    cases = [c for c in split.train if (lumen_dir / f"lumen_mask_{c}.npy").exists()][: args.train_cases]
    log.info("[data] %d train cases with lumen GT", len(cases))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_epochs = 1 if args.smoke else args.epochs

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = rng.permutation(len(cases))
        running = {"total": 0.0, "l1": 0.0, "lumen": 0.0, "grad": 0.0}
        n = 0
        for i, idx in enumerate(order):
            if args.smoke and i >= 3:
                break
            clean_np, corr_np, lum_np = _load_case(cases[idx], processed_dir, lumen_dir)
            c, x, l = _sample_patch(clean_np, corr_np, lum_np, rng)
            clean = torch.from_numpy(c)[None, None].to(device)
            corrupted = torch.from_numpy(x)[None, None].to(device)
            lumen = torch.from_numpy(l)[None, None].to(device).bool()
            band = dilate_mask(lumen, LUMEN_BAND_RADIUS)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
                with torch.no_grad():
                    initial = initializer(corrupted).detach()
                final, _ = model(corrupted, initial, return_residual=True)
                ff = final.float()
                l1 = F.l1_loss(ff, clean.float())
                lumen_l = lumen_soft_dice_loss(ff, lumen, band_mask=band).mean()
                grad_l = F.l1_loss(gradient_magnitude(ff), gradient_magnitude(clean.float()))
                loss = args.l1_weight * l1 + args.lumen_weight * lumen_l + args.grad_weight * grad_l
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            running["total"] += float(loss.detach()); running["l1"] += float(l1.detach())
            running["lumen"] += float(lumen_l.detach()); running["grad"] += float(grad_l.detach())
            n += 1
        log.info("[epoch %d] total=%.4f l1=%.4f lumen=%.4f grad=%.4f",
                 epoch, running["total"]/max(n,1), running["l1"]/max(n,1),
                 running["lumen"]/max(n,1), running["grad"]/max(n,1))
        if epoch % args.ckpt_every == 0 or epoch == n_epochs:
            ckpt = args.ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "args": vars(args) | {"ckpt_dir": str(args.ckpt_dir)}}, ckpt)
            log.info("[ckpt] %s", ckpt)
    log.info("DONE")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
