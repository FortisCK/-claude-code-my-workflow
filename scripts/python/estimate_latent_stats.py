#!/usr/bin/env python
"""Estimate frozen-VAE latent statistics for Stage-2 latent generators.

The EDM wrapper can train directly in the raw VAE latent space, but its
`sigma_data` should match the target latent standard deviation. This script
encodes the fixed training split and reports streaming statistics for
`z_clean`, `z_cond`, and `z_delta = z_clean - z_cond`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from monai.utils import set_determinism
from torch.utils.data import DataLoader

from code.data import paths as path_registry  # noqa: E402
from code.data.imagecas_dataset import ImageCASPairedDataset  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.data.transforms import diffusion_train_transforms  # noqa: E402
from code.training.train_diffusion import load_frozen_vae  # noqa: E402

log = logging.getLogger("estimate_latent_stats")


class LatentStats:
    """Streaming overall and per-channel moments for 5D latent tensors."""

    def __init__(self) -> None:
        self.n_total = 0
        self.sum_total = 0.0
        self.sumsq_total = 0.0
        self.min_total = math.inf
        self.max_total = -math.inf
        self.n_channel: Optional[torch.Tensor] = None
        self.sum_channel: Optional[torch.Tensor] = None
        self.sumsq_channel: Optional[torch.Tensor] = None
        self.min_channel: Optional[torch.Tensor] = None
        self.max_channel: Optional[torch.Tensor] = None

    def update(self, z: torch.Tensor) -> None:
        z_cpu = z.detach().double().cpu()
        c = z_cpu.shape[1]
        flat_channel = z_cpu.permute(1, 0, 2, 3, 4).reshape(c, -1)

        self.n_total += z_cpu.numel()
        self.sum_total += float(z_cpu.sum().item())
        self.sumsq_total += float((z_cpu * z_cpu).sum().item())
        self.min_total = min(self.min_total, float(z_cpu.min().item()))
        self.max_total = max(self.max_total, float(z_cpu.max().item()))

        if self.n_channel is None:
            self.n_channel = torch.zeros(c, dtype=torch.float64)
            self.sum_channel = torch.zeros(c, dtype=torch.float64)
            self.sumsq_channel = torch.zeros(c, dtype=torch.float64)
            self.min_channel = torch.full((c,), math.inf, dtype=torch.float64)
            self.max_channel = torch.full((c,), -math.inf, dtype=torch.float64)

        assert self.n_channel is not None
        assert self.sum_channel is not None
        assert self.sumsq_channel is not None
        assert self.min_channel is not None
        assert self.max_channel is not None

        self.n_channel += flat_channel.shape[1]
        self.sum_channel += flat_channel.sum(dim=1)
        self.sumsq_channel += (flat_channel * flat_channel).sum(dim=1)
        self.min_channel = torch.minimum(self.min_channel, flat_channel.min(dim=1).values)
        self.max_channel = torch.maximum(self.max_channel, flat_channel.max(dim=1).values)

    @staticmethod
    def _mean_std(n: int | torch.Tensor, total: float | torch.Tensor, sumsq: float | torch.Tensor):
        mean = total / n
        var = sumsq / n - mean * mean
        if isinstance(var, torch.Tensor):
            return mean, var.clamp_min(0.0).sqrt()
        return float(mean), float(math.sqrt(max(float(var), 0.0)))

    def as_dict(self) -> dict[str, object]:
        mean, std = self._mean_std(self.n_total, self.sum_total, self.sumsq_total)
        out: dict[str, object] = {
            "n_elements": int(self.n_total),
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min_total),
            "max": float(self.max_total),
        }
        if self.n_channel is not None:
            assert self.sum_channel is not None
            assert self.sumsq_channel is not None
            assert self.min_channel is not None
            assert self.max_channel is not None
            ch_mean, ch_std = self._mean_std(
                self.n_channel, self.sum_channel, self.sumsq_channel
            )
            out["per_channel"] = [
                {
                    "channel": int(i),
                    "n_elements": int(self.n_channel[i].item()),
                    "mean": float(ch_mean[i].item()),
                    "std": float(ch_std[i].item()),
                    "min": float(self.min_channel[i].item()),
                    "max": float(self.max_channel[i].item()),
                }
                for i in range(int(self.n_channel.numel()))
            ]
        return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate VAE latent stats for Stage-2.")
    p.add_argument("--vae-config", type=Path, default=Path("code/training/configs/vae_v2_128.yaml"))
    p.add_argument("--vae-ckpt", type=Path, default=Path("experiments/checkpoints/vae_v2/best_val.pt"))
    p.add_argument("--processed-dir", type=Path, default=None)
    p.add_argument("--split-file", type=Path, default=Path("data/imagecas/splits/v1.json"))
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--max-cases", type=int, default=0, help="0 means all selected split cases.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--p-flip", type=float, default=0.5)
    p.add_argument("--rotate-range-deg", type=float, default=5.0)
    p.add_argument("--shift-range-voxels", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", help="Use CUDA autocast for VAE encoding.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--out-json", type=Path, default=None)
    return p


@torch.inference_mode()
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_determinism(seed=args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    split = load_split(args.split_file)
    case_ids = list(getattr(split, args.split))
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    if not case_ids:
        raise ValueError("no cases selected")

    processed_dir = args.processed_dir or path_registry.get("IMAGECAS_PROCESSED")
    transform = diffusion_train_transforms(
        p_flip=args.p_flip,
        rotate_range_deg=args.rotate_range_deg,
        shift_range_voxels=args.shift_range_voxels,
        patch_size=tuple(int(v) for v in args.patch_size) if args.patch_size else None,
    )
    ds = ImageCASPairedDataset(
        case_ids=case_ids,
        processed_dir=processed_dir,
        transform=transform,
        mode="precomputed",
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    vae = load_frozen_vae(args.vae_config, args.vae_ckpt, device)
    use_amp = bool(args.amp and device.type == "cuda")

    stats = {
        "z_clean": LatentStats(),
        "z_cond": LatentStats(),
        "z_delta": LatentStats(),
    }

    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        v_clean = batch["volume"].to(device, non_blocking=True)
        v_cond = batch["corrupted"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            z_clean = vae.encode_to_latent(v_clean).float()
            z_cond = vae.encode_to_latent(v_cond).float()

        stats["z_clean"].update(z_clean)
        stats["z_cond"].update(z_cond)
        stats["z_delta"].update(z_clean - z_cond)

        if step == 1 or step % args.log_every == 0 or step == len(loader):
            seen = min(step * args.batch_size, len(case_ids))
            log.info("[stats] %d/%d cases encoded", seen, len(case_ids))

    result_stats = {name: acc.as_dict() for name, acc in stats.items()}
    sigma_data = float(result_stats["z_clean"]["std"])
    summary = {
        "vae_config": str(args.vae_config),
        "vae_checkpoint": str(args.vae_ckpt),
        "processed_dir": str(processed_dir),
        "split_file": str(args.split_file),
        "split": args.split,
        "n_cases": len(case_ids),
        "batch_size": args.batch_size,
        "patch_size": list(args.patch_size) if args.patch_size else None,
        "augmentation": {
            "p_flip": args.p_flip,
            "rotate_range_deg": args.rotate_range_deg,
            "shift_range_voxels": args.shift_range_voxels,
        },
        "seed": args.seed,
        "device": str(device),
        "amp": use_amp,
        "elapsed_sec": round(time.time() - t0, 3),
        "recommended": {
            "edm_sigma_data": sigma_data,
            "ldm_scale_factor_if_using_ddpm": 1.0 / sigma_data if sigma_data > 0 else None,
        },
        "stats": result_stats,
    }

    if args.out_json is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_json = Path("experiments/runs/vae_v2") / f"latent_stats_{args.split}_{stamp}.json"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    log.info("[stats] wrote %s", args.out_json)
    log.info("[stats] recommended edm.sigma_data = %.6f", sigma_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
