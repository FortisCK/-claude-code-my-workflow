"""train_vae.py — Stage-1 KL-VAE training on ImageCAS preprocessed cache.

Usage:
    # Real training (GPU):
    python -m code.training.train_vae --config code/training/configs/vae_v1.yaml

    # CPU smoke test (1 epoch × 4 batches, no WandB, no AMP):
    python -m code.training.train_vae --config code/training/configs/vae_v1.yaml --smoke

Inputs:
    - ImageCAS preprocessed volumes from `data/imagecas/processed/v1/case_*.npz`
    - YAML config (see `code/training/configs/vae_v1.yaml`)

Outputs:
    - Checkpoints at `experiments/checkpoints/<run-slug>/epoch_<N>.pt`
    - Resolved config at `experiments/runs/<run-slug>/config_resolved.yaml`
    - WandB run named after the slug (when enabled)

Loss:
    L = L1(v, v_rec) + ssim_w * (1 - SSIM(v, v_rec)) + kl_w * KL(N(mu, sigma) || N(0, I))

Adapted from: external/GenerativeModels/tutorials/generative/3d_autoencoderkl/
              3d_autoencoderkl_tutorial.py training loop. Adversarial / perceptual
              losses are intentionally dropped for v1 (faster to debug; revisit
              if recon quality is insufficient).

Per `.claude/rules/python-code-conventions.md`:
    - set_determinism once at top
    - relative paths via `code.data.paths`
    - logging not print
    - AMP via torch.autocast (not apex)
    - run-card slug referenced in header
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from monai.losses import SSIMLoss
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# Repo-root import shim (when invoked as a script rather than a module)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import paths as path_registry  # noqa: E402
from code.data.imagecas_dataset import ImageCASCleanDataset  # noqa: E402
from code.data.transforms import vae_train_transforms  # noqa: E402
from code.models.vae import CardiacVAE  # noqa: E402

log = logging.getLogger(__name__)

EPS_FP16 = 1.0e-4
EPS_FP32 = 1.0e-7


# ============================================================
# Loss
# ============================================================


def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor, eps: float = EPS_FP32) -> torch.Tensor:
    """KL( N(mu, sigma^2) || N(0, I) ) — per-element mean."""
    var = sigma.pow(2).clamp_min(eps)
    logvar = var.log()
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - var)


def vae_loss(
    v: torch.Tensor,
    v_rec: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    ssim_loss_fn: SSIMLoss,
    kl_weight: float,
    ssim_weight: float,
    recon_type: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if recon_type == "l1":
        recon = torch.nn.functional.l1_loss(v_rec, v)
    elif recon_type == "l2":
        recon = torch.nn.functional.mse_loss(v_rec, v)
    else:
        raise ValueError(f"recon_type must be 'l1' or 'l2', got {recon_type!r}")

    # MONAI SSIMLoss returns 1 - SSIM (already a loss); needs values in [0, 1]
    # Our volumes are in [-1, 1]; rescale before SSIM, leave the trainable
    # tensors untouched.
    v_01 = (v + 1.0) * 0.5
    rec_01 = (v_rec + 1.0) * 0.5
    ssim_l = ssim_loss_fn(rec_01, v_01)

    kl = kl_divergence(mu, sigma)

    total = recon + ssim_weight * ssim_l + kl_weight * kl
    return total, {
        "loss/total": float(total.detach()),
        "loss/recon": float(recon.detach()),
        "loss/ssim": float(ssim_l.detach()),
        "loss/kl": float(kl.detach()),
    }


# ============================================================
# Data
# ============================================================


def discover_case_ids(processed_dir: Path) -> list[str]:
    """List case IDs available in the preprocessed cache (case_<id>.npz)."""
    return sorted(
        p.stem.removeprefix("case_")
        for p in processed_dir.glob("case_*.npz")
        if "__pair_" not in p.name
    )


def split_train_val(case_ids: list[str], val_split: float, seed: int) -> tuple[list[str], list[str]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(case_ids), generator=g).tolist()
    n_val = max(1, int(round(val_split * len(case_ids))))
    val_ids = [case_ids[i] for i in perm[:n_val]]
    train_ids = [case_ids[i] for i in perm[n_val:]]
    return train_ids, val_ids


def build_dataloaders(
    cfg: DictConfig, smoke: bool
) -> tuple[DataLoader, Optional[DataLoader]]:
    proc_dir = (
        Path(cfg.data.processed_dir) if cfg.data.processed_dir
        else path_registry.get("IMAGECAS_PROCESSED")
    )

    if cfg.data.case_ids_train is None:
        all_ids = discover_case_ids(proc_dir)
        if not all_ids:
            raise FileNotFoundError(f"No preprocessed cases at {proc_dir}")
        train_ids, val_ids = split_train_val(all_ids, cfg.data.val_split, cfg.run.seed)
    else:
        train_ids = list(cfg.data.case_ids_train)
        val_ids = list(cfg.data.case_ids_val) if cfg.data.case_ids_val else []

    if smoke:
        # In smoke mode pool train+val (we don't run validation anyway)
        # so that even a single case on disk produces enough batches.
        pool = (train_ids + val_ids) or list(cfg.data.case_ids_train or [])
        if not pool:
            raise FileNotFoundError(f"smoke: no preprocessed cases in {proc_dir}")
        target_len = cfg.train.batch_size * 4
        # Pad by repetition: 1 case → batch_size*4 entries
        while len(pool) < target_len:
            pool = pool + pool
        train_ids = pool[:target_len]
        val_ids = []

    aug = cfg.augmentation
    transform = vae_train_transforms(
        p_flip=aug.p_flip,
        rotate_range_deg=aug.rotate_range_deg,
        shift_range_voxels=aug.shift_range_voxels,
        intensity_shift_offset=aug.intensity_shift_offset,
    )

    train_ds = ImageCASCleanDataset(train_ids, processed_dir=proc_dir, transform=transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0 if smoke else cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and not smoke,
        persistent_workers=cfg.data.persistent_workers and not smoke and cfg.data.num_workers > 0,
        drop_last=True,
    )

    val_loader: Optional[DataLoader] = None
    if val_ids:
        val_ds = ImageCASCleanDataset(val_ids, processed_dir=proc_dir, transform=None)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0 if smoke else cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and not smoke,
        )

    log.info("[data] train cases=%d  val cases=%d", len(train_ds), len(val_ids) if val_ids else 0)
    return train_loader, val_loader


# ============================================================
# Train loop
# ============================================================


def run_epoch(
    epoch: int,
    vae: CardiacVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ssim_loss_fn: SSIMLoss,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    smoke: bool,
    wandb_run=None,
) -> dict[str, float]:
    vae.train()
    running: dict[str, float] = {}
    n_batches = 0
    n_max = 4 if smoke else len(loader)

    for step, batch in enumerate(loader):
        if smoke and step >= n_max:
            break
        v = batch["volume"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                v_rec, mu, sigma = vae(v)
                loss, metrics = vae_loss(
                    v, v_rec, mu, sigma, ssim_loss_fn,
                    kl_weight=cfg.loss.kl_weight,
                    ssim_weight=cfg.loss.ssim_weight,
                    recon_type=cfg.loss.recon_type,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            v_rec, mu, sigma = vae(v)
            loss, metrics = vae_loss(
                v, v_rec, mu, sigma, ssim_loss_fn,
                kl_weight=cfg.loss.kl_weight,
                ssim_weight=cfg.loss.ssim_weight,
                recon_type=cfg.loss.recon_type,
            )
            loss.backward()
            optimizer.step()

        for k, x in metrics.items():
            running[k] = running.get(k, 0.0) + x
        n_batches += 1

        if step % cfg.train.log_every_steps == 0:
            log.info(
                "epoch %d step %d  total=%.4f  recon=%.4f  ssim=%.4f  kl=%.2f",
                epoch, step, metrics["loss/total"], metrics["loss/recon"],
                metrics["loss/ssim"], metrics["loss/kl"],
            )
        if wandb_run is not None:
            wandb_run.log({**metrics, "epoch": epoch})

    return {k: v / max(1, n_batches) for k, v in running.items()}


def save_checkpoint(
    vae: CardiacVAE, optim: torch.optim.Optimizer, epoch: int,
    ckpt_dir: Path, model_cfg: Optional[DictConfig] = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    payload: dict = {"epoch": epoch, "model": vae.state_dict(), "optim": optim.state_dict()}
    if model_cfg is not None:
        payload["model_cfg"] = OmegaConf.to_container(model_cfg, resolve=True)
    torch.save(payload, out)
    return out


# ============================================================
# Entry-point
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train cardiac KL-VAE (Stage 1).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke run: 1 epoch × 4 batches, no WandB, no AMP.")
    p.add_argument("--ckpt", type=Path, default=None, help="resume from checkpoint")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)

    cfg: DictConfig = OmegaConf.load(args.config)
    smoke = bool(args.smoke)
    set_determinism(seed=cfg.run.seed)

    if smoke:
        device = torch.device("cpu")
        cfg.train.amp = False
        cfg.wandb.enabled = False
        cfg.train.n_epochs = 1
        # Tiny model so a CPU forward through 192³ completes in seconds.
        # All channels remain multiples of norm_num_groups (32).
        cfg.model.channels = [32, 32, 32, 32]
        cfg.model.num_res_blocks = [1, 1, 1, 1]
        cfg.model.attention_levels = [False, False, False, False]
        log.info(
            "[smoke] forcing device=cpu, amp=false, wandb=off, n_epochs=1, "
            "model=tiny (channels=[32]*4, no attn)"
        )
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    log.info("[setup] device=%s  slug=%s", device, cfg.run.slug)

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(cfg, smoke)

    # ---- Model ----
    vae = CardiacVAE(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        channels=tuple(cfg.model.channels),
        num_res_blocks=tuple(cfg.model.num_res_blocks),
        attention_levels=tuple(cfg.model.attention_levels),
        latent_channels=cfg.model.latent_channels,
        norm_num_groups=cfg.model.norm_num_groups,
        use_checkpoint=cfg.model.use_checkpoint,
    ).to(device)
    n_params = sum(p.numel() for p in vae.parameters())
    log.info("[model] CardiacVAE params=%.2fM", n_params / 1e6)

    optimizer = torch.optim.Adam(
        vae.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location=device)
        vae.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        log.info("[resume] loaded %s @ epoch %d", args.ckpt, ck.get("epoch", -1))

    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0).to(device)

    scaler: Optional[torch.amp.GradScaler] = None
    if cfg.train.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ---- WandB ----
    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(project=cfg.wandb.project, name=cfg.run.slug,
                               config=OmegaConf.to_container(cfg, resolve=True))

    # ---- Resolved config dump ----
    runs_dir = Path("experiments/runs") / cfg.run.slug
    runs_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, runs_dir / "config_resolved.yaml")

    # ---- Train ----
    ckpt_dir = Path(cfg.train.ckpt_dir)
    t0 = time.time()
    for epoch in range(1, cfg.train.n_epochs + 1):
        train_metrics = run_epoch(
            epoch, vae, train_loader, optimizer, ssim_loss_fn, cfg, device, scaler,
            smoke=smoke, wandb_run=wandb_run,
        )
        log.info("[epoch %d] mean train: %s", epoch, train_metrics)

        if (not smoke) and (epoch % cfg.train.ckpt_every_epochs == 0):
            out = save_checkpoint(vae, optimizer, epoch, ckpt_dir, model_cfg=cfg.model)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(vae, optimizer, epoch=0, ckpt_dir=ckpt_dir / "smoke",
                               model_cfg=cfg.model)
        log.info("[smoke] checkpoint at %s", out)

    log.info("DONE in %.1fs", time.time() - t0)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
