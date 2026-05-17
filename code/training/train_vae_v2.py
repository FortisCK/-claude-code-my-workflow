"""train_vae_v2.py — MAISI-aligned KL-VAE training (cardiac CT).

Stage-1 self-supervised KL-VAE pretraining with the canonical medical
latent-diffusion loss recipe:

    L_G = L1 + kl_w · KL + perceptual_w · LPIPS_2.5d + adv_w · disc(reconstruction)
    L_D = 0.5 · ( disc(real)→true + disc(fake)→false )

Two networks (autoencoder + PatchDiscriminator), two Adam optimizers, two
GradScalers (`init_scale=2**8, growth_factor=1.5` — GAN-friendly defaults that
avoid the early-iteration NaN spikes of the PyTorch defaults).

LR warmup is mandatory for GAN stability:
    epoch <10 :  lr × 0.01
    epoch <20 :  lr × 0.1
    epoch ≥20 :  lr × 1.0

Validation runs through a `SlidingWindowInferer` (with `dynamic_infer` falling
back to a single forward when the volume already fits in `roi_size`).

Usage:
    # Real training (GPU):
    python -m code.training.train_vae_v2 --config code/training/configs/vae_v2.yaml

    # CPU smoke (1 epoch × 4 batches, tiny model, AMP off, WandB off):
    python -m code.training.train_vae_v2 --config code/training/configs/vae_v2.yaml --smoke

Inputs:
    - ImageCAS preprocessed volumes (`data/imagecas/processed/v1/case_*.npz`)
    - YAML config (see `code/training/configs/vae_v2.yaml`)
    - Frozen split file (`data/imagecas/splits/v1.json`)

Outputs:
    - Generator checkpoints  `experiments/checkpoints/<slug>/epoch_<N>.pt`
    - Discriminator checkpoints `experiments/checkpoints/<slug>/epoch_<N>_disc.pt`
    - Resolved config        `experiments/runs/<slug>/config_resolved.yaml`
    - WandB run named after the slug (when enabled)

Run-card slug placeholder: `experiments/runs/<DATE>_<HHMM>_vae-v2-pretrain.md`

Reference: external/MAISI-v2/train_vae_tutorial.ipynb (Apache-2.0, NVIDIA).

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
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import paths as path_registry  # noqa: E402
from code.data.imagecas_dataset import ImageCASCleanDataset  # noqa: E402
from code.data.splits import load_split  # noqa: E402
from code.data.transforms import vae_v2_train_transforms, vae_v2_val_transforms  # noqa: E402
from code.models.vae import CardiacVAE  # noqa: E402

log = logging.getLogger(__name__)

EPS_FP32 = 1.0e-7


# ============================================================
# Loss helpers
# ============================================================


def kl_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """KL( q(z|x) || N(0, I) ) — MAISI v2 formulation.

    Sum over channel + spatial dims, mean over batch dim. Equivalent to the
    canonical N-dimensional Gaussian KL with isotropic prior.
    """
    eps = EPS_FP32
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1.0,
        dim=list(range(1, z_sigma.dim())),
    )
    return torch.sum(kl) / kl.shape[0]


def dynamic_infer(
    inferer, model: torch.nn.Module, images: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward `model(images)` directly when volume fits in `inferer.roi_size`,
    otherwise delegate to `inferer` (sliding window). VAE returns (rec, mu, sigma).
    """
    if torch.numel(images[0:1, 0:1, ...]) <= math.prod(inferer.roi_size):
        return model(images)
    return inferer(images, model)


# ============================================================
# Data
# ============================================================


def discover_case_ids(processed_dir: Path) -> list[str]:
    return sorted(
        p.stem.removeprefix("case_")
        for p in processed_dir.glob("case_*.npz")
        if "__pair_" not in p.name
    )


def build_dataloaders(
    cfg: DictConfig, smoke: bool
) -> tuple[DataLoader, Optional[DataLoader]]:
    proc_dir = (
        Path(cfg.data.processed_dir) if cfg.data.processed_dir
        else path_registry.get("IMAGECAS_PROCESSED")
    )

    # Resolve train/val ids: explicit override → split_file → fallback discover
    if cfg.data.case_ids_train is not None:
        train_ids = list(cfg.data.case_ids_train)
        val_ids = list(cfg.data.case_ids_val) if cfg.data.case_ids_val else []
    elif cfg.data.get("split_file"):
        split = load_split(Path(cfg.data.split_file))
        train_ids = list(split.train)
        val_ids = list(split.val)
        log.info("[data] using split file %s (test=%d held out)",
                 cfg.data.split_file, len(split.test))
    else:
        all_ids = discover_case_ids(proc_dir)
        if not all_ids:
            raise FileNotFoundError(f"no preprocessed cases at {proc_dir}")
        log.warning("[data] no split_file — using all %d cases as train, no val", len(all_ids))
        train_ids, val_ids = all_ids, []

    if smoke:
        pool = (train_ids + val_ids) or list(cfg.data.case_ids_train or [])
        if not pool:
            raise FileNotFoundError(f"smoke: no usable cases in {proc_dir}")
        target_len = cfg.train.batch_size * 4
        while len(pool) < target_len:
            pool = pool + pool
        train_ids = pool[:target_len]
        val_ids = []

    aug = cfg.augmentation
    train_transform = vae_v2_train_transforms(
        patch_size=tuple(cfg.train.patch_size),
        p_flip=aug.p_flip,
        rotate_range_deg=aug.rotate_range_deg,
        shift_range_voxels=aug.shift_range_voxels,
        intensity_shift_offset=aug.intensity_shift_offset,
    )
    val_transform = vae_v2_val_transforms()

    train_ds = ImageCASCleanDataset(train_ids, processed_dir=proc_dir, transform=train_transform)
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
        val_ds = ImageCASCleanDataset(val_ids, processed_dir=proc_dir, transform=val_transform)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,                       # full 192³ via sliding-window
            shuffle=False,
            num_workers=0 if smoke else cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory and not smoke,
        )

    log.info("[data] train cases=%d  val cases=%d  patch=%s",
             len(train_ds), len(val_ids), list(cfg.train.patch_size))
    return train_loader, val_loader


# ============================================================
# Train loop (one epoch)
# ============================================================


def run_epoch(
    epoch: int,
    autoencoder: CardiacVAE,
    discriminator: PatchDiscriminator,
    loader: DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    intensity_loss: torch.nn.Module,
    perceptual: PerceptualLoss,
    adv_loss: PatchAdversarialLoss,
    cfg: DictConfig,
    device: torch.device,
    scaler_g: Optional[GradScaler],
    scaler_d: Optional[GradScaler],
    smoke: bool,
    wandb_run=None,
) -> dict[str, float]:
    autoencoder.train()
    discriminator.train()
    running: dict[str, float] = {"recon": 0.0, "kl": 0.0, "p": 0.0, "g_adv": 0.0, "d": 0.0}
    n_batches = 0
    n_max = 4 if smoke else len(loader)
    use_amp = scaler_g is not None and device.type == "cuda"

    for step, batch in enumerate(loader):
        if smoke and step >= n_max:
            break
        images = batch["volume"].to(device, non_blocking=True).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp):
            # Generator step: recon + KL + perceptual + adversarial
            reconstruction, z_mu, z_sigma = autoencoder(images)
            recon_l = intensity_loss(reconstruction, images)
            kl_l = kl_loss(z_mu, z_sigma)
            p_l = perceptual(reconstruction.float(), images.float())
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            g_adv_l = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = (
                recon_l
                + cfg.loss.kl_weight * kl_l
                + cfg.loss.perceptual_weight * p_l
                + cfg.loss.adv_weight * g_adv_l
            )

        if use_amp:
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(optimizer_g)
            scaler_g.step(optimizer_g)
            scaler_g.update()
        else:
            loss_g.backward()
            optimizer_g.step()

        with autocast(device.type, enabled=use_amp):
            # Discriminator step: real → 1, fake (detached) → 0
            logits_fake_d = discriminator(reconstruction.contiguous().detach().float())[-1]
            loss_d_fake = adv_loss(logits_fake_d, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().float())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            loss_d = 0.5 * (loss_d_fake + loss_d_real)

        if use_amp:
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            loss_d.backward()
            optimizer_d.step()

        # Aggregate
        running["recon"] += float(recon_l.detach())
        running["kl"]    += float(kl_l.detach())
        running["p"]     += float(p_l.detach())
        running["g_adv"] += float(g_adv_l.detach())
        running["d"]     += float(loss_d.detach())
        n_batches += 1

        if step % cfg.train.log_every_steps == 0:
            log.info(
                "epoch %d step %d  recon=%.4f kl=%.2f p=%.4f g_adv=%.4f d=%.4f",
                epoch, step,
                float(recon_l.detach()), float(kl_l.detach()),
                float(p_l.detach()), float(g_adv_l.detach()), float(loss_d.detach()),
            )
        if wandb_run is not None:
            wandb_run.log({
                "loss/recon":  float(recon_l.detach()),
                "loss/kl":     float(kl_l.detach()),
                "loss/p":      float(p_l.detach()),
                "loss/g_adv":  float(g_adv_l.detach()),
                "loss/d":      float(loss_d.detach()),
                "epoch": epoch,
            })

    return {f"loss/{k}_mean": v / max(1, n_batches) for k, v in running.items()}


# ============================================================
# Validation
# ============================================================


@torch.no_grad()
def run_val(
    epoch: int,
    autoencoder: CardiacVAE,
    loader: DataLoader,
    intensity_loss: torch.nn.Module,
    perceptual: PerceptualLoss,
    val_inferer,
    device: torch.device,
    use_amp: bool,
    wandb_run=None,
) -> dict[str, float]:
    autoencoder.eval()
    losses = {"recon": 0.0, "kl": 0.0, "p": 0.0}
    n = 0
    for batch in loader:
        images = batch["volume"].to(device, non_blocking=True)
        with autocast(device.type, enabled=use_amp):
            reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, autoencoder, images)
            reconstruction = reconstruction.to(device)
            losses["recon"] += float(intensity_loss(reconstruction, images))
            losses["kl"]    += float(kl_loss(z_mu, z_sigma))
            losses["p"]     += float(perceptual(reconstruction.float(), images.float()))
        n += 1
    metrics = {f"val/{k}": v / max(1, n) for k, v in losses.items()}
    log.info("[val epoch %d] %s", epoch, metrics)
    if wandb_run is not None:
        wandb_run.log({**metrics, "epoch": epoch})
    autoencoder.train()
    return metrics


# ============================================================
# Checkpoint
# ============================================================


def save_checkpoint(
    autoencoder: CardiacVAE,
    discriminator: PatchDiscriminator,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int, ckpt_dir: Path,
    model_cfg: Optional[DictConfig] = None,
    disc_cfg: Optional[DictConfig] = None,
) -> tuple[Path, Path]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    g_out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    d_out = ckpt_dir / f"epoch_{epoch:03d}_disc.pt"
    g_payload: dict = {"epoch": epoch, "model": autoencoder.state_dict(), "optim": optimizer_g.state_dict()}
    d_payload: dict = {"epoch": epoch, "model": discriminator.state_dict(), "optim": optimizer_d.state_dict()}
    if model_cfg is not None:
        g_payload["model_cfg"] = OmegaConf.to_container(model_cfg, resolve=True)
    if disc_cfg is not None:
        d_payload["model_cfg"] = OmegaConf.to_container(disc_cfg, resolve=True)
    torch.save(g_payload, g_out)
    torch.save(d_payload, d_out)
    return g_out, d_out


# ============================================================
# Entry-point
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train cardiac KL-VAE v2 (MAISI-aligned, GAN+perceptual).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke: tiny model + 1 epoch × 4 batches, no WandB, no AMP.")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Resume autoencoder from this checkpoint (loads optim state).")
    p.add_argument("--ckpt-disc", type=Path, default=None,
                   help="Resume discriminator from this checkpoint (matched epoch).")
    return p


def _make_warmup_lambda(cfg: DictConfig):
    def warmup(epoch: int) -> float:
        if epoch < 10:
            return float(cfg.optim.warmup_lt10)
        if epoch < 20:
            return float(cfg.optim.warmup_lt20)
        return float(cfg.optim.warmup_else)
    return warmup


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
        cfg.train.patch_size = [32, 32, 32]   # small patches keep CPU forward fast
        cfg.train.batch_size = 1
        cfg.model.channels = [32, 32, 32]
        cfg.model.num_res_blocks = [1, 1, 1]
        cfg.model.attention_levels = [False, False, False]
        cfg.discriminator.channels = 16
        cfg.discriminator.num_layers_d = 2
        cfg.val.sliding_window_patch_size = [32, 32, 32]
        log.info("[smoke] forcing device=cpu, amp=false, wandb=off, tiny model & disc")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    log.info("[setup] device=%s  slug=%s", device, cfg.run.slug)

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(cfg, smoke)

    # ---- Generator (autoencoder) ----
    autoencoder = CardiacVAE(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        channels=tuple(cfg.model.channels),
        num_res_blocks=tuple(cfg.model.num_res_blocks),
        attention_levels=tuple(cfg.model.attention_levels),
        latent_channels=cfg.model.latent_channels,
        norm_num_groups=cfg.model.norm_num_groups,
        use_checkpoint=cfg.model.use_checkpoint,
    ).to(device)
    n_g = sum(p.numel() for p in autoencoder.parameters())
    log.info("[model] CardiacVAE params=%.2fM", n_g / 1e6)

    # ---- Discriminator ----
    discriminator = PatchDiscriminator(
        spatial_dims=cfg.discriminator.spatial_dims,
        num_layers_d=cfg.discriminator.num_layers_d,
        channels=cfg.discriminator.channels,
        in_channels=cfg.discriminator.in_channels,
        out_channels=cfg.discriminator.out_channels,
        norm=cfg.discriminator.norm,
    ).to(device)
    n_d = sum(p.numel() for p in discriminator.parameters())
    log.info("[model] PatchDiscriminator params=%.2fM", n_d / 1e6)

    # ---- Losses ----
    if cfg.loss.recon_type == "l2":
        intensity_loss: torch.nn.Module = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion=cfg.loss.adv_criterion)
    perceptual = PerceptualLoss(
        spatial_dims=3,
        network_type=cfg.loss.perceptual_network_type,
        is_fake_3d=cfg.loss.perceptual_is_fake_3d,
        fake_3d_ratio=cfg.loss.perceptual_fake_3d_ratio,
    ).eval().to(device)

    # ---- Optimizers + lr warmup ----
    eps_amp = float(cfg.optim.amp_eps) if cfg.train.amp else 1e-8
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=cfg.optim.lr,
                                   betas=tuple(cfg.optim.betas),
                                   weight_decay=cfg.optim.weight_decay,
                                   eps=eps_amp)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.optim.lr,
                                   betas=tuple(cfg.optim.betas),
                                   weight_decay=cfg.optim.weight_decay,
                                   eps=eps_amp)
    warmup = _make_warmup_lambda(cfg)
    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup)

    # ---- AMP ----
    scaler_g: Optional[GradScaler] = None
    scaler_d: Optional[GradScaler] = None
    if cfg.train.amp and device.type == "cuda":
        scaler_g = GradScaler("cuda",
                              init_scale=float(cfg.train.amp_init_scale),
                              growth_factor=float(cfg.train.amp_growth_factor))
        scaler_d = GradScaler("cuda",
                              init_scale=float(cfg.train.amp_init_scale),
                              growth_factor=float(cfg.train.amp_growth_factor))

    # ---- Resume ----
    start_epoch = 1
    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location=device)
        autoencoder.load_state_dict(ck["model"])
        optimizer_g.load_state_dict(ck["optim"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        log.info("[resume] G loaded from %s @ epoch %d", args.ckpt, ck.get("epoch", -1))
    if args.ckpt_disc is not None:
        ck = torch.load(args.ckpt_disc, map_location=device)
        discriminator.load_state_dict(ck["model"])
        optimizer_d.load_state_dict(ck["optim"])
        log.info("[resume] D loaded from %s @ epoch %d", args.ckpt_disc, ck.get("epoch", -1))

    # Advance LR scheduler so warmup matches resumed epoch (otherwise warmup
    # restarts from epoch 0 and we waste 20 epochs at reduced lr).
    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler_g.step()
            scheduler_d.step()
        log.info("[resume] scheduler advanced to epoch %d (lr factor %.3f)",
                 start_epoch - 1, warmup(start_epoch - 1))

    # ---- Validation inferer ----
    val_inferer = (
        SlidingWindowInferer(
            roi_size=tuple(cfg.val.sliding_window_patch_size),
            sw_batch_size=cfg.val.sliding_window_sw_batch_size,
            overlap=cfg.val.sliding_window_overlap,
            progress=False,
            device=torch.device("cpu"),  # output stays on CPU to save GPU
            sw_device=device,
        )
        if cfg.val.sliding_window_patch_size
        else SimpleInferer()
    )

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
    use_amp = scaler_g is not None
    best_val_recon = float("inf")        # tracks best val/recon across the run
    best_val_epoch = -1
    t0 = time.time()
    for epoch in range(start_epoch, cfg.train.n_epochs + 1):
        train_metrics = run_epoch(
            epoch, autoencoder, discriminator, train_loader,
            optimizer_g, optimizer_d,
            intensity_loss, perceptual, adv_loss,
            cfg, device, scaler_g, scaler_d, smoke=smoke, wandb_run=wandb_run,
        )
        scheduler_g.step()
        scheduler_d.step()
        log.info("[epoch %d] mean train: %s", epoch, train_metrics)

        if val_loader is not None and (epoch % cfg.train.val_every_epochs == 0):
            val_metrics = run_val(epoch, autoencoder, val_loader, intensity_loss, perceptual,
                                  val_inferer, device, use_amp=use_amp, wandb_run=wandb_run)
            # Save best-val checkpoint when val/recon improves (G only — D not
            # needed for downstream Stage-2 since VAE will be frozen there).
            val_recon = float(val_metrics.get("val/recon", float("inf")))
            if (not smoke) and val_recon < best_val_recon:
                best_val_recon = val_recon
                best_val_epoch = epoch
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                best_out = ckpt_dir / "best_val.pt"
                payload = {
                    "epoch": epoch,
                    "model": autoencoder.state_dict(),
                    "val_recon": val_recon,
                    "val_metrics": val_metrics,
                    "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
                }
                torch.save(payload, best_out)
                log.info("[best-val] new best val/recon=%.5f at epoch %d → %s",
                         val_recon, epoch, best_out)
                if wandb_run is not None:
                    wandb_run.log({"best_val/recon": val_recon, "best_val/epoch": epoch})

        if (not smoke) and (epoch % cfg.train.ckpt_every_epochs == 0):
            g_out, d_out = save_checkpoint(
                autoencoder, discriminator, optimizer_g, optimizer_d, epoch, ckpt_dir,
                model_cfg=cfg.model, disc_cfg=cfg.discriminator,
            )
            log.info("[ckpt] %s  %s", g_out, d_out)

    if best_val_epoch > 0:
        log.info("[summary] best val/recon=%.5f at epoch %d (ckpt: %s)",
                 best_val_recon, best_val_epoch, ckpt_dir / "best_val.pt")

    if smoke:
        g_out, d_out = save_checkpoint(
            autoencoder, discriminator, optimizer_g, optimizer_d, epoch=0,
            ckpt_dir=ckpt_dir / "smoke",
            model_cfg=cfg.model, disc_cfg=cfg.discriminator,
        )
        log.info("[smoke] checkpoints at %s  %s", g_out, d_out)

    log.info("DONE in %.1fs", time.time() - t0)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
