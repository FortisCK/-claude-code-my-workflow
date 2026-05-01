"""train_diffusion.py — Stage-2 conditional EDM training (latent diffusion).

The VAE is loaded frozen and encodes (V_clean, V_corrupted) to (z_clean, z_cond)
on the fly; the EDM denoiser learns p(z_clean | z_cond) via concat conditioning.

Usage:
    # Real training (GPU):
    python -m code.training.train_diffusion \\
        --config code/training/configs/diffusion_v1.yaml

    # CPU smoke (1 epoch × 4 batches, no AMP, no WandB, tiny denoiser):
    python -m code.training.train_diffusion \\
        --config code/training/configs/diffusion_v1.yaml --smoke

Inputs:
    - VAE checkpoint (Stage-1)
    - ImageCASPairedDataset (precomputed pair files OR online motion synth)

Outputs:
    - Checkpoints at `experiments/checkpoints/<run-slug>/epoch_<N>.pt`
    - EMA copy at the same path with `_ema.pt` suffix
    - Resolved config at `experiments/runs/<run-slug>/config_resolved.yaml`

Adapted from: external/HM-EDM/diffusion_models/conditional_EDM_3D.py
              `Trainer` class (MIT, lucidrains) + MONAI 3D LDM tutorial.

Per `.claude/rules/python-code-conventions.md`:
    - set_determinism once at top
    - relative paths via `code.data.paths`
    - logging not print
    - run-card slug referenced in header
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.data import paths as path_registry  # noqa: E402
from code.data.imagecas_dataset import ImageCASPairedDataset  # noqa: E402
from code.data.transforms import diffusion_train_transforms  # noqa: E402
from code.models.conditional_denoiser import ConditionalDenoiser  # noqa: E402
from code.models.edm import EDM  # noqa: E402
from code.models.vae import CardiacVAE  # noqa: E402

log = logging.getLogger(__name__)


# ============================================================
# Smoke-mode online motion stub (no GPU needed)
# ============================================================


def _smoke_online_motion_factory(case_id: str) -> dict:
    """Return clean + noise-corrupted copy from a single case_<id>.npz.

    Stand-in for the real motion-synth pipeline (which needs GPU + tomosipo).
    Used only in `--smoke`. Real training uses precomputed pair files OR a
    GPU-backed online factory plugged in via the dataloader builder.
    """
    import numpy as np
    npz = np.load(
        path_registry.get("IMAGECAS_PROCESSED") / f"case_{case_id}.npz",
        allow_pickle=True,
    )
    volume = torch.from_numpy(npz["volume"]).unsqueeze(0).float()
    heart_mask = torch.from_numpy(npz["heart_mask"]).unsqueeze(0).byte()
    corrupted = volume + 0.05 * torch.randn_like(volume)
    return {
        "volume": volume,
        "corrupted": corrupted,
        "heart_mask": heart_mask,
        "case_id": case_id,
        "motion_params": {"stub": True},
    }


# ============================================================
# VAE loading (frozen)
# ============================================================


def load_frozen_vae(vae_cfg_path: Path, ckpt_path: Path, device: torch.device) -> CardiacVAE:
    """Reconstruct the VAE from its training config + state_dict, freeze, eval.

    Prefers the `model_cfg` payload embedded in the checkpoint (so smoke and
    real checkpoints with different architectures both load correctly);
    falls back to the YAML config otherwise.
    """
    state = torch.load(ckpt_path, map_location=device)
    if "model_cfg" in state:
        m = state["model_cfg"]
        log.info("[vae] using model_cfg embedded in checkpoint")
    else:
        m = OmegaConf.load(vae_cfg_path).model
        log.info("[vae] using model_cfg from %s", vae_cfg_path)
    vae = CardiacVAE(
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        num_res_blocks=tuple(m["num_res_blocks"]),
        attention_levels=tuple(m["attention_levels"]),
        latent_channels=m["latent_channels"],
        norm_num_groups=m["norm_num_groups"],
    ).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    log.info("[vae] loaded frozen from %s @ epoch %d", ckpt_path, state.get("epoch", -1))
    return vae


# ============================================================
# Data
# ============================================================


def discover_paired_case_ids(processed_dir: Path) -> list[str]:
    """Cases with at least one precomputed pair file."""
    return sorted({p.stem.split("__pair_")[0].removeprefix("case_")
                   for p in processed_dir.glob("case_*__pair_*.npz")})


def discover_clean_case_ids(processed_dir: Path) -> list[str]:
    return sorted(p.stem.removeprefix("case_")
                  for p in processed_dir.glob("case_*.npz")
                  if "__pair_" not in p.name)


def split_train_val(case_ids: list[str], val_split: float, seed: int) -> tuple[list[str], list[str]]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(case_ids), generator=g).tolist()
    n_val = max(1, int(round(val_split * len(case_ids))))
    val_ids = [case_ids[i] for i in perm[:n_val]]
    train_ids = [case_ids[i] for i in perm[n_val:]]
    return train_ids, val_ids


def build_paired_dataloader(
    cfg: DictConfig, smoke: bool, online_motion_factory=None
) -> DataLoader:
    proc_dir = (
        Path(cfg.data.processed_dir) if cfg.data.processed_dir
        else path_registry.get("IMAGECAS_PROCESSED")
    )

    if cfg.data.case_ids_train is None:
        if cfg.data.pair_mode == "precomputed":
            ids = discover_paired_case_ids(proc_dir)
            if not ids:
                raise FileNotFoundError(
                    f"No precomputed pair files at {proc_dir}; "
                    f"run scripts/python/generate_motion_pairs.py first or "
                    f"set data.pair_mode='online'."
                )
        else:
            ids = discover_clean_case_ids(proc_dir)
        train_ids, val_ids = split_train_val(ids, cfg.data.val_split, cfg.run.seed)
    else:
        train_ids = list(cfg.data.case_ids_train)
        val_ids = []

    if smoke:
        pool = (train_ids + val_ids) or list(cfg.data.case_ids_train or [])
        if not pool:
            raise FileNotFoundError(f"smoke: no usable cases in {proc_dir}")
        target_len = cfg.train.batch_size * 4
        while len(pool) < target_len:
            pool = pool + pool
        train_ids = pool[:target_len]

    aug = cfg.augmentation
    transform = diffusion_train_transforms(
        p_flip=aug.p_flip,
        rotate_range_deg=aug.rotate_range_deg,
        shift_range_voxels=aug.shift_range_voxels,
    )

    ds = ImageCASPairedDataset(
        case_ids=train_ids,
        processed_dir=proc_dir,
        transform=transform,
        mode=cfg.data.pair_mode,
        online_motion_factory=online_motion_factory,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0 if smoke else cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and not smoke,
        persistent_workers=cfg.data.persistent_workers and not smoke and cfg.data.num_workers > 0,
        drop_last=True,
    )
    log.info("[data] paired cases=%d  mode=%s", len(ds), cfg.data.pair_mode)
    return loader


# ============================================================
# EMA
# ============================================================


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


# ============================================================
# Train loop
# ============================================================


def run_epoch(
    epoch: int,
    edm: EDM,
    vae: CardiacVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    ema_denoiser: Optional[torch.nn.Module],
    smoke: bool,
    wandb_run=None,
) -> dict[str, float]:
    edm.train()
    vae.eval()
    n_batches = 0
    running = 0.0
    n_max = 4 if smoke else len(loader)

    for step, batch in enumerate(loader):
        if smoke and step >= n_max:
            break
        v_clean = batch["volume"].to(device, non_blocking=True)
        v_cond = batch["corrupted"].to(device, non_blocking=True)

        with torch.no_grad():
            z_clean = vae.encode_to_latent(v_clean)
            z_cond = vae.encode_to_latent(v_cond)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = edm.loss(z_clean, z_cond)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = edm.loss(z_clean, z_cond)
            loss.backward()
            optimizer.step()

        if ema_denoiser is not None and cfg.ema.enabled:
            ema_update(ema_denoiser, edm.denoiser, decay=cfg.ema.decay)

        running += float(loss.detach())
        n_batches += 1

        if step % cfg.train.log_every_steps == 0:
            log.info("epoch %d step %d  edm_loss=%.4f", epoch, step, float(loss.detach()))
        if wandb_run is not None:
            wandb_run.log({"loss/edm": float(loss.detach()), "epoch": epoch})

    return {"loss/edm_mean": running / max(1, n_batches)}


def save_checkpoint(
    edm: EDM, optim: torch.optim.Optimizer, ema_denoiser: Optional[torch.nn.Module],
    epoch: int, ckpt_dir: Path, denoiser_cfg: Optional[DictConfig] = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    payload: dict = {
        "epoch": epoch,
        "denoiser": edm.denoiser.state_dict(),
        "optim": optim.state_dict(),
        "ema_denoiser": ema_denoiser.state_dict() if ema_denoiser is not None else None,
    }
    if denoiser_cfg is not None:
        payload["denoiser_cfg"] = OmegaConf.to_container(denoiser_cfg, resolve=True)
    torch.save(payload, out)
    return out


# ============================================================
# Entry-point
# ============================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train cardiac conditional EDM (Stage 2).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke: 1 epoch × 4 batches, tiny denoiser, no AMP, no WandB.")
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
        # Tiny denoiser so CPU forward at 24³ × 8 ch is seconds.
        cfg.denoiser.channels = [32, 32, 32, 32]
        cfg.denoiser.num_res_blocks = [1, 1, 1, 1]
        cfg.denoiser.attention_levels = [False, False, False, False]
        # MONAI's middle block always uses attention, so num_head_channels
        # at the last level must divide the last channel count (32 % 32 = 0).
        cfg.denoiser.num_head_channels = [32, 32, 32, 32]
        # Force online pairs in smoke (we typically don't have precomputed
        # pair files when running this on CPU without GPU motion synth).
        cfg.data.pair_mode = "online"
        log.info(
            "[smoke] forcing device=cpu, amp=false, wandb=off, n_epochs=1, "
            "denoiser=tiny, pair_mode=online"
        )
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    log.info("[setup] device=%s  slug=%s", device, cfg.run.slug)

    # ---- Frozen VAE ----
    vae_ckpt = Path(cfg.vae.smoke_checkpoint) if smoke and "smoke_checkpoint" in cfg.vae \
               else Path(cfg.vae.checkpoint)
    vae = load_frozen_vae(Path(cfg.vae.config), vae_ckpt, device)

    # ---- Data ----
    online_factory = None
    if smoke and cfg.data.pair_mode == "online":
        online_factory = _smoke_online_motion_factory  # built-in noise stub
    loader = build_paired_dataloader(cfg, smoke, online_motion_factory=online_factory)

    # ---- Denoiser + EDM ----
    denoiser = ConditionalDenoiser(
        latent_channels=cfg.denoiser.latent_channels,
        channels=tuple(cfg.denoiser.channels),
        num_res_blocks=tuple(cfg.denoiser.num_res_blocks),
        attention_levels=tuple(cfg.denoiser.attention_levels),
        num_head_channels=tuple(cfg.denoiser.num_head_channels),
        norm_num_groups=cfg.denoiser.norm_num_groups,
    ).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    log.info("[model] ConditionalDenoiser params=%.2fM", n_params / 1e6)

    edm = EDM(
        denoiser=denoiser,
        latent_channels=cfg.denoiser.latent_channels,
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

    optimizer = torch.optim.Adam(
        denoiser.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )

    ema_denoiser: Optional[torch.nn.Module] = None
    if cfg.ema.enabled:
        ema_denoiser = copy.deepcopy(denoiser).to(device)
        for p in ema_denoiser.parameters():
            p.requires_grad_(False)

    if args.ckpt is not None:
        ck = torch.load(args.ckpt, map_location=device)
        denoiser.load_state_dict(ck["denoiser"])
        optimizer.load_state_dict(ck["optim"])
        if ema_denoiser is not None and ck.get("ema_denoiser") is not None:
            ema_denoiser.load_state_dict(ck["ema_denoiser"])
        log.info("[resume] loaded %s @ epoch %d", args.ckpt, ck.get("epoch", -1))

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
            epoch, edm, vae, loader, optimizer, cfg, device, scaler,
            ema_denoiser=ema_denoiser, smoke=smoke, wandb_run=wandb_run,
        )
        log.info("[epoch %d] mean train: %s", epoch, train_metrics)
        if (not smoke) and (epoch % cfg.train.ckpt_every_epochs == 0):
            out = save_checkpoint(edm, optimizer, ema_denoiser, epoch, ckpt_dir,
                                   denoiser_cfg=cfg.denoiser)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(edm, optimizer, ema_denoiser, epoch=0,
                                ckpt_dir=ckpt_dir / "smoke", denoiser_cfg=cfg.denoiser)
        log.info("[smoke] checkpoint at %s", out)

    log.info("DONE in %.1fs", time.time() - t0)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
