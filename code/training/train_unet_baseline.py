"""train_unet_baseline.py — supervised 3D residual U-Net baseline.

The model trains directly in normalized voxel space on the same paired
ImageCAS motion-artifact data used by diffusion_v1:

    corrupted volume -> residual U-Net -> corrected volume

Loss is L1 plus an optional gradient-magnitude L1 term. Checkpoints embed the
model config so evaluation can reconstruct the architecture without relying on
the YAML alone.
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
import torch.nn.functional as F
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import gradient_magnitude  # noqa: E402
from code.models.residual_unet import ResidualUNet3D  # noqa: E402
from code.training.train_diffusion import (  # noqa: E402
    _smoke_online_motion_factory,
    build_paired_dataloader,
    ema_update,
)

log = logging.getLogger(__name__)


def build_model(model_cfg: DictConfig) -> ResidualUNet3D:
    return ResidualUNet3D(
        in_channels=model_cfg.in_channels,
        out_channels=model_cfg.out_channels,
        features=tuple(int(v) for v in model_cfg.features),
        residual=bool(model_cfg.residual),
        clamp_output=bool(model_cfg.clamp_output),
    )


def correction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    l1 = F.l1_loss(pred, target)
    l2 = F.mse_loss(pred, target)
    grad = F.l1_loss(gradient_magnitude(pred), gradient_magnitude(target))
    total = cfg.loss.l1_weight * l1 + cfg.loss.l2_weight * l2 + cfg.loss.gradient_weight * grad
    return total, {
        "loss/total": float(total.detach()),
        "loss/l1": float(l1.detach()),
        "loss/l2": float(l2.detach()),
        "loss/gradient": float(grad.detach()),
    }


def run_epoch(
    epoch: int,
    model: ResidualUNet3D,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    ema_model: Optional[ResidualUNet3D],
    smoke: bool,
    limit_batches: Optional[int] = None,
    wandb_run=None,
) -> dict[str, float]:
    model.train()
    n_batches = 0
    running: dict[str, float] = {"loss/total": 0.0, "loss/l1": 0.0, "loss/l2": 0.0, "loss/gradient": 0.0}
    n_max = 4 if smoke else len(loader)
    if limit_batches is not None:
        n_max = min(n_max, int(limit_batches))
    accum = max(1, int(cfg.train.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if step >= n_max:
            break
        clean = batch["volume"].to(device, non_blocking=True)
        corrupted = batch["corrupted"].to(device, non_blocking=True)

        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(corrupted)
                loss, metrics = correction_loss(pred, clean, cfg)
                scaled_loss = loss / accum
            scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            pred = model(corrupted)
            loss, metrics = correction_loss(pred, clean, cfg)
            (loss / accum).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if should_step and ema_model is not None and cfg.ema.enabled:
            ema_update(ema_model, model, decay=cfg.ema.decay)

        for key, value in metrics.items():
            running[key] += value
        n_batches += 1

        if step % cfg.train.log_every_steps == 0:
            log.info("epoch %d step %d  loss=%.5f l1=%.5f grad=%.5f",
                     epoch, step, metrics["loss/total"], metrics["loss/l1"], metrics["loss/gradient"])
        if wandb_run is not None:
            wandb_run.log({**metrics, "epoch": epoch})

    return {f"{key}_mean": value / max(1, n_batches) for key, value in running.items()}


def save_checkpoint(
    model: ResidualUNet3D,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[ResidualUNet3D],
    epoch: int,
    ckpt_dir: Path,
    model_cfg: DictConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optim": optimizer.state_dict(),
            "model_cfg": OmegaConf.to_container(model_cfg, resolve=True),
        },
        out,
    )
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train supervised 3D residual U-Net baseline.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="CPU smoke: tiny model, 1 epoch x 4 batches.")
    p.add_argument("--ckpt", type=Path, default=None, help="resume from checkpoint")
    p.add_argument("--epochs", type=int, default=None, help="override train.n_epochs")
    p.add_argument("--limit-train-batches", type=int, default=None, help="cap batches per epoch")
    p.add_argument("--no-wandb", action="store_true", help="disable WandB for this run")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    cfg: DictConfig = OmegaConf.load(args.config)
    smoke = bool(args.smoke)
    if args.epochs is not None:
        cfg.train.n_epochs = int(args.epochs)
    if args.no_wandb:
        cfg.wandb.enabled = False
    set_determinism(seed=cfg.run.seed)

    if smoke:
        device = torch.device("cpu")
        cfg.train.amp = False
        cfg.wandb.enabled = False
        cfg.train.n_epochs = 1
        cfg.train.batch_size = 1
        cfg.train.patch_size = [32, 32, 32]
        cfg.model.features = [8, 8, 16, 32, 64, 8]
        cfg.data.pair_mode = "online"
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        log.info("[smoke] forcing CPU, tiny model, 32^3 patches, online noise pairs")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[model] ResidualUNet3D params=%.2fM device=%s", n_params / 1e6, device)

    online_factory = _smoke_online_motion_factory if smoke and cfg.data.pair_mode == "online" else None
    loader = build_paired_dataloader(cfg, smoke, online_motion_factory=online_factory)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    ema_model: Optional[ResidualUNet3D] = None
    if cfg.ema.enabled:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        if ema_model is not None and ckpt.get("ema_model") is not None:
            ema_model.load_state_dict(ckpt["ema_model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        log.info("[resume] loaded %s @ epoch %d", args.ckpt, ckpt.get("epoch", -1))
    else:
        start_epoch = 1

    scaler: Optional[torch.amp.GradScaler] = None
    if cfg.train.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    wandb_run = None
    if cfg.wandb.enabled:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.run.slug,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    runs_dir = Path("experiments/runs") / cfg.run.slug
    runs_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, runs_dir / "config_resolved.yaml")

    t0 = time.time()
    ckpt_dir = Path(cfg.train.ckpt_dir)
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.train.n_epochs + 1):
        last_epoch = epoch
        metrics = run_epoch(
            epoch,
            model,
            loader,
            optimizer,
            cfg,
            device,
            scaler,
            ema_model,
            smoke,
            limit_batches=args.limit_train_batches,
            wandb_run=wandb_run,
        )
        log.info("[epoch %d] mean train: %s", epoch, metrics)
        if (not smoke) and (epoch % cfg.train.ckpt_every_epochs == 0):
            out = save_checkpoint(model, optimizer, ema_model, epoch, ckpt_dir, cfg.model)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(model, optimizer, ema_model, 0, ckpt_dir / "smoke", cfg.model)
        log.info("[smoke] checkpoint at %s", out)
    elif last_epoch > 0 and last_epoch % cfg.train.ckpt_every_epochs != 0:
        out = save_checkpoint(model, optimizer, ema_model, last_epoch, ckpt_dir, cfg.model)
        log.info("[ckpt final] %s", out)

    log.info("DONE in %.1fs (last_epoch=%d)", time.time() - t0, last_epoch)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
