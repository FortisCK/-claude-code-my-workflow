"""Train a deterministic residual refiner on top of frozen U-Net v1.

This is the controlled v2.0 baseline for the posterior-residual roadmap:

    corrupted -> frozen U-Net -> x_u
    [corrupted, x_u, corrupted - x_u] -> residual refiner -> clean - x_u

The script intentionally mirrors `train_unet_baseline.py` so comparisons are
kept clean and the only major behavioral change is the frozen initializer.
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
from code.models.residual_refiner import ResidualRefinerUNet3D  # noqa: E402
from code.models.residual_unet import ResidualUNet3D  # noqa: E402
from code.training.train_diffusion import (  # noqa: E402
    _smoke_online_motion_factory,
    build_paired_dataloader,
    ema_update,
)

log = logging.getLogger(__name__)


def build_refiner(model_cfg: DictConfig) -> ResidualRefinerUNet3D:
    return ResidualRefinerUNet3D(
        in_channels=int(model_cfg.in_channels),
        out_channels=int(model_cfg.out_channels),
        features=tuple(int(v) for v in model_cfg.features),
        clamp_output=bool(model_cfg.clamp_output),
    )


def load_initializer(initializer_cfg: DictConfig, device: torch.device) -> ResidualUNet3D:
    config_path = Path(initializer_cfg.config)
    ckpt_path = Path(initializer_cfg.ckpt)
    cfg = OmegaConf.load(config_path)
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = state.get("model_cfg") or OmegaConf.to_container(cfg.model, resolve=True)
    model = ResidualUNet3D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        features=tuple(model_cfg["features"]),
        residual=model_cfg["residual"],
        clamp_output=model_cfg.get("clamp_output", False),
    ).to(device)
    weights = str(initializer_cfg.get("weights", "ema"))
    if weights == "ema" and state.get("ema_model") is not None:
        model.load_state_dict(state["ema_model"])
        log.info("[initializer] loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
        log.info("[initializer] loaded online weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("[initializer] loaded %s @ epoch %s", ckpt_path, state.get("epoch", "unknown"))
    return model


def refiner_loss(
    final: torch.Tensor,
    clean: torch.Tensor,
    residual_pred: torch.Tensor,
    residual_target: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_l1 = F.l1_loss(final, clean)
    residual_l1 = F.l1_loss(residual_pred, residual_target)
    grad = F.l1_loss(gradient_magnitude(final), gradient_magnitude(clean))
    total = (
        cfg.loss.final_l1_weight * final_l1
        + cfg.loss.residual_l1_weight * residual_l1
        + cfg.loss.gradient_weight * grad
    )
    return total, {
        "loss/total": float(total.detach()),
        "loss/final_l1": float(final_l1.detach()),
        "loss/residual_l1": float(residual_l1.detach()),
        "loss/gradient": float(grad.detach()),
    }


def run_epoch(
    epoch: int,
    model: ResidualRefinerUNet3D,
    initializer: ResidualUNet3D,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    ema_model: Optional[ResidualRefinerUNet3D],
    smoke: bool,
    limit_batches: Optional[int] = None,
    wandb_run=None,
) -> dict[str, float]:
    model.train()
    initializer.eval()
    n_batches = 0
    running: dict[str, float] = {
        "loss/total": 0.0,
        "loss/final_l1": 0.0,
        "loss/residual_l1": 0.0,
        "loss/gradient": 0.0,
    }
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
                with torch.no_grad():
                    initial = initializer(corrupted).detach()
                final, residual_pred = model(corrupted, initial, return_residual=True)
                residual_target = clean - initial
                loss, metrics = refiner_loss(final, clean, residual_pred, residual_target, cfg)
                scaled_loss = loss / accum
            scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                initial = initializer(corrupted).detach()
            final, residual_pred = model(corrupted, initial, return_residual=True)
            residual_target = clean - initial
            loss, metrics = refiner_loss(final, clean, residual_pred, residual_target, cfg)
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
            log.info(
                "epoch %d step %d  loss=%.5f final_l1=%.5f residual_l1=%.5f grad=%.5f",
                epoch,
                step,
                metrics["loss/total"],
                metrics["loss/final_l1"],
                metrics["loss/residual_l1"],
                metrics["loss/gradient"],
            )
        if wandb_run is not None:
            wandb_run.log({**metrics, "epoch": epoch})

    return {f"{key}_mean": value / max(1, n_batches) for key, value in running.items()}


def save_checkpoint(
    model: ResidualRefinerUNet3D,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[ResidualRefinerUNet3D],
    epoch: int,
    ckpt_dir: Path,
    cfg: DictConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model is not None else None,
            "optim": optimizer.state_dict(),
            "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
            "initializer_cfg": OmegaConf.to_container(cfg.initializer, resolve=True),
        },
        out,
    )
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train deterministic residual refiner on frozen U-Net output.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="CPU smoke: tiny refiner, 1 epoch x 4 batches.")
    p.add_argument("--ckpt", type=Path, default=None, help="resume refiner checkpoint")
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
        log.info("[smoke] forcing CPU, tiny refiner, 32^3 patches, online noise pairs")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    initializer = load_initializer(cfg.initializer, device)
    model = build_refiner(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[model] ResidualRefinerUNet3D params=%.2fM device=%s", n_params / 1e6, device)

    online_factory = _smoke_online_motion_factory if smoke and cfg.data.pair_mode == "online" else None
    loader = build_paired_dataloader(cfg, smoke, online_motion_factory=online_factory)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        weight_decay=cfg.optim.weight_decay,
    )
    ema_model: Optional[ResidualRefinerUNet3D] = None
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
            initializer,
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
            out = save_checkpoint(model, optimizer, ema_model, epoch, ckpt_dir, cfg)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(model, optimizer, ema_model, 0, ckpt_dir / "smoke", cfg)
        log.info("[smoke] checkpoint at %s", out)
    elif last_epoch > 0 and last_epoch % cfg.train.ckpt_every_epochs != 0:
        out = save_checkpoint(model, optimizer, ema_model, last_epoch, ckpt_dir, cfg)
        log.info("[ckpt final] %s", out)

    log.info("DONE in %.1fs (last_epoch=%d)", time.time() - t0, last_epoch)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
