"""Train U-Net-conditioned posterior residual diffusion v2."""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
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

import torch
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.models.conditional_image_denoiser import ConditionalImageDenoiser  # noqa: E402
from code.models.residual_edm import ResidualEDM  # noqa: E402
from code.models.residual_refiner import ResidualRefinerUNet3D  # noqa: E402
from code.models.residual_unet import ResidualUNet3D  # noqa: E402
from code.training.train_diffusion import (  # noqa: E402
    _smoke_online_motion_factory,
    build_paired_dataloader,
    ema_update,
)
from code.training.train_residual_refiner import load_initializer  # noqa: E402

log = logging.getLogger(__name__)


def build_denoiser(cfg: DictConfig) -> ConditionalImageDenoiser:
    return ConditionalImageDenoiser(
        target_channels=int(cfg.target_channels),
        condition_channels=int(cfg.condition_channels),
        channels=tuple(int(v) for v in cfg.channels),
        num_res_blocks=tuple(int(v) for v in cfg.num_res_blocks),
        attention_levels=tuple(bool(v) for v in cfg.attention_levels),
        num_head_channels=tuple(int(v) for v in cfg.num_head_channels),
        norm_num_groups=int(cfg.norm_num_groups),
    )


def build_engine(cfg: DictConfig, denoiser: ConditionalImageDenoiser) -> ResidualEDM:
    ec = cfg.edm
    return ResidualEDM(
        denoiser=denoiser,
        target_channels=int(cfg.denoiser.target_channels),
        condition_channels=int(cfg.denoiser.condition_channels),
        sigma_min=float(ec.sigma_min),
        sigma_max=float(ec.sigma_max),
        sigma_data=float(ec.sigma_data),
        rho=float(ec.rho),
        P_mean=float(ec.P_mean),
        P_std=float(ec.P_std),
        S_churn=float(ec.S_churn),
        S_tmin=float(ec.S_tmin),
        S_tmax=float(ec.S_tmax),
        S_noise=float(ec.S_noise),
        clip_pred=bool(ec.clip_pred),
        clip_value=float(ec.clip_value),
        final_l1_weight=float(ec.final_l1_weight),
        residual_l1_weight=float(ec.residual_l1_weight),
        gradient_weight=float(ec.gradient_weight),
    )


def run_epoch(
    epoch: int,
    engine: ResidualEDM,
    initializer: ResidualUNet3D,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    ema_denoiser: Optional[torch.nn.Module],
    smoke: bool,
    limit_batches: Optional[int] = None,
    wandb_run=None,
) -> dict[str, float]:
    engine.train()
    initializer.eval()
    n_batches = 0
    running: dict[str, float] = {
        "loss/total": 0.0,
        "loss/edm": 0.0,
        "loss/residual_l1": 0.0,
        "loss/final_l1": 0.0,
        "loss/gradient": 0.0,
        "sigma/mean": 0.0,
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
                condition = ResidualRefinerUNet3D.make_condition(corrupted, initial)
                residual_target = clean - initial
                loss, metrics = engine.loss(
                    residual_target=residual_target,
                    condition=condition,
                    initial=initial,
                    clean=clean,
                )
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
            condition = ResidualRefinerUNet3D.make_condition(corrupted, initial)
            residual_target = clean - initial
            loss, metrics = engine.loss(
                residual_target=residual_target,
                condition=condition,
                initial=initial,
                clean=clean,
            )
            (loss / accum).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if should_step and ema_denoiser is not None and cfg.ema.enabled:
            ema_update(ema_denoiser, engine.denoiser, decay=cfg.ema.decay)

        for key, value in metrics.items():
            running[key] += value
        n_batches += 1

        if step % cfg.train.log_every_steps == 0:
            log.info(
                "epoch %d step %d  loss=%.5f edm=%.5f final_l1=%.5f grad=%.5f sigma=%.4f",
                epoch,
                step,
                metrics["loss/total"],
                metrics["loss/edm"],
                metrics["loss/final_l1"],
                metrics["loss/gradient"],
                metrics["sigma/mean"],
            )
        if wandb_run is not None:
            wandb_run.log({**metrics, "epoch": epoch})

    return {f"{key}_mean": value / max(1, n_batches) for key, value in running.items()}


def save_checkpoint(
    engine: ResidualEDM,
    optimizer: torch.optim.Optimizer,
    ema_denoiser: Optional[torch.nn.Module],
    epoch: int,
    ckpt_dir: Path,
    cfg: DictConfig,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "denoiser": engine.denoiser.state_dict(),
            "ema_denoiser": ema_denoiser.state_dict() if ema_denoiser is not None else None,
            "optim": optimizer.state_dict(),
            "denoiser_cfg": OmegaConf.to_container(cfg.denoiser, resolve=True),
            "engine": "residual_edm",
            "engine_cfg": OmegaConf.to_container(cfg.edm, resolve=True),
            "initializer_cfg": OmegaConf.to_container(cfg.initializer, resolve=True),
        },
        out,
    )
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train U-Net-conditioned posterior residual EDM.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true", help="CPU smoke: tiny denoiser, 1 epoch x 4 batches.")
    p.add_argument("--ckpt", type=Path, default=None, help="resume checkpoint")
    p.add_argument("--epochs", type=int, default=None, help="override train.n_epochs")
    p.add_argument("--limit-train-batches", type=int, default=None, help="cap batches per epoch")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="override train.ckpt_dir")
    p.add_argument("--run-slug", type=str, default=None, help="override run.slug")
    p.add_argument("--no-wandb", action="store_true", help="disable WandB for this run")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _build_argparser().parse_args(argv)
    cfg: DictConfig = OmegaConf.load(args.config)
    smoke = bool(args.smoke)
    if args.run_slug is not None:
        cfg.run.slug = args.run_slug
    if args.ckpt_dir is not None:
        cfg.train.ckpt_dir = str(args.ckpt_dir)
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
        cfg.denoiser.channels = [16, 16, 16, 16]
        cfg.denoiser.num_res_blocks = [1, 1, 1, 1]
        cfg.denoiser.attention_levels = [False, False, False, False]
        cfg.denoiser.num_head_channels = [16, 16, 16, 16]
        cfg.denoiser.norm_num_groups = 8
        cfg.data.pair_mode = "online"
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        log.info("[smoke] forcing CPU, tiny denoiser, 32^3 patches, online noise pairs")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    initializer = load_initializer(cfg.initializer, device)
    denoiser = build_denoiser(cfg.denoiser).to(device)
    engine = build_engine(cfg, denoiser).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    log.info("[model] ConditionalImageDenoiser params=%.2fM device=%s", n_params / 1e6, device)

    online_factory = _smoke_online_motion_factory if smoke and cfg.data.pair_mode == "online" else None
    loader = build_paired_dataloader(cfg, smoke, online_motion_factory=online_factory)

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
        ckpt = torch.load(args.ckpt, map_location=device)
        denoiser.load_state_dict(ckpt["denoiser"])
        optimizer.load_state_dict(ckpt["optim"])
        if ema_denoiser is not None and ckpt.get("ema_denoiser") is not None:
            ema_denoiser.load_state_dict(ckpt["ema_denoiser"])
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
            engine,
            initializer,
            loader,
            optimizer,
            cfg,
            device,
            scaler,
            ema_denoiser,
            smoke,
            limit_batches=args.limit_train_batches,
            wandb_run=wandb_run,
        )
        log.info("[epoch %d] mean train: %s", epoch, metrics)
        if (not smoke) and (epoch % cfg.train.ckpt_every_epochs == 0):
            out = save_checkpoint(engine, optimizer, ema_denoiser, epoch, ckpt_dir, cfg)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(engine, optimizer, ema_denoiser, 0, ckpt_dir / "smoke", cfg)
        log.info("[smoke] checkpoint at %s", out)
    elif last_epoch > 0 and last_epoch % cfg.train.ckpt_every_epochs != 0:
        out = save_checkpoint(engine, optimizer, ema_denoiser, last_epoch, ckpt_dir, cfg)
        log.info("[ckpt final] %s", out)

    log.info("DONE in %.1fs (last_epoch=%d)", time.time() - t0, last_epoch)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
