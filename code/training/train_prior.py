"""train_prior.py — train the unconditional clean cardiac 3D EDM image prior p(x).

For the sparse-view CCTA reconstruction pivot: DPS needs an unconditional image prior
(residual_edm cannot serve). Trains UnconditionalEDM over clean ImageCAS volumes (patch-cropped),
same data/split/EMA/checkpoint conventions as the other trainers.

Usage:
    python -m code.training.train_prior --config code/training/configs/prior_edm_v1.yaml
    python -m code.training.train_prior --config ... --smoke

Run card: experiments/runs/2026-07-03_1600_prior-edm-v1.md
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.models.conditional_image_denoiser import ConditionalImageDenoiser  # noqa: E402
from code.models.unconditional_edm import UnconditionalEDM  # noqa: E402
from code.training.train_diffusion import build_paired_dataloader, ema_update  # noqa: E402

log = logging.getLogger(__name__)


def build_engine(cfg: DictConfig) -> UnconditionalEDM:
    dn = cfg.denoiser
    denoiser = ConditionalImageDenoiser(
        target_channels=int(dn.target_channels), condition_channels=int(dn.condition_channels),
        channels=tuple(int(v) for v in dn.channels), num_res_blocks=tuple(int(v) for v in dn.num_res_blocks),
        attention_levels=tuple(bool(v) for v in dn.attention_levels),
        num_head_channels=tuple(int(v) for v in dn.num_head_channels), norm_num_groups=int(dn.norm_num_groups),
    )
    e = cfg.edm
    return UnconditionalEDM(denoiser, sigma_min=float(e.sigma_min), sigma_max=float(e.sigma_max),
                            sigma_data=float(e.sigma_data), rho=float(e.rho), P_mean=float(e.P_mean), P_std=float(e.P_std))


def run_epoch(epoch, engine, loader, optimizer, cfg, device, scaler, ema, smoke, limit=None):
    engine.train()
    n, running = 0, 0.0
    n_max = 4 if smoke else (min(len(loader), limit) if limit else len(loader))
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if step >= n_max:
            break
        clean = batch["volume"].to(device, non_blocking=True)
        if scaler is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                loss, m = engine.loss(clean)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss, m = engine.loss(clean); loss.backward(); optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None and cfg.ema.enabled:
            ema_update(ema, engine.denoiser, decay=cfg.ema.decay)
        running += m["loss/edm"]; n += 1
        if step % cfg.train.log_every_steps == 0:
            log.info("epoch %d step %d  edm=%.5f sigma=%.3f", epoch, step, m["loss/edm"], m["sigma/mean"])
    return running / max(1, n)


def save_ckpt(engine, ema, optimizer, epoch, ckpt_dir: Path, cfg):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save({"epoch": epoch, "denoiser": engine.denoiser.state_dict(),
                "ema_denoiser": ema.state_dict() if ema is not None else None,
                "optim": optimizer.state_dict(),
                "denoiser_cfg": OmegaConf.to_container(cfg.denoiser, resolve=True),
                "edm_cfg": OmegaConf.to_container(cfg.edm, resolve=True)}, out)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit-train-batches", type=int, default=None)
    args = p.parse_args(argv)
    cfg: DictConfig = OmegaConf.load(args.config)
    smoke = bool(args.smoke)
    if args.epochs is not None:
        cfg.train.n_epochs = int(args.epochs)
    set_determinism(seed=cfg.run.seed)
    torch.backends.cudnn.benchmark = True

    if smoke:
        device = torch.device("cpu")
        cfg.train.amp = False; cfg.train.n_epochs = 1; cfg.train.patch_size = [32, 32, 32]
        cfg.denoiser.channels = [16, 32, 32, 32]; cfg.denoiser.num_head_channels = [16, 16, 16, 16]
        cfg.denoiser.norm_num_groups = 16
        cfg.data.pair_mode = "online"; cfg.data.num_workers = 0; cfg.data.pin_memory = False; cfg.data.persistent_workers = False
        cfg.wandb.enabled = False
        log.info("[smoke] CPU, tiny denoiser, 32^3, online pairs")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    engine = build_engine(cfg).to(device)
    log.info("[model] UnconditionalEDM denoiser params=%.2fM device=%s",
             sum(pp.numel() for pp in engine.denoiser.parameters()) / 1e6, device)

    from code.training.train_diffusion import _smoke_online_motion_factory
    of = _smoke_online_motion_factory if smoke and cfg.data.pair_mode == "online" else None
    loader = build_paired_dataloader(cfg, smoke, online_motion_factory=of)
    optimizer = torch.optim.Adam(engine.denoiser.parameters(), lr=cfg.optim.lr,
                                 betas=tuple(cfg.optim.betas), weight_decay=cfg.optim.weight_decay)
    ema = None
    if cfg.ema.enabled:
        ema = copy.deepcopy(engine.denoiser).to(device)
        for pp in ema.parameters():
            pp.requires_grad_(False)
    start = 1
    if args.ckpt is not None:
        st = torch.load(args.ckpt, map_location=device)
        engine.denoiser.load_state_dict(st["denoiser"]); optimizer.load_state_dict(st["optim"])
        if ema is not None and st.get("ema_denoiser") is not None:
            ema.load_state_dict(st["ema_denoiser"])
        start = int(st.get("epoch", 0)) + 1
        log.info("[resume] %s @ epoch %d", args.ckpt, st.get("epoch", -1))
    scaler = torch.amp.GradScaler("cuda") if (cfg.train.amp and device.type == "cuda") else None

    runs = Path("experiments/runs") / cfg.run.slug; runs.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, runs / "config_resolved.yaml")
    t0 = time.time(); ckpt_dir = Path(cfg.train.ckpt_dir); last = start - 1
    for epoch in range(start, cfg.train.n_epochs + 1):
        last = epoch
        mean = run_epoch(epoch, engine, loader, optimizer, cfg, device, scaler, ema, smoke, args.limit_train_batches)
        log.info("[epoch %d] mean edm loss %.5f", epoch, mean)
        if (not smoke) and epoch % cfg.train.ckpt_every_epochs == 0:
            log.info("[ckpt] %s", save_ckpt(engine, ema, optimizer, epoch, ckpt_dir, cfg))
    if smoke:
        log.info("[smoke] ckpt %s", save_ckpt(engine, ema, optimizer, 0, ckpt_dir / "smoke", cfg))
    elif last % cfg.train.ckpt_every_epochs != 0:
        log.info("[ckpt final] %s", save_ckpt(engine, ema, optimizer, last, ckpt_dir, cfg))
    log.info("DONE in %.1fs (last_epoch=%d)", time.time() - t0, last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
