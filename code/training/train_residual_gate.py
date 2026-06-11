"""Train a reliability gate on cached posterior residual-diffusion features."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.evaluation.metrics import (  # noqa: E402
    gradient_magnitude,
    make_boundary_band,
    masked_gradient_l1,
)
from code.models.residual_gate import ResidualGateNet3D  # noqa: E402
from code.training.train_diffusion import ema_update  # noqa: E402

log = logging.getLogger(__name__)


FLOAT_KEYS = ("clean", "corrupted", "initial", "residual_mean", "residual_std")


def build_gate(cfg: DictConfig) -> ResidualGateNet3D:
    return ResidualGateNet3D(
        in_channels=int(cfg.in_channels),
        features=tuple(int(v) for v in cfg.features),
        g_max=float(cfg.g_max),
        init_bias=float(cfg.init_bias),
        clamp_output=bool(cfg.clamp_output),
    )


def _as_3tuple(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"expected 3 values, got {values}")
    return tuple(int(v) for v in values)


def _discover_cache_files(
    cache_dir: Path,
    splits: list[str],
    case_ids: Optional[list[str]] = None,
) -> list[Path]:
    allowed = {str(cid) for cid in case_ids} if case_ids else None
    files: list[Path] = []
    for split_name in splits:
        split_dir = cache_dir / split_name
        if not split_dir.exists():
            log.warning("[cache] split dir does not exist: %s", split_dir)
            continue
        for path in sorted(split_dir.glob("case_*.npz")):
            cid = path.stem.removeprefix("case_")
            if allowed is None or cid in allowed:
                files.append(path)
    if not files:
        raise FileNotFoundError(
            f"no cached feature files under {cache_dir} for splits={splits} case_ids={case_ids}"
        )
    return files


def _load_npz_tensor(data: np.lib.npyio.NpzFile, key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data[key])).unsqueeze(0).to(dtype=dtype)


def _random_start(spatial: tuple[int, int, int], patch: tuple[int, int, int]) -> tuple[int, int, int]:
    starts: list[int] = []
    for size, roi in zip(spatial, patch):
        if size <= roi:
            starts.append(0)
        else:
            starts.append(int(torch.randint(0, size - roi + 1, (1,)).item()))
    return tuple(starts)  # type: ignore[return-value]


def _heart_centered_start(mask: torch.Tensor, patch: tuple[int, int, int]) -> tuple[int, int, int] | None:
    coords = torch.nonzero(mask[0] > 0, as_tuple=False)
    if coords.numel() == 0:
        return None
    spatial = tuple(int(v) for v in mask.shape[-3:])
    idx = int(torch.randint(0, coords.shape[0], (1,)).item())
    center = coords[idx].tolist()
    starts = []
    for c, size, roi in zip(center, spatial, patch):
        low = max(0, int(c) - roi + 1)
        high = min(int(c), size - roi)
        if high < low:
            starts.append(max(0, min(size - roi, int(c) - roi // 2)))
        else:
            starts.append(int(torch.randint(low, high + 1, (1,)).item()))
    return tuple(starts)  # type: ignore[return-value]


def _crop(tensor: torch.Tensor, start: tuple[int, int, int], patch: tuple[int, int, int]) -> torch.Tensor:
    z, y, x = start
    dz, dy, dx = patch
    return tensor[..., z : z + dz, y : y + dy, x : x + dx]


def make_gate_features(
    corrupted: torch.Tensor,
    initial: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
) -> torch.Tensor:
    """Build the six-channel GateNet feature tensor."""
    if corrupted.shape != initial.shape or initial.shape != residual_mean.shape:
        raise ValueError("corrupted, initial, and residual_mean shapes must match")
    grad_initial = gradient_magnitude(initial.unsqueeze(0))[0]
    return torch.cat(
        [
            corrupted,
            initial,
            corrupted - initial,
            residual_mean,
            residual_std,
            grad_initial,
        ],
        dim=0,
    )


class CachedResidualFeatureDataset(Dataset):
    """Random-crop dataset over full-volume residual-diffusion feature caches."""

    def __init__(
        self,
        files: list[Path],
        patch_size: tuple[int, int, int],
        samples_per_case_per_epoch: int = 4,
        heart_patch_prob: float = 0.75,
    ) -> None:
        self.files = list(files)
        self.patch_size = patch_size
        self.samples_per_case_per_epoch = max(1, int(samples_per_case_per_epoch))
        self.heart_patch_prob = float(heart_patch_prob)
        if not 0.0 <= self.heart_patch_prob <= 1.0:
            raise ValueError(f"heart_patch_prob must be in [0, 1], got {heart_patch_prob}")

    def __len__(self) -> int:
        return len(self.files) * self.samples_per_case_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index % len(self.files)]
        with np.load(path, allow_pickle=False) as data:
            loaded = {key: _load_npz_tensor(data, key) for key in FLOAT_KEYS}
            heart_mask = torch.from_numpy(np.asarray(data["heart_mask"])).unsqueeze(0).bool()
            case_id = str(data["case_id"]) if "case_id" in data.files else path.stem.removeprefix("case_")
        spatial = tuple(int(v) for v in loaded["clean"].shape[-3:])
        if torch.rand(()) < self.heart_patch_prob:
            start = _heart_centered_start(heart_mask, self.patch_size)
            if start is None:
                start = _random_start(spatial, self.patch_size)
        else:
            start = _random_start(spatial, self.patch_size)

        cropped = {key: _crop(value, start, self.patch_size).float() for key, value in loaded.items()}
        mask_crop = _crop(heart_mask, start, self.patch_size)
        features = make_gate_features(
            cropped["corrupted"],
            cropped["initial"],
            cropped["residual_mean"],
            cropped["residual_std"],
        )
        return {
            "features": features.float(),
            "clean": cropped["clean"].float(),
            "initial": cropped["initial"].float(),
            "residual_mean": cropped["residual_mean"].float(),
            "heart_mask": mask_crop,
            "case_id": case_id,
        }


class SyntheticGateDataset(Dataset):
    """Small random dataset for CPU smoke tests."""

    def __init__(self, length: int = 4, spatial: tuple[int, int, int] = (32, 32, 32)) -> None:
        self.length = int(length)
        self.spatial = spatial

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        del index
        clean = torch.randn(1, *self.spatial).mul(0.1).clamp(-1, 1)
        corrupted = (clean + 0.05 * torch.randn_like(clean)).clamp(-1, 1)
        initial = (clean + 0.03 * torch.randn_like(clean)).clamp(-1, 1)
        residual_target = clean - initial
        residual_mean = residual_target + 0.02 * torch.randn_like(clean)
        residual_std = torch.full_like(clean, 0.02)
        heart_mask = torch.zeros(1, *self.spatial, dtype=torch.bool)
        z0, z1 = self.spatial[0] // 4, 3 * self.spatial[0] // 4
        y0, y1 = self.spatial[1] // 4, 3 * self.spatial[1] // 4
        x0, x1 = self.spatial[2] // 4, 3 * self.spatial[2] // 4
        heart_mask[:, z0:z1, y0:y1, x0:x1] = True
        features = make_gate_features(corrupted, initial, residual_mean, residual_std)
        return {
            "features": features.float(),
            "clean": clean.float(),
            "initial": initial.float(),
            "residual_mean": residual_mean.float(),
            "heart_mask": heart_mask,
            "case_id": "synthetic",
        }


def _safe_mask_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    if mask_f.shape[1] == 1 and pred.shape[1] > 1:
        mask_f = mask_f.expand(-1, pred.shape[1], -1, -1, -1)
    denom = mask_f.sum()
    if float(denom.detach().cpu()) <= 0:
        return pred.new_tensor(0.0)
    return ((pred - target).abs() * mask_f).sum() / denom.clamp_min(1.0)


def _gate_tv(gate: torch.Tensor) -> torch.Tensor:
    dz = (gate[..., 1:, :, :] - gate[..., :-1, :, :]).abs().mean()
    dy = (gate[..., :, 1:, :] - gate[..., :, :-1, :]).abs().mean()
    dx = (gate[..., :, :, 1:] - gate[..., :, :, :-1]).abs().mean()
    return (dx + dy + dz) / 3.0


def _oracle_gate_target(
    clean: torch.Tensor,
    initial: torch.Tensor,
    residual_mean: torch.Tensor,
    g_max: float,
    eps: float,
) -> torch.Tensor:
    """Per-voxel bounded gate minimizing squared residual error.

    For the scalar correction `initial + g * residual_mean`, the least-squares
    optimum is `(clean - initial) * residual_mean / residual_mean^2`, clipped to
    the feasible gate range.
    """
    target_residual = clean - initial
    denom = residual_mean.square().clamp_min(float(eps))
    return ((target_residual * residual_mean) / denom).clamp(0.0, float(g_max))


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _oracle_gate_loss(
    gate: torch.Tensor,
    clean: torch.Tensor,
    initial: torch.Tensor,
    residual_mean: torch.Tensor,
    heart_mask: torch.Tensor,
    boundary: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    g_max = float(cfg.model.g_max)
    eps = float(cfg.loss.get("oracle_gate_eps", 1.0e-6))
    target = _oracle_gate_target(clean, initial, residual_mean, g_max=g_max, eps=eps)
    oracle_final = initial + target * residual_mean
    oracle_improvement = (initial - clean).abs() - (oracle_final - clean).abs()
    min_improvement = float(cfg.loss.get("oracle_gate_min_improvement", 0.0))
    if min_improvement > 0.0:
        improve_mask = oracle_improvement > min_improvement
        target = torch.where(improve_mask, target, torch.zeros_like(target))
    else:
        improve_mask = torch.ones_like(target, dtype=torch.bool)
    target_norm = (target / g_max).clamp(0.0, 1.0)
    pred_norm = (gate / g_max).clamp(1.0e-5, 1.0 - 1.0e-5)

    heart_f = heart_mask.to(device=gate.device, dtype=gate.dtype)
    boundary_f = boundary.to(device=gate.device, dtype=gate.dtype)
    positive = (target_norm > float(cfg.loss.get("oracle_gate_min_target", 0.01))).to(dtype=gate.dtype)

    weights = torch.full_like(gate, float(cfg.loss.get("oracle_gate_base_weight", 1.0)))
    weights = weights + float(cfg.loss.get("oracle_gate_positive_weight", 0.0)) * positive
    weights = weights + float(cfg.loss.get("oracle_gate_heart_weight", 0.0)) * heart_f
    weights = weights + float(cfg.loss.get("oracle_gate_boundary_weight", 0.0)) * boundary_f

    loss_type = str(cfg.loss.get("oracle_gate_loss", "bce")).lower()
    if loss_type == "l1":
        per_voxel = (pred_norm - target_norm).abs()
    elif loss_type == "mse":
        per_voxel = (pred_norm - target_norm).square()
    elif loss_type == "bce":
        pred_f = pred_norm.float()
        target_f = target_norm.float()
        per_voxel = -(target_f * pred_f.log() + (1.0 - target_f) * (1.0 - pred_f).log())
    else:
        raise ValueError(f"unknown oracle_gate_loss={loss_type!r}; expected bce, l1, or mse")

    loss = _weighted_mean(per_voxel, weights)
    heart_denom = heart_f.sum().clamp_min(1.0)
    boundary_denom = boundary_f.sum().clamp_min(1.0)
    metrics = {
        "loss/oracle_gate": float(loss.detach()),
        "gate_target/mean": float(target.detach().mean()),
        "gate_target/active_frac": float(positive.detach().mean()),
        "gate_target/heart_mean": float((target.detach() * heart_f).sum() / heart_denom),
        "gate_target/boundary_mean": float((target.detach() * boundary_f).sum() / boundary_denom),
        "gate_target/improve_frac": float(improve_mask.detach().float().mean()),
        "gate_target/improvement_mean": float(oracle_improvement.detach().clamp_min(0.0).mean()),
    }
    return loss, metrics


def gate_loss(
    final: torch.Tensor,
    clean: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    residual_mean: torch.Tensor,
    heart_mask: torch.Tensor,
    cfg: DictConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_l1 = F.l1_loss(final, clean)
    heart_l1 = _safe_mask_l1(final, clean, heart_mask)
    boundary = make_boundary_band(heart_mask, radius=int(cfg.loss.boundary_radius)).to(device=final.device)
    boundary_l1 = _safe_mask_l1(final, clean, boundary)
    if bool(boundary.any().item()):
        boundary_gradient = masked_gradient_l1(final, clean, boundary).mean()
    else:
        boundary_gradient = final.new_tensor(0.0)
    gate_l1 = gate.mean()
    gate_tv = _gate_tv(gate)
    oracle_gate = final.new_tensor(0.0)
    oracle_metrics: dict[str, float] = {}
    oracle_gate_weight = float(cfg.loss.get("oracle_gate_weight", 0.0))
    if oracle_gate_weight > 0.0:
        oracle_gate, oracle_metrics = _oracle_gate_loss(
            gate=gate,
            clean=clean,
            initial=initial,
            residual_mean=residual_mean,
            heart_mask=heart_mask,
            boundary=boundary,
            cfg=cfg,
        )
    total = (
        float(cfg.loss.final_l1_weight) * final_l1
        + float(cfg.loss.heart_l1_weight) * heart_l1
        + float(cfg.loss.boundary_l1_weight) * boundary_l1
        + float(cfg.loss.boundary_gradient_weight) * boundary_gradient
        + float(cfg.loss.gate_l1_weight) * gate_l1
        + float(cfg.loss.gate_tv_weight) * gate_tv
        + oracle_gate_weight * oracle_gate
    )
    metrics = {
        "loss/total": float(total.detach()),
        "loss/final_l1": float(final_l1.detach()),
        "loss/heart_l1": float(heart_l1.detach()),
        "loss/boundary_l1": float(boundary_l1.detach()),
        "loss/boundary_gradient": float(boundary_gradient.detach()),
        "loss/gate_l1": float(gate_l1.detach()),
        "loss/gate_tv": float(gate_tv.detach()),
        "gate/mean": float(gate.mean().detach()),
        "gate/max": float(gate.max().detach()),
    }
    metrics.update(oracle_metrics)
    return total, metrics


def build_dataloader(cfg: DictConfig, smoke: bool) -> DataLoader:
    if smoke:
        ds: Dataset = SyntheticGateDataset(length=4, spatial=_as_3tuple(cfg.train.patch_size))
        return DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    cache_dir = Path(cfg.data.cache_dir)
    train_files = _discover_cache_files(
        cache_dir=cache_dir,
        splits=list(cfg.data.train_splits),
        case_ids=list(cfg.data.case_ids_train) if cfg.data.case_ids_train else None,
    )
    ds = CachedResidualFeatureDataset(
        train_files,
        patch_size=_as_3tuple(cfg.train.patch_size),
        samples_per_case_per_epoch=int(cfg.train.get("samples_per_case_per_epoch", 4)),
        heart_patch_prob=float(cfg.train.get("heart_patch_prob", 0.75)),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers) and int(cfg.data.num_workers) > 0,
        drop_last=True,
    )
    log.info("[data] cache files=%d effective patches/epoch=%d", len(train_files), len(ds))
    return loader


def run_epoch(
    epoch: int,
    model: ResidualGateNet3D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    ema_model: Optional[ResidualGateNet3D],
    smoke: bool,
    limit_batches: Optional[int] = None,
    wandb_run=None,
) -> dict[str, float]:
    model.train()
    running: dict[str, float] = {
        "loss/total": 0.0,
        "loss/final_l1": 0.0,
        "loss/heart_l1": 0.0,
        "loss/boundary_l1": 0.0,
        "loss/boundary_gradient": 0.0,
        "loss/gate_l1": 0.0,
        "loss/gate_tv": 0.0,
        "gate/mean": 0.0,
        "gate/max": 0.0,
    }
    n_batches = 0
    n_max = 4 if smoke else len(loader)
    if limit_batches is not None:
        n_max = min(n_max, int(limit_batches))
    accum = max(1, int(cfg.train.grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        if step >= n_max:
            break
        features = batch["features"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        initial = batch["initial"].to(device, non_blocking=True)
        residual_mean = batch["residual_mean"].to(device, non_blocking=True)
        heart_mask = batch["heart_mask"].to(device, non_blocking=True).bool()

        if scaler is not None and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                final, gate = model(features, initial, residual_mean, return_gate=True)
                loss, metrics = gate_loss(final, clean, gate, initial, residual_mean, heart_mask, cfg)
                scaled_loss = loss / accum
            scaler.scale(scaled_loss).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            final, gate = model(features, initial, residual_mean, return_gate=True)
            loss, metrics = gate_loss(final, clean, gate, initial, residual_mean, heart_mask, cfg)
            (loss / accum).backward()
            should_step = (step + 1) % accum == 0 or (step + 1) >= n_max
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if should_step and ema_model is not None and cfg.ema.enabled:
            ema_update(ema_model, model, decay=float(cfg.ema.decay))

        for key, value in metrics.items():
            if key not in running:
                running[key] = 0.0
            running[key] += value
        n_batches += 1
        if step % int(cfg.train.log_every_steps) == 0:
            log.info(
                "epoch %d step %d loss=%.5f final=%.5f heart=%.5f boundary=%.5f gate=%.5f",
                epoch,
                step,
                metrics["loss/total"],
                metrics["loss/final_l1"],
                metrics["loss/heart_l1"],
                metrics["loss/boundary_l1"],
                metrics["gate/mean"],
            )
        if wandb_run is not None:
            wandb_run.log({**metrics, "epoch": epoch})

    return {f"{key}_mean": value / max(1, n_batches) for key, value in running.items()}


def save_checkpoint(
    model: ResidualGateNet3D,
    optimizer: torch.optim.Optimizer,
    ema_model: Optional[ResidualGateNet3D],
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
            "loss_cfg": OmegaConf.to_container(cfg.loss, resolve=True),
            "cache_dir": str(cfg.data.cache_dir),
        },
        out,
    )
    return out


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train residual-diffusion reliability GateNet.")
    p.add_argument("--config", type=Path, default=Path("code/training/configs/residual_gate_v1.yaml"))
    p.add_argument("--smoke", action="store_true", help="CPU smoke with synthetic cached-feature tensors.")
    p.add_argument("--ckpt", type=Path, default=None, help="resume gate checkpoint")
    p.add_argument("--epochs", type=int, default=None, help="override train.n_epochs")
    p.add_argument("--limit-train-batches", type=int, default=None, help="cap batches per epoch")
    p.add_argument("--ckpt-dir", type=Path, default=None, help="override train.ckpt_dir")
    p.add_argument("--run-slug", type=str, default=None, help="override run.slug")
    p.add_argument("--cache-dir", type=Path, default=None, help="override data.cache_dir")
    p.add_argument("--train-splits", nargs="+", default=None, help="override data.train_splits")
    p.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    p.add_argument("--no-pin-memory", action="store_true", help="disable DataLoader pin_memory")
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
    if args.cache_dir is not None:
        cfg.data.cache_dir = str(args.cache_dir)
    if args.train_splits is not None:
        cfg.data.train_splits = list(args.train_splits)
    if args.num_workers is not None:
        cfg.data.num_workers = int(args.num_workers)
        cfg.data.persistent_workers = int(args.num_workers) > 0
    if args.no_pin_memory:
        cfg.data.pin_memory = False
    if args.epochs is not None:
        cfg.train.n_epochs = int(args.epochs)
    if args.no_wandb:
        cfg.wandb.enabled = False
    set_determinism(seed=int(cfg.run.seed))

    if smoke:
        device = torch.device("cpu")
        cfg.train.amp = False
        cfg.wandb.enabled = False
        cfg.train.n_epochs = 1
        cfg.train.batch_size = 1
        cfg.train.patch_size = [32, 32, 32]
        cfg.model.features = [8, 8, 16, 32, 64, 8]
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.persistent_workers = False
        log.info("[smoke] forcing CPU, tiny gate, synthetic 32^3 feature tensors")
    else:
        device = torch.device(cfg.run.device if torch.cuda.is_available() else "cpu")

    model = build_gate(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("[model] ResidualGateNet3D params=%.2fM device=%s", n_params / 1e6, device)
    loader = build_dataloader(cfg, smoke=smoke)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.optim.lr),
        betas=tuple(float(v) for v in cfg.optim.betas),
        weight_decay=float(cfg.optim.weight_decay),
    )
    ema_model: Optional[ResidualGateNet3D] = None
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
    for epoch in range(start_epoch, int(cfg.train.n_epochs) + 1):
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
        if (not smoke) and (epoch % int(cfg.train.ckpt_every_epochs) == 0):
            out = save_checkpoint(model, optimizer, ema_model, epoch, ckpt_dir, cfg)
            log.info("[ckpt] %s", out)

    if smoke:
        out = save_checkpoint(model, optimizer, ema_model, 0, ckpt_dir / "smoke", cfg)
        log.info("[smoke] checkpoint at %s", out)
    elif last_epoch > 0 and last_epoch % int(cfg.train.ckpt_every_epochs) != 0:
        out = save_checkpoint(model, optimizer, ema_model, last_epoch, ckpt_dir, cfg)
        log.info("[ckpt final] %s", out)

    log.info("DONE in %.1fs (last_epoch=%d)", time.time() - t0, last_epoch)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
