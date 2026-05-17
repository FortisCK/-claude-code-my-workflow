"""smoke_flow.py — CPU smoke test for the Stochastic Interpolants module.

Mirrors Step 5 of `quality_reports/plans/atomic-munching-naur.md` (the EDM
smoke test) but exercises `code.models.stochastic_interpolant.StochasticInterpolant`
instead. Runs both schedules ("linear" = rectified flow; "trig" = VP-style)
and checks that:

    1. `loss(z_clean, z_cond)` returns a finite scalar that supports backward.
    2. `sample(z_cond, num_steps, n_samples)` returns a tensor of shape
       (B * n_samples, C, D, H, W) — the per-condition posterior chain layout.
    3. The deterministic ODE path matches the SDE path with `sde_noise_scale=0`
       (sanity: pragmatic SDE collapses to ODE when noise scale is zero).

CPU-only. Run from repo root:

    python -m scripts.python.smoke_flow
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from code.models.conditional_denoiser import ConditionalDenoiser
from code.models.stochastic_interpolant import StochasticInterpolant

log = logging.getLogger(__name__)

# Smoke-arch latent shape: (B, C, D, H, W) — small enough for CPU UNet forward.
LATENT_CHANNELS = 4
LATENT_SIZE = (8, 8, 8)
BATCH = 1
NUM_STEPS = 4  # Heun-2 → 8 NFE total
N_SAMPLES = 2

# Smaller UNet than the production default so CPU forward is bearable.
SMOKE_CHANNELS: tuple[int, ...] = (32, 64, 64, 64)
SMOKE_NUM_RES_BLOCKS: tuple[int, ...] = (1, 1, 1, 1)
SMOKE_ATTENTION_LEVELS: tuple[bool, ...] = (False, False, False, True)
SMOKE_NUM_HEAD_CHANNELS: tuple[int, ...] = (0, 0, 0, 32)


def _build_si(schedule: str, sde_noise_scale: float = 0.0) -> StochasticInterpolant:
    denoiser = ConditionalDenoiser(
        latent_channels=LATENT_CHANNELS,
        channels=SMOKE_CHANNELS,
        num_res_blocks=SMOKE_NUM_RES_BLOCKS,
        attention_levels=SMOKE_ATTENTION_LEVELS,
        num_head_channels=SMOKE_NUM_HEAD_CHANNELS,
        norm_num_groups=32,
    )
    return StochasticInterpolant(
        denoiser=denoiser,
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        schedule=schedule,
        sde_noise_scale=sde_noise_scale,
    )


def _rand_latent() -> tuple[torch.Tensor, torch.Tensor]:
    z_clean = torch.randn(BATCH, LATENT_CHANNELS, *LATENT_SIZE)
    z_cond = torch.randn(BATCH, LATENT_CHANNELS, *LATENT_SIZE)
    return z_clean, z_cond


def smoke_loss(schedule: str) -> None:
    log.info("[%s] loss + backward ...", schedule)
    si = _build_si(schedule)
    si.train()
    z_clean, z_cond = _rand_latent()
    loss = si.loss(z_clean, z_cond)
    assert loss.dim() == 0, f"expected scalar loss, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()
    grad_seen = any(p.grad is not None and torch.isfinite(p.grad).all() for p in si.parameters())
    assert grad_seen, "no finite gradient on any parameter"
    log.info("[%s] loss=%.6f  backward ok", schedule, float(loss.item()))


def smoke_sample(schedule: str) -> None:
    log.info("[%s] sample (deterministic ODE) ...", schedule)
    si = _build_si(schedule)
    si.eval()
    _, z_cond = _rand_latent()
    out = si.sample(z_cond, num_steps=NUM_STEPS, n_samples=N_SAMPLES, deterministic=True, seed=0)
    expected = (BATCH * N_SAMPLES, LATENT_CHANNELS, *LATENT_SIZE)
    assert tuple(out.shape) == expected, f"expected {expected}, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite values in sample output"
    log.info(
        "[%s] sample ok: shape %s, range [%.3f, %.3f]",
        schedule, tuple(out.shape), float(out.min()), float(out.max()),
    )


def smoke_sde_noise_zero_collapse() -> None:
    """sde_noise_scale=0 with deterministic=False ⇒ identical to deterministic ODE."""
    log.info("[linear] SDE-with-zero-noise == ODE sanity ...")
    si_ode = _build_si("linear", sde_noise_scale=0.0)
    si_ode.eval()
    _, z_cond = _rand_latent()
    out_ode = si_ode.sample(z_cond, num_steps=NUM_STEPS, n_samples=1, deterministic=True, seed=42)
    out_sde = si_ode.sample(z_cond, num_steps=NUM_STEPS, n_samples=1, deterministic=False, seed=42)
    assert torch.allclose(out_ode, out_sde, atol=1e-6), "ODE and zero-noise SDE diverged"
    log.info("[linear] ODE/SDE-zero collapse ok (max abs diff %.2e)",
             float((out_ode - out_sde).abs().max()))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for schedule in ("linear", "trig"):
        smoke_loss(schedule)
        smoke_sample(schedule)
    smoke_sde_noise_zero_collapse()
    log.info("STEP 5b SMOKE TEST (Stochastic Interpolant): PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
