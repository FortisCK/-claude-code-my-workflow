"""dps_sample.py — measurement-consistency posterior sampling (DiffPIR-style DPS).

Guides the converged residual-EDM prior toward a clean image x whose motion
re-synthesis matches the observed corrupted volume, using the differentiable
forward operator A (dps_forward.DifferentiableMotionForward).

Per reverse step (DiffPIR formulation — stable, memory-light: only A is
backpropagated, NOT the denoiser):
    r0 = denoise(residual, cond, sigma)              # Tweedie clean-residual (no grad)
    x0 = initial + r0                                  # clean estimate (normalized)
    g  = d/dx0 || A(denorm(x0)) - y_hu ||^2            # data-consistency grad (A backward)
    x0 <- x0 - zeta_t * g_clamped                      # data-consistency correction
    r0 <- x0 - initial ; Euler step residual toward sigma_next using r0

Plan: quality_reports/plans/2026-06-17_stage1-dps-mc-pilot.md (S2)
"""
from __future__ import annotations

import torch

_HU_LO, _HU_HI = -1024.0, 3071.0


def _denorm(x_norm: torch.Tensor) -> torch.Tensor:
    return (x_norm + 1.0) * 0.5 * (_HU_HI - _HU_LO) + _HU_LO


@torch.no_grad()
def dps_residual_sample(
    engine,
    condition: torch.Tensor,        # (1, 3, D,H,W) [corrupted, initial, corrupted-initial], normalized
    initial: torch.Tensor,          # (1, 1, D,H,W) U-Net output, normalized
    corrupted_hu: torch.Tensor,     # (D,H,W) observed corrupted, HU
    forward_op,                     # DifferentiableMotionForward: A(x_hu)->hu
    num_steps: int = 32,
    zeta: float = 0.3,
    guidance_clip: float = 0.05,
    seed: int | None = 0,
    log_every: int = 0,
) -> torch.Tensor:
    """Return the DC-guided corrected clean estimate `x = initial + residual` (normalized)."""
    device = condition.device
    spatial = tuple(int(v) for v in condition.shape[-3:])
    noise_shape = (1, engine.target_channels, *spatial)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    init_noise = torch.randn(noise_shape, generator=gen, device=device)

    sigmas = engine.sample_schedule(num_steps, device)
    residual = sigmas[0] * init_noise

    for idx in range(num_steps):
        sigma = float(sigmas[idx].item())
        sigma_next = float(sigmas[idx + 1].item())
        sig_t = torch.full((1,), sigma, device=device)
        r0 = engine._denoise(residual, condition, sig_t)          # Tweedie (no grad)
        x0 = initial + r0

        # ---- data-consistency gradient on x0 (only A is differentiated) ----
        with torch.enable_grad():
            x0g = x0.detach().requires_grad_(True)
            pred_hu = forward_op(_denorm(x0g[0, 0]))
            loss = torch.mean((pred_hu - corrupted_hu) ** 2)
            g = torch.autograd.grad(loss, x0g)[0]
        # unit-RMS normalize so `zeta` is a per-step displacement in [-1,1] units
        # (stable; no vanishing-at-low-sigma anneal). Optional per-step clip.
        g = g / (g.pow(2).mean().sqrt() + 1e-8)
        step = (zeta * g).clamp(-guidance_clip, guidance_clip)
        x0_dc = x0 - step
        r0_dc = x0_dc - initial

        # Euler step of the residual toward sigma_next using the DC-corrected denoised estimate
        d_cur = (residual - r0_dc) / sigma
        residual = residual + (sigma_next - sigma) * d_cur
        if log_every and (idx % log_every == 0 or idx == num_steps - 1):
            print(f"  step {idx} sigma={sigma:.3f} dc_loss={float(loss):.3e} |g|={float(g.abs().mean()):.3e}", flush=True)

    return (initial + residual).detach()


def _smoke() -> int:
    import sys
    import numpy as np
    from omegaconf import OmegaConf
    from code.data import paths as P
    from code.data.motion_synth import MotionParams, ScanParams
    from code.inference.dps_forward import DifferentiableMotionForward
    from code.models.residual_refiner import ResidualRefinerUNet3D
    from code.training.train_residual_refiner import load_initializer

    # local imports of the eval helpers for loading the residual-EDM engine
    sys.path.insert(0, "scripts/python")
    from evaluate_residual_diffusion_full_volume import load_residual_diffusion

    dev = torch.device("cuda")
    cid = "21"
    d = np.load(P.get("IMAGECAS_PROCESSED") / f"case_{cid}__pair_0.npz", allow_pickle=True)
    meta = d["metadata"].item()
    clean = torch.from_numpy(d["volume"].astype(np.float32))[None, None].to(dev)
    corrupted = torch.from_numpy(d["corrupted"].astype(np.float32))[None, None].to(dev)
    heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev)

    cfg = OmegaConf.load("code/training/configs/diffusion_v2_residual.yaml")
    initializer = load_initializer(cfg.initializer, dev)
    engine, _ = load_residual_diffusion("code/training/configs/diffusion_v2_residual.yaml",
                                        "experiments/checkpoints/diffusion_v2_residual/longrun/epoch_080.pt",
                                        "ema", dev)
    with torch.no_grad():
        initial = initializer(corrupted)
    condition = ResidualRefinerUNet3D.make_condition(corrupted.float(), initial.float())

    mp = MotionParams(**meta["motion_params"]); sp = ScanParams(**meta["scan_params"]); cal = meta["hu_calibration"]
    corrupted_hu = _denorm(corrupted[0, 0])
    A = DifferentiableMotionForward(heart, (1.0, 1.0, 1.0), mp, sp, int(meta["n_phases"]),
                                    cal["scale"], cal["offset"], view_stride=1)
    hb = heart > 0
    HU = 2047.5
    mae_unet = float((initial - clean).abs()[0, 0][hb].mean() * HU)
    zetas = [float(z) for z in (sys.argv[1:] or [0.02, 0.05, 0.1, 0.2])]
    print(f"[dps S3 sweep] U-Net heart-MAE={mae_unet:.1f} HU")
    for z in zetas:
        torch.cuda.reset_peak_memory_stats(dev)
        x = dps_residual_sample(engine, condition, initial, corrupted_hu, A,
                                num_steps=24, zeta=z, guidance_clip=0.3, seed=0, log_every=23)
        mae_dps = float((x - clean).abs()[0, 0][hb].mean() * HU)
        print(f"  zeta={z}: DPS heart-MAE={mae_dps:.1f} HU (U-Net {mae_unet:.1f}) finite={torch.isfinite(x).all().item()} "
              f"peak {torch.cuda.max_memory_allocated(dev)/1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
