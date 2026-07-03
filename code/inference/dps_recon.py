"""dps_recon.py — DPS sparse-view CCTA reconstruction (unconditional prior + data-consistency).

Reconstructs a clean CCTA volume from ultra-sparse cone-beam projections y using the
unconditional EDM image prior (UnconditionalEDM) guided by measurement-consistency:
    per reverse step: x_hat = prior.denoise(x, sigma)   [sliding-window over 192^3]
                      g     = d/dx_hat || A_sparse(denorm(x_hat)) - y_hu ||^2   (A differentiable)
                      x_hat <- x_hat - zeta * unit_rms(g)                        [DiffPIR guidance]
                      Euler step of x toward sigma_next using x_hat.
The prior works in normalized [-1,1] space; only the data term denormalizes to HU (A is in HU).

Plan: quality_reports/plans/2026-07-03_sparse-view-ccta-reconstruction-pivot.md (Stage 1)
"""
from __future__ import annotations

import torch
from monai.inferers import SlidingWindowInferer

_HU_LO, _HU_HI = -1024.0, 3071.0


def _denorm(x):
    return (x + 1.0) * 0.5 * (_HU_HI - _HU_LO) + _HU_LO


@torch.no_grad()
def dps_recon_sample(engine, fwd, y_hu, shape, device, num_steps: int = 40, zeta: float = 0.3,
                     roi: int = 128, guidance_clip: float = 0.1, x_init=None, seed: int | None = 0):
    """Return the DPS-reconstructed clean volume (1,1,D,H,W), normalized [-1,1].

    engine : UnconditionalEDM (patch-trained prior)
    fwd    : differentiable sparse cone-beam operator, HU-volume (D,H,W) -> sinogram
    y_hu   : observed sparse sinogram (HU-domain projections of the clean volume)
    """
    inferer = SlidingWindowInferer(roi_size=(roi, roi, roi), sw_batch_size=1, overlap=0.5,
                                   mode="gaussian", sw_device=device, device=device)
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    sigmas = engine.sample_sigmas(num_steps, device)
    x = (sigmas[0] * torch.randn(shape, generator=gen, device=device)) if x_init is None \
        else (x_init + sigmas[0] * torch.randn(shape, generator=gen, device=device))

    for i in range(num_steps):
        s = sigmas[i]; s_next = sigmas[i + 1]

        def pred(patch, _s=s):
            return engine.denoise(patch, _s.expand(patch.shape[0]))
        x_hat = inferer(x, pred)  # Tweedie clean estimate, normalized (1,1,D,H,W)

        # data-consistency gradient on x_hat (only A is differentiated)
        with torch.enable_grad():
            xh = x_hat.detach().requires_grad_(True)
            loss = torch.mean((fwd(_denorm(xh[0, 0])) - y_hu) ** 2)
            g = torch.autograd.grad(loss, xh)[0]
        g = g / (g.pow(2).mean().sqrt() + 1e-8)
        x_hat = x_hat - (zeta * g).clamp(-guidance_clip, guidance_clip)

        d = (x - x_hat) / s
        x = x + (s_next - s) * d
    return x_hat.detach()


def _smoke() -> int:
    """Verify the recon pipeline end-to-end on an (intermediate) prior checkpoint."""
    import sys
    import numpy as np
    from pathlib import Path
    from omegaconf import OmegaConf
    REPO = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "python"))
    sys.modules.pop("code", None)
    from code.data import paths as P
    from code.data.splits import load_split
    from code.models.conditional_image_denoiser import ConditionalImageDenoiser
    from code.models.unconditional_edm import UnconditionalEDM
    from sparse_view_sim import build_ops, denorm as sv_denorm

    dev = torch.device("cuda")
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "experiments/checkpoints/prior_edm_v1/epoch_010.pt"
    n_views = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    st = torch.load(ckpt, map_location=dev)
    dn = st["denoiser_cfg"]; ed = st["edm_cfg"]
    denoiser = ConditionalImageDenoiser(target_channels=dn["target_channels"], condition_channels=dn["condition_channels"],
                                        channels=tuple(dn["channels"]), num_res_blocks=tuple(dn["num_res_blocks"]),
                                        attention_levels=tuple(dn["attention_levels"]), num_head_channels=tuple(dn["num_head_channels"]),
                                        norm_num_groups=dn["norm_num_groups"]).to(dev)
    denoiser.load_state_dict(st["ema_denoiser"] if st.get("ema_denoiser") else st["denoiser"])
    denoiser.eval()
    engine = UnconditionalEDM(denoiser, **{k: ed[k] for k in ("sigma_min", "sigma_max", "sigma_data", "rho", "P_mean", "P_std")})

    cid = str(load_split(P.get("IMAGECAS_SPLITS") / "v1.json").test[0])
    d = np.load(P.get("IMAGECAS_PROCESSED") / f"case_{cid}__pair_0.npz", allow_pickle=True)
    clean = sv_denorm(torch.from_numpy(d["volume"].astype(np.float32))).to(dev)
    heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev) > 0
    shape = (1, 1, *clean.shape)
    fwd, _ = build_ops(tuple(clean.shape), n_views, dev)
    y = fwd(clean)
    torch.cuda.reset_peak_memory_stats(dev)
    rec = dps_recon_sample(engine, fwd, y, shape, dev, num_steps=32, zeta=0.3)
    rec_hu = _denorm(rec[0, 0])
    mae = float((rec_hu - clean)[heart].abs().mean())
    print(f"[dps_recon smoke] ckpt {Path(ckpt).name} views={n_views}: recon OK finite={torch.isfinite(rec).all().item()} "
          f"heart-MAE={mae:.1f} HU peak {torch.cuda.max_memory_allocated(dev)/1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
