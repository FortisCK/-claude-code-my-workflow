"""dps_operator_smoke.py — Stage-0 feasibility smoke for DPS.

Verifies the cone-beam forward (project -> ramp -> backproject) is DIFFERENTIABLE
via tomosipo.to_autograd and that grad d||A(x)-y||^2 / dx flows back to the volume
without OOM, at the scales DPS needs. This is the single biggest engineering risk
of the DPS plan; confirm it cheaply before building the full sampler.

Usage:
    python -m scripts.python.dps_operator_smoke --n-views 250
    python -m scripts.python.dps_operator_smoke --n-views 1000   # memory probe
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np, torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd
from code.data import paths as path_registry
from code.data.motion_synth import ScanParams, ramp_filter_sinogram, parker_short_scan_weight


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", default="21")
    p.add_argument("--n-views", type=int, default=250)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    dev = torch.device(args.device)
    torch.cuda.reset_peak_memory_stats(dev) if dev.type == "cuda" else None

    d = np.load(path_registry.get("IMAGECAS_PROCESSED") / f"case_{args.case_id}__pair_0.npz", allow_pickle=True)
    clean = torch.from_numpy(d["volume"].astype(np.float32)).to(dev)       # (Z,Y,X) [-1,1]
    corrupted = torch.from_numpy(d["corrupted"].astype(np.float32)).to(dev)
    z, y, x = clean.shape
    sp = ScanParams(n_views=args.n_views)

    # ---- geometry (mirror cone_beam_simulate; 1mm iso volume) ----
    vg = ts.volume(shape=(z, y, x), size=(float(z), float(y), float(x)))
    fan = math.atan(sp.detector_n_cols * sp.detector_col_pitch_mm / 2 / sp.sdd_mm)
    scan_range = math.pi + 2 * fan
    angles = math.radians(sp.initial_scan_angle_deg) + torch.linspace(0, scan_range, sp.n_views, device=dev)
    det_v = sp.detector_n_rows * sp.detector_row_pitch_mm
    det_u = sp.detector_n_cols * sp.detector_col_pitch_mm
    pg = ts.cone(angles=angles.cpu().numpy(),
                 shape=(sp.detector_n_rows, sp.detector_n_cols), size=(det_v, det_u),
                 src_orig_dist=sp.sid_mm, src_det_dist=sp.sdd_mm)
    A = ts.operator(vg, pg)
    A_fwd = to_autograd(A, num_extra_dims=0)
    A_bwd = to_autograd(A.T, num_extra_dims=0)

    def forward(vol: torch.Tensor) -> torch.Tensor:
        sino = A_fwd(vol)                                   # (rows, n_views, cols)
        sino = parker_short_scan_weight(sino, angles, fan, sp)
        sino = ramp_filter_sinogram(sino)
        return A_bwd(sino)                                  # (Z,Y,X) FDK recon

    # ---- differentiable A(x); loss = ||A(x) - target||^2 with x != the volume that
    #      produced target, so the loss is non-trivial and grad must flow through A ----
    with torch.no_grad():
        target = forward(clean)
    xv = corrupted.clone().requires_grad_(True)   # different input -> nonzero loss
    rec = forward(xv)
    loss = torch.mean((rec - target) ** 2)
    loss.backward()

    g = xv.grad
    print(f"[smoke] n_views={args.n_views} vol={tuple(clean.shape)}")
    print(f"  forward recon: shape={tuple(rec.shape)} finite={torch.isfinite(rec).all().item()}")
    print(f"  loss={float(loss):.6e}")
    print(f"  grad: finite={torch.isfinite(g).all().item()} nonzero_frac={float((g.abs()>0).float().mean()):.3f} "
          f"absmax={float(g.abs().max()):.3e} absmean={float(g.abs().mean()):.3e}")
    if dev.type == "cuda":
        print(f"  peak GPU mem: {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB")
    ok = bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0
    print(f"  RESULT: {'PASS — differentiable forward works, grad flows' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
