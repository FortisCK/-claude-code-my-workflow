"""dps_forward.py — differentiable multi-phase cardiac-motion forward operator.

A(x) re-synthesizes the motion-corrupted volume from a clean estimate x using the
KNOWN per-case motion + cone-beam + Parker-FBP pipeline (motion_synth), made
differentiable via tomosipo.to_autograd (Stage-0 verified). Used for DPS:
guide a diffusion sample by grad ||A(x) - y||^2 where y = observed corrupted.

Forward (HU domain): warp x by each phase DVF (grid_sample) -> project that phase's
subset of views (to_autograd operator) -> assemble sinogram -> Parker weight +
ramp filter -> backproject (to_autograd) -> apply the STORED hu_calibration affine.

Plan: quality_reports/plans/2026-06-17_stage1-dps-mc-pilot.md (S1)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import tomosipo as ts
from tomosipo.torch_support import to_autograd

from code.data.motion_synth import (
    MotionParams,
    ScanParams,
    compute_motion_weight,
    estimate_heart_anatomy,
    parametric_dvf,
    parker_short_scan_weight,
    ramp_filter_sinogram,
    warp_volume,
)


class DifferentiableMotionForward:
    """Precompute per-case geometry + per-phase DVFs; __call__ applies A(x).

    Args:
        heart_mask: (Z,Y,X) {0,1} torch tensor on `device`.
        voxel_size_mm: (z,y,x).
        motion_params, scan_params: the STORED per-case params.
        n_phases: stored n_phases.
        hu_scale, hu_offset: stored hu_calibration (frozen — no baseline pass).
        view_stride: keep every `view_stride`-th view (memory control).
    """

    def __init__(self, heart_mask, voxel_size_mm, motion_params: MotionParams,
                 scan_params: ScanParams, n_phases: int, hu_scale: float,
                 hu_offset: float, view_stride: int = 4):
        dev = heart_mask.device
        self.device = dev
        self.voxel = voxel_size_mm
        self.scale = float(hu_scale)
        self.offset = float(hu_offset)
        z, y, x = heart_mask.shape

        anatomy = estimate_heart_anatomy(heart_mask, voxel_size_mm)
        motion_weight = compute_motion_weight(heart_mask, voxel_size_mm, motion_params.mask_smooth_sigma_mm)
        zc = torch.arange(z, device=dev).float() * voxel_size_mm[0]
        yc = torch.arange(y, device=dev).float() * voxel_size_mm[1]
        xc = torch.arange(x, device=dev).float() * voxel_size_mm[2]
        gz, gy, gx = torch.meshgrid(zc, yc, xc, indexing="ij")
        self.grid_mm = torch.stack([gz, gy, gx], dim=-1)

        sp = scan_params
        fan = math.atan(sp.detector_n_cols * sp.detector_col_pitch_mm / 2 / sp.sdd_mm)
        scan_range = math.pi + 2 * fan
        all_angles = math.radians(sp.initial_scan_angle_deg) + torch.linspace(0, scan_range, sp.n_views, device=dev)
        scan_time = sp.rotation_period_ms * (scan_range / (2 * math.pi))
        times = torch.linspace(0, scan_time, sp.n_views, device=dev)
        phase_of_view = (times / motion_params.cardiac_period_ms * n_phases).long() % n_phases

        keep = torch.arange(0, sp.n_views, view_stride, device=dev)
        self.angles = all_angles[keep]
        self.fan = fan
        self.sp = sp
        n_sub = keep.numel()

        vg = ts.volume(shape=(z, y, x), size=(z * voxel_size_mm[0], y * voxel_size_mm[1], x * voxel_size_mm[2]))
        det_v = sp.detector_n_rows * sp.detector_row_pitch_mm
        det_u = sp.detector_n_cols * sp.detector_col_pitch_mm

        # Per-phase DVF (independent of x) + per-phase subset-view forward operators.
        self.n_sub = n_sub
        phase_times = torch.linspace(0, motion_params.cardiac_period_ms, n_phases + 1, device=dev)[:-1]
        phase_of_keep = phase_of_view[keep]
        self.phase_entries = []  # (dvf, local_view_idx_tensor, to_autograd_fwd)
        for ph in range(n_phases):
            local = torch.where(phase_of_keep == ph)[0]
            if local.numel() == 0:
                continue
            dvf = parametric_dvf(self.grid_mm, motion_weight, anatomy, motion_params, float(phase_times[ph]))
            pg = ts.cone(angles=self.angles[local].cpu().numpy(),
                         shape=(sp.detector_n_rows, sp.detector_n_cols), size=(det_v, det_u),
                         src_orig_dist=sp.sid_mm, src_det_dist=sp.sdd_mm)
            self.phase_entries.append((dvf, local, to_autograd(ts.operator(vg, pg), num_extra_dims=0)))

        pg_full = ts.cone(angles=self.angles.cpu().numpy(),
                          shape=(sp.detector_n_rows, sp.detector_n_cols), size=(det_v, det_u),
                          src_orig_dist=sp.sid_mm, src_det_dist=sp.sdd_mm)
        self.backproject = to_autograd(ts.operator(vg, pg_full).T, num_extra_dims=0)

    def __call__(self, x_hu: torch.Tensor) -> torch.Tensor:
        """A(x): clean-estimate HU volume (Z,Y,X) -> re-synthesized corrupted HU."""
        sino = torch.zeros(self.sp.detector_n_rows, self.n_sub, self.sp.detector_n_cols, device=self.device)
        for dvf, local, fwd in self.phase_entries:
            warped = warp_volume(x_hu, self.grid_mm, dvf, self.voxel)
            chunk = fwd(warped)  # (rows, n_local, cols)
            sino = sino.index_copy(1, local, chunk)
        sino = parker_short_scan_weight(sino, self.angles, self.fan, self.sp)
        sino = ramp_filter_sinogram(sino)
        recon = self.backproject(sino)
        return recon * self.scale + self.offset


def _smoke() -> int:
    import numpy as np
    from code.data import paths as path_registry
    dev = torch.device("cuda")
    cid = "21"
    d = np.load(path_registry.get("IMAGECAS_PROCESSED") / f"case_{cid}__pair_0.npz", allow_pickle=True)
    meta = d["metadata"].item()
    HU_LO, HU_HI = -1024.0, 3071.0
    def denorm(a): return (a + 1.0) * 0.5 * (HU_HI - HU_LO) + HU_LO
    clean_hu = torch.from_numpy(denorm(d["volume"].astype(np.float32))).to(dev)
    corrupted_hu = torch.from_numpy(denorm(d["corrupted"].astype(np.float32))).to(dev)
    heart = torch.from_numpy(d["heart_mask"].astype(np.float32)).to(dev)
    mp = MotionParams(**{k: meta["motion_params"][k] for k in meta["motion_params"]})
    sp = ScanParams(**{k: meta["scan_params"][k] for k in meta["scan_params"]})
    cal = meta["hu_calibration"]
    torch.cuda.reset_peak_memory_stats(dev)
    A = DifferentiableMotionForward(heart, (1.0, 1.0, 1.0), mp, sp, int(meta["n_phases"]),
                                    cal["scale"], cal["offset"], view_stride=4)
    # forward correctness: A(clean) should resemble the stored corrupted
    with torch.no_grad():
        a_clean = A(clean_hu)
    heart_b = heart > 0
    mae_clean = float((a_clean[heart_b] - corrupted_hu[heart_b]).abs().mean())
    mae_naive = float((clean_hu[heart_b] - corrupted_hu[heart_b]).abs().mean())
    print(f"[A_diff smoke] heart-MAE  A(clean) vs corrupted = {mae_clean:.1f} HU  (clean vs corrupted = {mae_naive:.1f} HU)")
    # differentiability: grad of ||A(x)-corrupted||^2 wrt x
    xv = corrupted_hu.clone().requires_grad_(True)
    loss = torch.mean((A(xv) - corrupted_hu) ** 2)
    loss.backward()
    g = xv.grad
    print(f"  loss={float(loss):.4e}  grad finite={torch.isfinite(g).all().item()} "
          f"nonzero={float((g.abs()>0).float().mean()):.3f} absmean={float(g.abs().mean()):.3e}")
    print(f"  peak GPU mem: {torch.cuda.max_memory_allocated(dev)/1e9:.2f} GB  (n_sub_views={A.n_sub})")
    ok = torch.isfinite(g).all().item() and float(g.abs().max()) > 0 and mae_clean < mae_naive
    print(f"  RESULT: {'PASS' if ok else 'CHECK'} (A(clean) closer to corrupted than clean is, and grad flows)")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_smoke())
