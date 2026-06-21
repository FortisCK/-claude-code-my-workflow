"""motion_synth.py — parametric DVF + cone-beam motion artifact synthesis.

Generates one motion-corrupted volume `V_corrupted` from a clean ImageCAS
single-phase CCTA volume `V_clean` via:

    1. Parametric 4-component cardiac motion model (LV/RV contraction +
       apex twist + long-axis shortening + small bulk translation).
    2. Per-phase dense displacement field (DVF), warping V_clean to a
       sequence of N_phases ≈ 48 dynamic snapshots.
    3. Cone-beam scan simulation: each projection view samples the volume
       at the cardiac phase corresponding to the view's acquisition time.
    4. Parker-weighted short-scan FBP reconstruction → V_corrupted.

Spec: quality_reports/specs/2026-04-30_motion-synthesis-pipeline.md (APPROVED v1.0)
Lit-review: quality_reports/lit_review_parametric-cardiac-CT-motion-simulation.md

Antecedents (closest published forward-simulation work):
    - Lossau et al. 2019 (MedIA 52:68-79) — CoMoFACT, parametric forward model
      on coronary-segment patches; we generalize to whole-heart volumes.
    - Hahn et al. 2017 (Med Phys 44(11):5795-5813) — PAMoCo, vessel-centerline
      polynomial motion model in PAR framework.
    - Maier et al. 2021 (Med Phys 48(7):3559-3571) — Deep PAMoCo, deep-learning
      extension of Hahn 2017.

Parameter-range anchors (LV mechanics literature):
    - LV twist normal mean ~7-8° ± 3° — Sengupta 2008 (JACC CV Img); Stöhr 2016
      (AJP-Heart 311:H633-H644); Notomi 2005 (Circulation).
    - LV AV-plane displacement ~12-15 mm in healthy subjects.
    - LV global radial strain ~47% (35-59%) — Truong 2024 meta.
    - Cardiac period ~600-1000 ms (60-100 bpm normal HR).

Backend: tomosipo + ASTRA (GPU cone-beam projector + back-projector).

Per `.claude/rules/python-code-conventions.md`:
    - pathlib.Path / no os.path.join
    - snake_case + type hints on public APIs
    - constants as UPPER_SNAKE_CASE module-level
    - no float ==, clamp probabilities, EPS for FP32

Public entry point:

    synthesize_motion_artifact(V_clean, heart_mask, voxel_size_mm,
                               motion_params, scan_params, n_phases, seed)
        → (V_corrupted, debug_info)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS_FP32: float = 1e-7
DEFAULT_N_PHASES: int = 48


# ============================================================
# Parameter dataclasses
# ============================================================


@dataclass
class MotionParams:
    """Cardiac motion model parameters (mm / deg / ms).

    Defaults are anchored to PEAK-systolic values from the LV mechanics
    literature (Stöhr 2016 Fig 1C — peak twist ~15° in resting healthy
    individuals; AV-plane systolic excursion 12-15 mm; global radial strain
    35-59% mean 47%).

    For training-data generation, we recommend sampling each parameter from a
    wider distribution to expose the diffusion model to a range of motion
    artifact strengths (rest → mild exercise regime):

        contraction_amp_mm:  U(7, 14)         # ~30-55% radial strain
        twist_amp_deg:       U(8, 25)         # rest peak 15° → mild exercise 22°
        long_axis_amp_mm:    U(8, 14)         # AV-plane 12-15 mm range
        cardiac_period_ms:   U(600, 1000)     # 60-100 bpm normal HR

    The simplest training augmentation is to sample only `motion_strength`
    (single global multiplier in [0, 1.5], following Lossau 2019 CoMoFACT's
    `s ∈ [0, 10]` convention) while keeping the four amplitude parameters at
    their physiological defaults. This reduces the augmentation hyperparameter
    space from 4D to 1D.

    See motion-synthesis-pipeline.md §A.5 for the upstream randomization hook.
    """

    contraction_amp_mm: float = 10.0   # ~40% radial strain at mid-LV (Truong 2024 mean)
    twist_amp_deg: float = 12.0        # peak twist resting healthy (Stöhr 2016 Fig 1C: ~15°; conservative)
    long_axis_amp_mm: float = 10.0     # AV-plane displacement (Stöhr 2016: 12-15 mm; conservative)
    cardiac_period_ms: float = 800.0   # 75 bpm — within Maier 2021 / Lossau 2019 ranges
    contraction_sigma_factor: float = 1.0  # Gaussian-decay σ = factor × heart_radius

    # New (2026-05-01): Lossau-2019 / Maier-2021 alignment.
    motion_strength: float = 1.0        # global scalar multiplier; Lossau s∈[0,10] convention; 0=no motion
    mask_smooth_sigma_mm: float = 5.0   # Gaussian σ for heart_mask boundary smoothing
                                        # (Lossau used 15-mm dilation + 12.4-mm uniform filter;
                                        # Maier used w(r) → 0 at patch boundary).
                                        # Set to 0 to disable smoothing (legacy behavior).

    # New (2026-06-19): cardiac phase offset — shifts WHICH gantry views see peak-systolic
    # deformation (the only operator DoF that re-SHAPES, not just scales, the artifact).
    # Applied once at the shared time→profile point in parametric_dvf (so it affects BOTH
    # synthesize_motion_artifact and the DPS forward operator identically). frac of one cycle.
    phase_offset_frac: float = 0.0

    @property
    def twist_amp_rad(self) -> float:
        return math.radians(self.twist_amp_deg)


@dataclass
class ScanParams:
    """Cone-beam scan geometry (clinical cardiac CT defaults).

    See spec §4 for source / rationale.
    """

    sid_mm: float = 540.0  # source-isocenter
    sdd_mm: float = 950.0  # source-detector
    n_views: int = 1000
    rotation_period_ms: float = 250.0
    detector_n_rows: int = 320
    detector_n_cols: int = 800
    detector_row_pitch_mm: float = 0.625
    detector_col_pitch_mm: float = 1.0
    initial_scan_angle_deg: float = 0.0
    reconstruction_phase_pct: float = 75.0  # mid-diastole standard


# ============================================================
# Time profile s(t)  —  spec §A.4
# ============================================================


def time_profile(t_ms: float, period_ms: float) -> float:
    """Asymmetric sinusoidal cardiac contraction profile.

    s(0)        = 0   (end-diastole)
    s(0.4·T)    = 1   (end-systole, peak contraction)
    s(T)        = 0   (next end-diastole)

    Systole (0–0.4·T) is fast (sin), diastole (0.4·T–T) is slow (cos).
    """
    t_mod = t_ms % period_ms
    systole_end = 0.4 * period_ms
    if t_mod < systole_end:
        return math.sin(math.pi * t_mod / systole_end)
    return math.cos(math.pi * (t_mod - systole_end) / (2 * 0.6 * period_ms))


# ============================================================
# Heart anatomy from mask  —  spec §3
# ============================================================


def estimate_heart_anatomy(
    heart_mask: torch.Tensor,
    voxel_size_mm: tuple[float, float, float],
) -> dict[str, object]:
    """Extract centroid (mm), long-axis unit vector, characteristic radius and axial extent.

    Args:
        heart_mask: (Z, Y, X) torch tensor of {0, 1}.
        voxel_size_mm: (z, y, x) mm voxel sizes.

    Returns dict with keys:
        centroid_mm: (3,) tensor in (z, y, x) mm
        long_axis:   (3,) unit tensor (PCA principal axis)
        radius_mm:   scalar — mean distance from centroid
        axial_extent_mm: scalar — extent along long axis
    """
    coords_z, coords_y, coords_x = torch.nonzero(heart_mask, as_tuple=True)
    if coords_z.numel() == 0:
        raise ValueError("Empty heart_mask — cannot estimate anatomy.")

    points_mm = torch.stack(
        [
            coords_z.float() * voxel_size_mm[0],
            coords_y.float() * voxel_size_mm[1],
            coords_x.float() * voxel_size_mm[2],
        ],
        dim=-1,
    )  # (N, 3) in (z, y, x) mm

    centroid_mm = points_mm.mean(dim=0)  # (3,)

    centered = points_mm - centroid_mm
    cov = (centered.T @ centered) / centered.shape[0]
    _eigvals, eigvecs = torch.linalg.eigh(cov)
    long_axis = eigvecs[:, -1]  # largest eigenvalue → principal axis
    long_axis = long_axis / torch.linalg.norm(long_axis)

    radius_mm = torch.linalg.norm(centered, dim=-1).mean().item()
    axial_proj = centered @ long_axis
    axial_extent_mm = (axial_proj.max() - axial_proj.min()).item()

    return {
        "centroid_mm": centroid_mm,
        "long_axis": long_axis,
        "radius_mm": radius_mm,
        "axial_extent_mm": axial_extent_mm,
    }


# ============================================================
# Heart-mask boundary smoothing (Lossau 2019 / Maier 2021 alignment)
# ============================================================


def compute_motion_weight(
    heart_mask: torch.Tensor,
    voxel_size_mm: tuple[float, float, float],
    sigma_mm: float,
) -> torch.Tensor:
    """Soft-edged motion-weighting mask analog to Lossau (2019) m_c̃ /
    Maier (2021) w(r): 1 inside the heart, smoothly tapering to 0 outside.

    Implements a simple Gaussian blur of the binary mask, normalized so the
    interior plateau equals 1. Set sigma_mm = 0 to disable smoothing and fall
    back to the original binary mask (legacy behavior).
    """
    if sigma_mm <= 0:
        return heart_mask.float()

    # Lazy import: scipy is not a hard dependency for the hot path.
    import numpy as np
    import scipy.ndimage as ndi

    sigma_vox = (
        sigma_mm / voxel_size_mm[0],
        sigma_mm / voxel_size_mm[1],
        sigma_mm / voxel_size_mm[2],
    )
    arr = heart_mask.detach().cpu().numpy().astype(np.float32)
    smoothed = ndi.gaussian_filter(arr, sigma=sigma_vox, mode="constant", cval=0.0)
    plateau = float(smoothed.max())
    if plateau > EPS_FP32:
        smoothed = smoothed / plateau
    smoothed = np.clip(smoothed, 0.0, 1.0)
    return torch.from_numpy(smoothed).to(heart_mask.device)


# ============================================================
# Parametric DVF generator  —  spec §A
# ============================================================


def parametric_dvf(
    grid_mm: torch.Tensor,
    motion_weight: torch.Tensor,
    anatomy: dict[str, object],
    motion_params: MotionParams,
    t_ms: float,
) -> torch.Tensor:
    """Generate dense displacement field at time t_ms within cardiac cycle.

    Args:
        grid_mm: (Z, Y, X, 3) physical mm coordinates in (z, y, x) order.
        motion_weight: (Z, Y, X) [0, 1] soft mask from compute_motion_weight().
            1 inside heart, smoothly decays to 0 outside (analog to Lossau
            2019 m_c̃ / Maier 2021 w(r) — prevents boundary discontinuities).
        anatomy: dict from estimate_heart_anatomy().
        motion_params: MotionParams.
        t_ms: time within cardiac cycle [0, period_ms).

    Returns:
        (Z, Y, X, 3) DVF in mm, in (z, y, x) order.
    """
    device = grid_mm.device
    centroid = anatomy["centroid_mm"].to(device)  # (3,)
    long_axis = anatomy["long_axis"].to(device)   # (3,)
    radius = float(anatomy["radius_mm"])
    axial_extent = float(anatomy["axial_extent_mm"])

    # phase offset shifts which cardiac time this phase bucket represents (shared by
    # synth + DPS forward op; time_profile mods by period internally).
    t_eff = t_ms + motion_params.phase_offset_frac * motion_params.cardiac_period_ms
    s_t = time_profile(t_eff, motion_params.cardiac_period_ms)

    r = grid_mm - centroid                                  # (Z, Y, X, 3)
    r_norm = torch.linalg.norm(r, dim=-1, keepdim=True)     # (Z, Y, X, 1)
    r_unit = r / (r_norm + EPS_FP32)

    # ----- Contraction: inward toward LV centroid -----
    sigma = max(radius * motion_params.contraction_sigma_factor, 1.0)
    decay = torch.exp(-(r_norm**2) / (2 * sigma**2))         # (Z, Y, X, 1)
    a_c = motion_params.contraction_amp_mm
    d_contract = -a_c * s_t * decay * r_unit                 # (Z, Y, X, 3)

    # ----- Long-axis projection -----
    d_z = (r * long_axis).sum(dim=-1, keepdim=True)          # (Z, Y, X, 1)
    h_axial = max(axial_extent, 1.0)

    # ----- Twist: small-angle rotation around long axis, scaled by axial position -----
    r_perp = r - d_z * long_axis                             # (Z, Y, X, 3)
    long_axis_b = long_axis.expand_as(r_perp)
    twist_scale = motion_params.twist_amp_rad * s_t * (d_z / h_axial)
    cross = torch.linalg.cross(long_axis_b, r_perp, dim=-1)
    d_twist = twist_scale * cross

    # ----- Long-axis shortening: apex moves toward base -----
    long_scale = motion_params.long_axis_amp_mm * s_t * (d_z / h_axial)
    d_long = -long_scale * long_axis

    d = d_contract + d_twist + d_long

    # Apply global motion-strength scalar (Lossau 2019 s convention) and
    # the soft motion-weight mask (smooth boundary decay → no boundary
    # discontinuity → no truncation streaks at lung/heart interface).
    return d * motion_params.motion_strength * motion_weight.unsqueeze(-1)


def make_random_smooth_dvf_fn(
    seed: int,
    peak_mm: float = 12.0,
    n_ctrl: int = 5,
):
    """Factory: a structurally-DIFFERENT motion model for OOD generalization tests.

    Returns a `dvf_fn` with the same signature as `parametric_dvf`, but the
    spatial deformation has NO anatomical structure (no contraction / twist /
    long-axis terms) — it is a smooth random field from a coarse seeded control
    grid, trilinearly upsampled and rescaled to a fixed peak displacement. The
    field is fixed in space and modulated over the cardiac cycle by the SAME
    `time_profile` s(t), so it is a coherent cardiac-like deformation that the
    parametric-trained model has never seen. Magnitude (`peak_mm`) is the knob to
    severity-match against the parametric control, so an OOD drop isolates
    "different motion MODEL" from "harder motion".

    Args:
        seed: RNG seed for the control grid (reproducible).
        peak_mm: peak displacement magnitude (mm) of the static field.
        n_ctrl: control-grid resolution per axis (low → smoother field).
    """
    cache: dict[tuple[int, int, int], torch.Tensor] = {}

    def dvf_fn(grid_mm, motion_weight, anatomy, motion_params, t_ms):
        device = grid_mm.device
        z_size, y_size, x_size, _ = grid_mm.shape
        key = (z_size, y_size, x_size)
        if key not in cache:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            ctrl = torch.randn(1, 3, n_ctrl, n_ctrl, n_ctrl, generator=gen)
            d0 = F.interpolate(
                ctrl, size=(z_size, y_size, x_size), mode="trilinear", align_corners=True
            )[0].permute(1, 2, 3, 0)  # (Z, Y, X, 3)
            mag = torch.linalg.norm(d0, dim=-1).max()
            d0 = d0 / (mag + EPS_FP32) * peak_mm
            cache[key] = d0.to(device)
        t_eff = t_ms + motion_params.phase_offset_frac * motion_params.cardiac_period_ms
        s_t = time_profile(t_eff, motion_params.cardiac_period_ms)
        return cache[key] * s_t * motion_params.motion_strength * motion_weight.unsqueeze(-1)

    return dvf_fn


# ============================================================
# Volume warping via grid_sample
# ============================================================


def warp_volume(
    volume: torch.Tensor,
    grid_mm: torch.Tensor,
    dvf: torch.Tensor,
    voxel_size_mm: tuple[float, float, float],
) -> torch.Tensor:
    """Warp `volume` (Z, Y, X) by DVF (Z, Y, X, 3) using PyTorch grid_sample.

    Sample positions are grid_mm + dvf, converted to normalized [-1, 1].
    Bilinear interpolation, border padding (matches CT scan behavior at boundary).
    """
    z_size, y_size, x_size = volume.shape
    sample_mm = grid_mm + dvf

    # mm -> voxel index
    sample_idx_z = sample_mm[..., 0] / voxel_size_mm[0]
    sample_idx_y = sample_mm[..., 1] / voxel_size_mm[1]
    sample_idx_x = sample_mm[..., 2] / voxel_size_mm[2]

    # voxel index -> normalized [-1, 1] (align_corners=True)
    norm_z = 2.0 * sample_idx_z / (z_size - 1) - 1.0
    norm_y = 2.0 * sample_idx_y / (y_size - 1) - 1.0
    norm_x = 2.0 * sample_idx_x / (x_size - 1) - 1.0

    # grid_sample expects last dim = (x, y, z)
    grid = torch.stack([norm_x, norm_y, norm_z], dim=-1).unsqueeze(0)  # (1, Z, Y, X, 3)
    vol_in = volume.unsqueeze(0).unsqueeze(0)                          # (1, 1, Z, Y, X)
    vol_out = F.grid_sample(
        vol_in, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return vol_out.squeeze(0).squeeze(0)


# ============================================================
# Cone-beam scan + Parker + FBP
# ============================================================


def parker_short_scan_weight(
    sino: torch.Tensor,
    angles_rad: torch.Tensor,
    fan_angle: float,
    scan_params: ScanParams,
) -> torch.Tensor:
    """Parker short-scan weighting (matches TT U-Net's getshortscanweighted.m).

    sino: (rows, n_views, cols)
    angles_rad: (n_views,) view angles relative to start
    """
    rows, n_views, cols = sino.shape
    device = sino.device

    col_idx = torch.arange(cols, device=device).float() - cols / 2.0 - 0.5
    col_pos = col_idx * scan_params.detector_col_pitch_mm
    alpha = torch.atan(col_pos / scan_params.sdd_mm)  # (cols,)
    delta = float(alpha.abs().max().item())

    beta = angles_rad - angles_rad[0]                        # (n_views,)
    beta_2d = beta.unsqueeze(-1).expand(n_views, cols)       # (V, C)
    alpha_2d = alpha.unsqueeze(0).expand(n_views, cols)      # (V, C)

    weight = torch.ones_like(beta_2d)

    # Region 1: 0 <= beta < 2(delta - alpha)
    cond1 = beta_2d < 2.0 * (delta - alpha_2d)
    arg1 = (math.pi / 4) * beta_2d / (delta - alpha_2d + EPS_FP32)
    weight = torch.where(cond1, torch.sin(arg1) ** 2, weight)

    # Region 2: pi - 2*alpha < beta <= pi + 2*delta
    cond2 = (beta_2d > math.pi - 2.0 * alpha_2d) & (beta_2d <= math.pi + 2.0 * delta)
    arg2 = (math.pi / 4) * (math.pi + 2 * delta - beta_2d) / (alpha_2d + delta + EPS_FP32)
    weight = torch.where(cond2, torch.sin(arg2) ** 2, weight)

    # Beyond scan range
    cond_zero = beta_2d > math.pi + 2.0 * delta
    weight = torch.where(cond_zero, torch.zeros_like(weight), weight)

    return sino * weight.unsqueeze(0)


def ramp_filter_sinogram(sino: torch.Tensor) -> torch.Tensor:
    """Ramp filter along the column (detector u) dimension via FFT (Hann-windowed)."""
    rows, views, cols = sino.shape
    n_pad = 2 ** int(math.ceil(math.log2(2 * cols)))

    freqs = torch.fft.fftfreq(n_pad, device=sino.device)
    ramp = torch.abs(freqs)
    ramp = ramp * (0.5 + 0.5 * torch.cos(2 * math.pi * freqs))  # Hann window

    sino_padded = F.pad(sino, (0, n_pad - cols))
    sino_fft = torch.fft.fft(sino_padded, dim=-1)
    sino_filt = torch.fft.ifft(sino_fft * ramp, dim=-1).real[..., :cols]
    return sino_filt


def cone_beam_simulate(
    v_dynamic: list[torch.Tensor],
    voxel_size_mm: tuple[float, float, float],
    scan_params: ScanParams,
    cardiac_period_ms: float,
) -> torch.Tensor:
    """Cone-beam scan + Parker-weighted FBP reconstruction.

    Each projection view is computed from the dynamic-volume snapshot at
    the cardiac phase corresponding to its acquisition time, simulating
    inter-view motion → motion artifact in the reconstructed image.

    Args:
        v_dynamic: list of n_phases volumes, each (Z, Y, X) HU on GPU.
        voxel_size_mm: (z, y, x) mm.
        scan_params: ScanParams.
        cardiac_period_ms: T (ms) — used for time-→-phase mapping.

    Returns:
        V_corrupted (Z, Y, X) reconstructed image.
    """
    import tomosipo as ts  # local import — not all callers need it

    device = v_dynamic[0].device
    z_size, y_size, x_size = v_dynamic[0].shape
    n_phases = len(v_dynamic)

    vol_size_z = z_size * voxel_size_mm[0]
    vol_size_y = y_size * voxel_size_mm[1]
    vol_size_x = x_size * voxel_size_mm[2]
    vg = ts.volume(shape=(z_size, y_size, x_size),
                   size=(vol_size_z, vol_size_y, vol_size_x))

    # ---- Angles + per-view phase assignment ----
    fan_angle = math.atan(
        scan_params.detector_n_cols * scan_params.detector_col_pitch_mm
        / 2 / scan_params.sdd_mm
    )
    scan_range = math.pi + 2 * fan_angle  # Parker short scan

    initial = math.radians(scan_params.initial_scan_angle_deg)
    angles_rad = initial + torch.linspace(
        0, scan_range, scan_params.n_views, device=device
    )

    scan_time_ms = scan_params.rotation_period_ms * (scan_range / (2 * math.pi))
    times_ms = torch.linspace(0, scan_time_ms, scan_params.n_views, device=device)
    phase_idx_per_view = (
        (times_ms / cardiac_period_ms * n_phases).long() % n_phases
    )

    det_size_v = scan_params.detector_n_rows * scan_params.detector_row_pitch_mm
    det_size_u = scan_params.detector_n_cols * scan_params.detector_col_pitch_mm

    # ---- Group views by phase, project each phase's chunk ----
    sinogram = torch.zeros(
        scan_params.detector_n_rows,
        scan_params.n_views,
        scan_params.detector_n_cols,
        device=device,
    )

    for phase_idx in range(n_phases):
        view_mask = phase_idx_per_view == phase_idx
        if not bool(view_mask.any()):
            continue
        view_idx = torch.where(view_mask)[0]
        chunk_angles = angles_rad[view_idx].cpu().numpy()

        pg_chunk = ts.cone(
            angles=chunk_angles,
            shape=(scan_params.detector_n_rows, scan_params.detector_n_cols),
            size=(det_size_v, det_size_u),
            src_orig_dist=scan_params.sid_mm,
            src_det_dist=scan_params.sdd_mm,
        )
        a_chunk = ts.operator(vg, pg_chunk)
        sino_chunk = a_chunk(v_dynamic[phase_idx])  # (rows, n_chunk, cols)
        sinogram[:, view_idx, :] = sino_chunk

    # ---- Parker weighting + ramp filter ----
    sino_w = parker_short_scan_weight(sinogram, angles_rad, fan_angle, scan_params)
    sino_filt = ramp_filter_sinogram(sino_w)

    # ---- Backproject (FDK approximation) ----
    pg_full = ts.cone(
        angles=angles_rad.cpu().numpy(),
        shape=(scan_params.detector_n_rows, scan_params.detector_n_cols),
        size=(det_size_v, det_size_u),
        src_orig_dist=scan_params.sid_mm,
        src_det_dist=scan_params.sdd_mm,
    )
    a_full = ts.operator(vg, pg_full)
    v_corrupted = a_full.T(sino_filt)

    # NOTE: tomosipo's adjoint returns voxel-summed values; FDK exact scaling
    # depends on geometry (involves π / n_views and 1/SOD² weighting). We
    # leave HU calibration to the caller — see synthesize_motion_artifact()
    # which runs a no-motion baseline pass to compute the empirical scaling
    # factor matching V_clean's HU range.

    return v_corrupted


# ============================================================
# Top-level entry point
# ============================================================


DVFFn = "callable(grid_mm, motion_weight, anatomy, motion_params, t_ms) -> (Z,Y,X,3) DVF mm"


def synthesize_motion_artifact(
    v_clean: torch.Tensor,
    heart_mask: torch.Tensor,
    voxel_size_mm: tuple[float, float, float],
    motion_params: MotionParams | None = None,
    scan_params: ScanParams | None = None,
    n_phases: int = DEFAULT_N_PHASES,
    seed: int = 0,
    calibrate_hu: bool = True,
    dvf_fn=None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Generate one motion-corrupted volume from a clean ImageCAS scan.

    Args:
        v_clean: (Z, Y, X) clean CCTA volume in HU on GPU/CPU torch tensor.
        heart_mask: (Z, Y, X) {0, 1} whole-heart mask (e.g., TotalSegmentator).
        voxel_size_mm: (z, y, x) mm.
        motion_params: optional, defaults to MotionParams() defaults.
        scan_params: optional, defaults to ScanParams() defaults.
        n_phases: number of cardiac-cycle snapshots (default 48).
        seed: torch random seed (currently unused inside; reserved for noise).
        calibrate_hu: if True, run a no-motion baseline pass and rescale
            V_corrupted so that mean HU inside heart matches V_clean. This
            absorbs the FDK scale ambiguity introduced by tomosipo's adjoint.

    Returns:
        (V_corrupted, debug_info)
    """
    if motion_params is None:
        motion_params = MotionParams()
    if scan_params is None:
        scan_params = ScanParams()

    torch.manual_seed(seed)

    device = v_clean.device
    z_size, y_size, x_size = v_clean.shape

    anatomy = estimate_heart_anatomy(heart_mask, voxel_size_mm)

    # Soft motion-weight mask — computed once outside the per-phase loop.
    # See compute_motion_weight() docstring for Lossau 2019 / Maier 2021 alignment.
    motion_weight = compute_motion_weight(
        heart_mask, voxel_size_mm, motion_params.mask_smooth_sigma_mm
    )

    # ---- Physical mm grid ----
    z_coords = torch.arange(z_size, device=device).float() * voxel_size_mm[0]
    y_coords = torch.arange(y_size, device=device).float() * voxel_size_mm[1]
    x_coords = torch.arange(x_size, device=device).float() * voxel_size_mm[2]
    g_z, g_y, g_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
    grid_mm = torch.stack([g_z, g_y, g_x], dim=-1)

    # ---- Dynamic volume sequence ----
    phase_times_ms = torch.linspace(
        0, motion_params.cardiac_period_ms, n_phases + 1, device=device
    )[:-1]

    dvf_generator = dvf_fn if dvf_fn is not None else parametric_dvf
    v_dynamic = []
    for t in phase_times_ms:
        dvf = dvf_generator(grid_mm, motion_weight, anatomy, motion_params, t.item())
        v_dynamic.append(warp_volume(v_clean, grid_mm, dvf, voxel_size_mm))

    # ---- Cone-beam scan + FBP ----
    v_corrupted = cone_beam_simulate(
        v_dynamic, voxel_size_mm, scan_params, motion_params.cardiac_period_ms
    )

    # ---- HU calibration (optional) ----
    scale_factor = 1.0
    hu_offset = 0.0
    if calibrate_hu:
        # No-motion baseline: send V_clean (a single static phase) through
        # the same forward+FBP path. Compute scale + offset to match heart
        # interior mean HU.
        v_static = [v_clean] * n_phases
        v_baseline = cone_beam_simulate(
            v_static, voxel_size_mm, scan_params, motion_params.cardiac_period_ms
        )
        clean_inside = v_clean[heart_mask > 0]
        baseline_inside = v_baseline[heart_mask > 0]
        # Fit linear: V_clean ≈ a * V_baseline + b
        base_mean = baseline_inside.mean()
        base_std = baseline_inside.std() + EPS_FP32
        clean_mean = clean_inside.mean()
        clean_std = clean_inside.std() + EPS_FP32
        scale_factor = (clean_std / base_std).item()
        hu_offset = (clean_mean - scale_factor * base_mean).item()
        v_corrupted = v_corrupted * scale_factor + hu_offset

    debug = {
        "anatomy": {
            "centroid_mm": anatomy["centroid_mm"].cpu().tolist(),
            "long_axis": anatomy["long_axis"].cpu().tolist(),
            "radius_mm": anatomy["radius_mm"],
            "axial_extent_mm": anatomy["axial_extent_mm"],
        },
        "motion_params": motion_params.__dict__,
        "scan_params": scan_params.__dict__,
        "n_phases": n_phases,
        "seed": seed,
        "hu_calibration": {"scale": scale_factor, "offset": hu_offset},
    }
    return v_corrupted, debug
