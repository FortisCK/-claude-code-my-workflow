"""stochastic_interpolant.py — Stochastic Interpolants for conditional latent generation.

Parallel to `code/models/edm.py`. Same `ConditionalDenoiser` backbone, same
channel-concat conditioning (z_t || z_cond → 2C input), same call surface
(`loss(z_clean, z_cond)` and `sample(z_cond, ...)`), so it is a drop-in
replacement when `train_diffusion.py` is invoked with `--engine flow`.

Stochastic Interpolants framework
---------------------------------
Albergo, Boffi, Vanden-Eijnden 2023 (arXiv 2303.08797, JMLR) unify flow- and
diffusion-based methods by interpolating between two probability densities
in finite time. We use the practical "one-sided Gaussian interpolant":

    x_t  = α(t) · ε  +  β(t) · z_clean              ε ∼ 𝒩(0, I)

where t ∈ [0, 1] satisfies α(0) = 1, α(1) = 0, β(0) = 0, β(1) = 1, so

    x_0 = ε              (pure noise)
    x_1 = z_clean        (target data)

Two schedules supported:

    "linear"  (rectified flow)         "trig"  (variance-preserving-ish)
    α(t) = 1 - t                       α(t) = cos(πt/2)
    β(t) = t                           β(t) = sin(πt/2)
    α'(t) = -1                         α'(t) = -(π/2) sin(πt/2)
    β'(t) = +1                         β'(t) = +(π/2) cos(πt/2)

Velocity target (the quantity the network regresses):

    v_target(t) = α'(t) · ε  +  β'(t) · z_clean

Training loss = unweighted MSE in velocity space:

    L = 𝔼_{t, z_clean, ε} ‖v_θ(x_t, z_cond, t) − v_target‖²

Sampling: integrate dx/dt = v_θ(x, z_cond, t) from t=0 (x = ε) to t=1.
Heun-2 by default. Optional Euler-Maruyama noise injection (`sde_noise_scale`)
for posterior-diversity SDE sampling — note this is a *pragmatic* SDE, not
the score-corrected SI SDE; for our N-sample-from-different-noise posterior
story, deterministic ODE already gives diversity, and `sde_noise_scale=0` is
the recommended default.

Why few-step works
------------------
Linear schedule = rectified flow: paths are straight lines, so single Euler
step is unbiased. We default to 8 Heun-2 steps (16 NFE) which matches the
Yazdani-2025 / Lipman-2024 recommendation for medical imaging.

References
----------
- Lipman, Chen, Ben-Hamu, Nickel, Le. "Flow Matching for Generative Modeling."
  ICLR 2023. arXiv 2210.02747.
- Liu, Gong, Liu. "Flow Straight and Fast: Rectified Flow." ICLR 2023 (Spotlight).
  arXiv 2209.03003.
- Albergo, Boffi, Vanden-Eijnden. "Stochastic Interpolants: A Unifying
  Framework for Flows and Diffusions." JMLR 2023. arXiv 2303.08797.
- Lipman et al. "Flow Matching Guide and Code." arXiv 2412.06264.
- Yazdani et al. "Flow Matching for Medical Image Synthesis." MICCAI 2025.

Per `.claude/rules/python-code-conventions.md`:
    - type hints on public methods
    - immutable defaults
    - no magic numbers in function bodies (schedule choices documented)
    - EPS for FP32 numerical clamping
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

EPS_FP32: float = 1.0e-7


def _broadcast_t(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape (B,) scalar timesteps to broadcast against a 5D tensor."""
    return t.view(-1, *([1] * (ref.dim() - 1)))


class StochasticInterpolant(nn.Module):
    """Stochastic Interpolants for conditional latent generation.

    Parameters
    ----------
    denoiser : nn.Module
        Network with signature `forward(x: (B, 2C, D, H, W), t: (B,)) → (B, C, D, H, W)`.
        First C channels = noisy x_t; trailing C = z_cond. Output predicts
        the velocity v(x_t, z_cond, t).
    latent_channels : int
        Channel count of z (one side of the cat).
    latent_size : tuple[int, int, int]
        Spatial size of z (info only; not used in math).
    schedule : {"linear", "trig"}
        Interpolant pair (α, β). "linear" = rectified flow (straight paths,
        few-step optimal). "trig" = variance-preserving-style (smoother near
        the endpoints).
    sde_noise_scale : float
        Coefficient for Euler-Maruyama noise injection at sampling time. 0 →
        pure deterministic ODE (recommended). Positive values add stochasticity
        but are NOT the score-corrected SI SDE — they are a pragmatic
        diversity knob. Keep 0 for the canonical FM/RF behaviour.
    clip_pred : bool
        If True, clip x at every sampling step (helps for unstable training).
    clip_value : float
        Symmetric clip range when `clip_pred` is True.
    """

    SCHEDULES = ("linear", "trig")

    def __init__(
        self,
        denoiser: nn.Module,
        latent_channels: int = 4,
        latent_size: tuple[int, int, int] = (24, 24, 24),
        schedule: str = "linear",
        sde_noise_scale: float = 0.0,
        clip_pred: bool = False,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        if schedule not in self.SCHEDULES:
            raise ValueError(f"schedule must be one of {self.SCHEDULES}, got {schedule!r}")
        self.denoiser = denoiser
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.schedule = schedule
        self.sde_noise_scale = float(sde_noise_scale)
        self.clip_pred = clip_pred
        self.clip_value = clip_value

    # ---- schedule (α, β) and their time-derivatives -------------------------

    def _ab(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (α(t), β(t)) for batched t ∈ [0, 1]."""
        if self.schedule == "linear":
            return 1.0 - t, t
        # "trig"
        half_pi_t = (math.pi / 2.0) * t
        return torch.cos(half_pi_t), torch.sin(half_pi_t)

    def _ab_dot(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (α'(t), β'(t)) for batched t ∈ [0, 1]."""
        if self.schedule == "linear":
            ones = torch.ones_like(t)
            return -ones, ones
        # "trig"
        half_pi = math.pi / 2.0
        half_pi_t = half_pi * t
        return -half_pi * torch.sin(half_pi_t), half_pi * torch.cos(half_pi_t)

    # ---- velocity from the denoiser (channel-concat conditioning) ----------

    def _velocity(
        self, x_t: torch.Tensor, z_cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Run the denoiser and interpret its output as the velocity v(x_t, z_cond, t)."""
        x_input = torch.cat([x_t, z_cond], dim=1)
        return self.denoiser(x_input, t)

    # ---- training loss -----------------------------------------------------

    def loss(self, z_clean: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        """Velocity-MSE training loss.

        Args:
            z_clean: (B, C, D, H, W) — VAE-encoded clean latent (data side, x_1).
            z_cond:  (B, C, D, H, W) — VAE-encoded corrupted latent (condition).
        """
        b = z_clean.shape[0]
        device = z_clean.device

        # Sample t ∈ U(0, 1) per element (clamped slightly inside to avoid the
        # endpoint in `trig` schedule where derivatives vanish or blow up).
        t = torch.rand(b, device=device).clamp(EPS_FP32, 1.0 - EPS_FP32)

        # Reference noise ε ∼ 𝒩(0, I)
        eps = torch.randn_like(z_clean)

        a, beta = self._ab(t)
        a5 = _broadcast_t(a, z_clean)
        b5 = _broadcast_t(beta, z_clean)

        x_t = a5 * eps + b5 * z_clean

        a_dot, b_dot = self._ab_dot(t)
        a_dot5 = _broadcast_t(a_dot, z_clean)
        b_dot5 = _broadcast_t(b_dot, z_clean)
        v_target = a_dot5 * eps + b_dot5 * z_clean

        v_pred = self._velocity(x_t, z_cond, t)
        return F.mse_loss(v_pred, v_target)

    # ---- sampling ----------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        z_cond: torch.Tensor,
        num_steps: int = 8,
        n_samples: int = 1,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Heun-2 ODE sampler (with optional SDE noise injection if non-deterministic).

        Args:
            z_cond: (B, C, D, H, W) conditioning latent.
            num_steps: number of Heun steps. 4-8 is typical for `linear` schedule.
            n_samples: posterior chains per condition. Each gets independent
                initial noise; z_cond is repeat-interleaved along the batch dim.
            deterministic: if True, pure ODE. If False AND `sde_noise_scale > 0`,
                inject Euler-Maruyama noise at each step.
            seed: optional initial-noise seed for reproducibility.

        Returns:
            (B*n_samples, C, D, H, W) — denoised latents grouped per condition:
            [c0_s0, c0_s1, ..., c0_s(N-1), c1_s0, ...].
        """
        device = z_cond.device

        if n_samples > 1:
            z_cond = z_cond.repeat_interleave(n_samples, dim=0)
        b_eff = z_cond.shape[0]

        # Initial noise = x at t=0
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(z_cond.shape, generator=g, device=device)
        else:
            x = torch.randn(z_cond.shape, device=device)

        sde_active = (not deterministic) and self.sde_noise_scale > 0.0

        # Uniform time grid t = 0, 1/N, 2/N, ..., 1
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t_i = ts[i]
            t_next = ts[i + 1]
            dt = (t_next - t_i).item()

            ti_b = torch.full((b_eff,), float(t_i), device=device)
            v_i = self._velocity(x, z_cond, ti_b)
            x_pred = x + v_i * dt

            # Heun second-order correction (skip on final step where t=1 anchor)
            if i < num_steps - 1:
                tn_b = torch.full((b_eff,), float(t_next), device=device)
                v_next = self._velocity(x_pred, z_cond, tn_b)
                x = x + 0.5 * dt * (v_i + v_next)
            else:
                x = x_pred

            # Pragmatic SDE: Euler-Maruyama noise injection for diversity
            if sde_active and i < num_steps - 1:
                x = x + self.sde_noise_scale * math.sqrt(abs(dt)) * torch.randn_like(x)

            if self.clip_pred:
                x = x.clamp(-self.clip_value, self.clip_value)

        return x
