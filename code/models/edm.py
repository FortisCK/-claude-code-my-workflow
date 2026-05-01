"""edm.py — Karras-2022 EDM math for conditional latent diffusion.

Ports the math from `external/HM-EDM/diffusion_models/conditional_EDM_3D.py`
(MIT licensed, Phil Wang) into a clean wrapper that:

    1. Uses MONAI's `DiffusionModelUNet` (via `ConditionalDenoiser`) instead
       of the lucidrains UNet.
    2. Implements channel-concat conditioning: the EDM forward concatenates
       z_t with z_cond along the channel axis and feeds the (2C, D,H,W)
       tensor to the denoiser. The denoiser predicts a (C, D,H,W) clean z.
    3. Strips lucidrains-specific dependencies (einops, conditional_diffusion
       flag, model.channels access).

EDM math is unchanged: σ-weighted MSE training loss, log-normal σ sampling,
ρ=7 schedule, Heun-2 sampler with stochastic churn (Algorithm 2 in Karras 2022).

References
----------
- Karras, Aittala, Aila, Laine. "Elucidating the Design Space of Diffusion-Based
  Generative Models." NeurIPS 2022.
- Chen et al. "HM-EDM" 2024 — medical-CT precedent we adapted.

License
-------
HM-EDM source: MIT (Phil Wang, lucidrains).
This port carries the same notice in module docstring per its terms.

Per `.claude/rules/python-code-conventions.md`:
    - type hints on public methods
    - immutable defaults
    - no magic numbers in function bodies (EDM hyperparams are documented)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _broadcast_sigma(sigma: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape (B,) → (B, 1, 1, 1, 1) for broadcasting against a 5D tensor."""
    return sigma.view(-1, *([1] * (ref.dim() - 1)))


class EDM(nn.Module):
    """Karras-2022 EDM wrapping a `ConditionalDenoiser` for concat conditioning.

    Parameters
    ----------
    denoiser : nn.Module
        A model with `forward(x: (B, 2C, D,H,W), t: (B,)) → (B, C, D,H,W)`.
        The first C channels of `x` are the noisy `z_t`; the remaining C are
        `z_cond` (clean conditioning latent from the corrupted volume).
    sigma_min, sigma_max : float
        EDM noise-level bounds (Karras 2022, Table 5).
    sigma_data : float
        Estimated std of the *latent* training distribution. Default 0.5
        is a starting point; should be re-estimated from VAE-encoded training
        data after Stage-1 finishes (logged in run card).
    rho : float
        Schedule curvature exponent (default 7, Karras 2022 §5).
    P_mean, P_std : float
        Log-normal parameters for σ during training.
    S_churn, S_tmin, S_tmax, S_noise : float
        Stochastic-sampling parameters (Algorithm 2). Set S_churn=0 for
        deterministic Heun.
    clip_pred : bool
        Clip the predicted clean image to ±clip_value at every step.
    clip_value : float
        Clip range for predicted clean image (latent stats permitting).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        latent_channels: int = 4,
        latent_size: tuple[int, int, int] = (24, 24, 24),
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        S_churn: float = 80.0,
        S_tmin: float = 0.05,
        S_tmax: float = 50.0,
        S_noise: float = 1.003,
        clip_pred: bool = False,
        clip_value: float = 3.0,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.clip_pred = clip_pred
        self.clip_value = clip_value

    # ---- preconditioning (Karras 2022, Table 1) ----

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma.clamp_min(1e-20))

    # ---- preconditioned network forward (eq. 7) ----

    def _denoise(
        self, z_noisy: torch.Tensor, z_cond: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Run the denoiser through Karras preconditioning; returns predicted clean z."""
        sig5 = _broadcast_sigma(sigma, z_noisy)
        # concat conditioning ON the c_in-scaled noisy input only;
        # z_cond is appended at full magnitude.
        z_input = torch.cat([self.c_in(sig5) * z_noisy, z_cond], dim=1)
        net_out = self.denoiser(z_input, self.c_noise(sigma))
        z_pred = self.c_skip(sig5) * z_noisy + self.c_out(sig5) * net_out
        if self.clip_pred:
            z_pred = z_pred.clamp(-self.clip_value, self.clip_value)
        return z_pred

    # ---- training loss ----

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def sample_training_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Log-normal σ ~ exp(P_mean + P_std·𝓝(0,1))."""
        return (self.P_mean + self.P_std * torch.randn(batch_size, device=device)).exp()

    def loss(self, z_clean: torch.Tensor, z_cond: torch.Tensor) -> torch.Tensor:
        """EDM training loss: σ-weighted MSE between predicted-clean and clean latent.

        Args:
            z_clean: (B, C, D, H, W) — VAE-encoded clean latent (target)
            z_cond:  (B, C, D, H, W) — VAE-encoded corrupted-volume latent (condition)
        """
        b = z_clean.shape[0]
        sigma = self.sample_training_sigma(b, z_clean.device)
        sig5 = _broadcast_sigma(sigma, z_clean)
        noise = torch.randn_like(z_clean)
        z_noisy = z_clean + sig5 * noise
        z_pred = self._denoise(z_noisy, z_cond, sigma)
        per_elem = F.mse_loss(z_pred, z_clean, reduction="none")
        per_sample = per_elem.flatten(1).mean(dim=1)
        return (per_sample * self.loss_weight(sigma)).mean()

    # ---- sampling ----

    def sample_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """ρ-shaped descending σ schedule from σ_max to σ_min, plus 0."""
        i = torch.arange(num_steps, device=device, dtype=torch.float32)
        inv_rho = 1.0 / self.rho
        s_max = self.sigma_max ** inv_rho
        s_min = self.sigma_min ** inv_rho
        sigmas = (s_max + i / (num_steps - 1) * (s_min - s_max)) ** self.rho
        # Final sigma is exactly 0 for the last Heun step.
        return F.pad(sigmas, (0, 1), value=0.0)

    @torch.no_grad()
    def sample(
        self,
        z_cond: torch.Tensor,
        num_steps: int = 50,
        n_samples: int = 1,
        deterministic: bool = False,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Heun-2 EDM sampler (Algorithm 2, Karras 2022).

        Args:
            z_cond: (B, C, D, H, W) — conditioning latent.
            num_steps: number of Heun steps (50 recommended).
            n_samples: number of independent samples PER condition (replicates
                z_cond along the batch dim and starts each from independent noise).
            deterministic: if True, force S_churn = 0 (no stochastic noise injection).
            seed: optional seed for the initial noise (per-call reproducibility).

        Returns:
            (B*n_samples, C, D, H, W) tensor of denoised latents in original
            order: [c0_s0, c0_s1, ..., c0_sN, c1_s0, ...].
        """
        device = z_cond.device
        b, c = z_cond.shape[:2]
        if n_samples > 1:
            z_cond = z_cond.repeat_interleave(n_samples, dim=0)
        b_eff = z_cond.shape[0]
        S_churn = 0.0 if deterministic else self.S_churn

        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            init_noise = torch.randn(z_cond.shape, generator=g, device=device)
        else:
            init_noise = torch.randn(z_cond.shape, device=device)

        sigmas = self.sample_schedule(num_steps, device)
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(S_churn / num_steps, math.sqrt(2) - 1),
            0.0,
        )
        z = sigmas[0] * init_noise

        for i in range(num_steps):
            sigma = sigmas[i].item()
            sigma_next = sigmas[i + 1].item()
            gamma = gammas[i].item()

            sigma_hat = sigma + gamma * sigma
            if gamma > 0:
                eps = self.S_noise * torch.randn_like(z)
                z = z + math.sqrt(max(sigma_hat ** 2 - sigma ** 2, 0.0)) * eps

            sig_t = torch.full((b_eff,), sigma_hat, device=device)
            d_cur = (z - self._denoise(z, z_cond, sig_t)) / sigma_hat
            z_next = z + (sigma_next - sigma_hat) * d_cur

            # 2nd-order Heun correction (skip on last step where sigma_next==0)
            if sigma_next > 0:
                sig_n = torch.full((b_eff,), sigma_next, device=device)
                d_prime = (z_next - self._denoise(z_next, z_cond, sig_n)) / sigma_next
                z_next = z + 0.5 * (sigma_next - sigma_hat) * (d_cur + d_prime)

            z = z_next

        return z
