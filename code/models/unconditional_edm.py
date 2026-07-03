"""unconditional_edm.py — unconditional Karras-EDM image prior p(x) for DPS reconstruction.

The sparse-view CCTA pivot needs a clean cardiac IMAGE prior (our residual_edm is
conditioned on motion inputs and cannot serve). This is a plain unconditional 3D EDM over
clean CCTA volumes: denoiser D_theta(x + n*sigma, sigma) -> x. Reuses the Karras
preconditioning/loss from residual_edm, minus the conditioning and residual target.

Sampler = deterministic 2nd-order Heun (EDM). The denoiser is the seam DPS plugs into:
data-consistency ||A_sparse x - y||^2 is applied on the Tweedie x_hat between reverse steps.

Plan: quality_reports/plans/2026-07-03_sparse-view-ccta-reconstruction-pivot.md (Stage 1)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _b(sigma: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return sigma.view(-1, *([1] * (ref.dim() - 1)))


class UnconditionalEDM(nn.Module):
    """Karras EDM for an unconditional image prior over clean volumes."""

    def __init__(self, denoiser: nn.Module, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 sigma_data: float = 0.5, rho: float = 7.0, P_mean: float = -1.2, P_std: float = 1.2) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigma_min, self.sigma_max, self.sigma_data = float(sigma_min), float(sigma_max), float(sigma_data)
        self.rho, self.P_mean, self.P_std = float(rho), float(P_mean), float(P_std)

    def c_skip(self, s): return self.sigma_data ** 2 / (s ** 2 + self.sigma_data ** 2)
    def c_out(self, s): return s * self.sigma_data / (s ** 2 + self.sigma_data ** 2).sqrt()
    def c_in(self, s): return 1.0 / (s ** 2 + self.sigma_data ** 2).sqrt()
    def c_noise(self, s): return 0.25 * torch.log(s.clamp_min(1e-20))
    def loss_weight(self, s): return (s ** 2 + self.sigma_data ** 2) / (s * self.sigma_data) ** 2

    def denoise(self, x_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Tweedie clean estimate x_hat = D(x_noisy; sigma)."""
        s5 = _b(sigma, x_noisy)
        net = self.denoiser(self.c_in(s5) * x_noisy, self.c_noise(sigma))
        return self.c_skip(s5) * x_noisy + self.c_out(s5) * net

    def loss(self, clean: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        b = clean.shape[0]
        sigma = (self.P_mean + self.P_std * torch.randn(b, device=clean.device)).exp()
        s5 = _b(sigma, clean)
        noisy = clean + s5 * torch.randn_like(clean)
        pred = self.denoise(noisy, sigma)
        per = F.mse_loss(pred, clean, reduction="none").flatten(1).mean(1)
        loss = (per * self.loss_weight(sigma)).mean()
        return loss, {"loss/edm": float(loss.detach()), "sigma/mean": float(sigma.mean())}

    def sample_sigmas(self, n: int, device) -> torch.Tensor:
        i = torch.arange(n, device=device)
        smin_r, smax_r = self.sigma_min ** (1 / self.rho), self.sigma_max ** (1 / self.rho)
        sig = (smax_r + i / max(n - 1, 1) * (smin_r - smax_r)) ** self.rho
        return torch.cat([sig, sig.new_zeros(1)])  # append sigma=0

    @torch.no_grad()
    def sample(self, shape, device, num_steps: int = 32, seed: int | None = 0) -> torch.Tensor:
        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        sigmas = self.sample_sigmas(num_steps, device)
        x = sigmas[0] * torch.randn(shape, generator=gen, device=device)
        for i in range(num_steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            x_hat = self.denoise(x, s.expand(shape[0]))
            d = (x - x_hat) / s
            x_next = x + (s_next - s) * d
            if s_next > 0:  # Heun 2nd-order correction
                x_hat2 = self.denoise(x_next, s_next.expand(shape[0]))
                d2 = (x_next - x_hat2) / s_next
                x_next = x + (s_next - s) * 0.5 * (d + d2)
            x = x_next
        return x
