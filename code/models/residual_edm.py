"""EDM wrapper for U-Net-conditioned posterior residual diffusion."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from code.evaluation.metrics import gradient_magnitude


def _broadcast_sigma(sigma: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return sigma.view(-1, *([1] * (ref.dim() - 1)))


class ResidualEDM(nn.Module):
    """Karras EDM for residual targets with multi-channel conditioning.

    Training target:

    ```text
    residual = clean - frozen_unet(corrupted)
    condition = [corrupted, frozen_unet(corrupted), corrupted - frozen_unet(corrupted)]
    ```

    The denoiser predicts the clean residual. The corrected image is formed
    outside the sampler as `condition[:, 1:2] + predicted_residual`.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        target_channels: int = 1,
        condition_channels: int = 3,
        sigma_min: float = 0.0005,
        sigma_max: float = 1.0,
        sigma_data: float = 0.05,
        rho: float = 7.0,
        P_mean: float = -3.0,
        P_std: float = 1.2,
        S_churn: float = 0.0,
        S_tmin: float = 0.001,
        S_tmax: float = 1.0,
        S_noise: float = 1.003,
        clip_pred: bool = False,
        clip_value: float = 1.0,
        final_l1_weight: float = 0.0,
        residual_l1_weight: float = 0.0,
        gradient_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.target_channels = int(target_channels)
        self.condition_channels = int(condition_channels)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.rho = float(rho)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.S_churn = float(S_churn)
        self.S_tmin = float(S_tmin)
        self.S_tmax = float(S_tmax)
        self.S_noise = float(S_noise)
        self.clip_pred = bool(clip_pred)
        self.clip_value = float(clip_value)
        self.final_l1_weight = float(final_l1_weight)
        self.residual_l1_weight = float(residual_l1_weight)
        self.gradient_weight = float(gradient_weight)

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return 0.25 * torch.log(sigma.clamp_min(1e-20))

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def sample_training_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return (self.P_mean + self.P_std * torch.randn(batch_size, device=device)).exp()

    def _validate_inputs(self, residual: torch.Tensor, condition: torch.Tensor) -> None:
        if residual.dim() != 5 or condition.dim() != 5:
            raise ValueError("residual and condition must be 5D tensors")
        if residual.shape[1] != self.target_channels:
            raise ValueError(
                f"expected residual C={self.target_channels}, got {residual.shape[1]}"
            )
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected condition C={self.condition_channels}, got {condition.shape[1]}"
            )
        if residual.shape[0] != condition.shape[0]:
            raise ValueError("residual and condition batch sizes differ")
        if tuple(residual.shape[-3:]) != tuple(condition.shape[-3:]):
            raise ValueError("residual and condition spatial shapes differ")

    def _denoise(
        self,
        residual_noisy: torch.Tensor,
        condition: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted clean residual."""
        sig5 = _broadcast_sigma(sigma, residual_noisy)
        model_input = torch.cat([self.c_in(sig5) * residual_noisy, condition], dim=1)
        net_out = self.denoiser(model_input, self.c_noise(sigma))
        pred = self.c_skip(sig5) * residual_noisy + self.c_out(sig5) * net_out
        if self.clip_pred:
            pred = pred.clamp(-self.clip_value, self.clip_value)
        return pred

    def loss(
        self,
        residual_target: torch.Tensor,
        condition: torch.Tensor,
        initial: Optional[torch.Tensor] = None,
        clean: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """EDM residual loss plus optional image-domain auxiliary losses."""
        self._validate_inputs(residual_target, condition)
        batch_size = residual_target.shape[0]
        sigma = self.sample_training_sigma(batch_size, residual_target.device)
        sig5 = _broadcast_sigma(sigma, residual_target)
        noisy = residual_target + sig5 * torch.randn_like(residual_target)
        residual_pred = self._denoise(noisy, condition, sigma)

        per_elem = F.mse_loss(residual_pred, residual_target, reduction="none")
        per_sample = per_elem.flatten(1).mean(dim=1)
        edm_loss = (per_sample * self.loss_weight(sigma)).mean()
        total = edm_loss
        metrics = {
            "loss/total": float(total.detach()),
            "loss/edm": float(edm_loss.detach()),
            "loss/residual_l1": 0.0,
            "loss/final_l1": 0.0,
            "loss/gradient": 0.0,
            "sigma/mean": float(sigma.mean().detach()),
        }

        if self.residual_l1_weight > 0:
            residual_l1 = F.l1_loss(residual_pred, residual_target)
            total = total + self.residual_l1_weight * residual_l1
            metrics["loss/residual_l1"] = float(residual_l1.detach())

        if self.final_l1_weight > 0 or self.gradient_weight > 0:
            if initial is None or clean is None:
                raise ValueError("initial and clean are required for image-domain auxiliary losses")
            final_pred = initial + residual_pred
            if self.final_l1_weight > 0:
                final_l1 = F.l1_loss(final_pred, clean)
                total = total + self.final_l1_weight * final_l1
                metrics["loss/final_l1"] = float(final_l1.detach())
            if self.gradient_weight > 0:
                gradient = F.l1_loss(gradient_magnitude(final_pred), gradient_magnitude(clean))
                total = total + self.gradient_weight * gradient
                metrics["loss/gradient"] = float(gradient.detach())

        metrics["loss/total"] = float(total.detach())
        return total, metrics

    def sample_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        if num_steps < 2:
            raise ValueError(f"num_steps must be >= 2, got {num_steps}")
        i = torch.arange(num_steps, device=device, dtype=torch.float32)
        inv_rho = 1.0 / self.rho
        s_max = self.sigma_max ** inv_rho
        s_min = self.sigma_min ** inv_rho
        sigmas = (s_max + i / (num_steps - 1) * (s_min - s_max)) ** self.rho
        return F.pad(sigmas, (0, 1), value=0.0)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 32,
        n_samples: int = 1,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample residuals conditioned on `[corrupted, initial, difference]`.

        Returns a tensor shaped `(B*n_samples, target_channels, D, H, W)`.
        """
        if condition.dim() != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected condition `(B, {self.condition_channels}, D, H, W)`, "
                f"got {tuple(condition.shape)}"
            )
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        device = condition.device
        spatial = tuple(int(v) for v in condition.shape[-3:])
        if n_samples > 1:
            condition = condition.repeat_interleave(n_samples, dim=0)
        batch_eff = condition.shape[0]
        noise_shape = (batch_eff, self.target_channels, *spatial)
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            init_noise = torch.randn(noise_shape, generator=generator, device=device)
        else:
            init_noise = torch.randn(noise_shape, device=device)

        sigmas = self.sample_schedule(num_steps, device)
        churn = 0.0 if deterministic else self.S_churn
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(churn / num_steps, math.sqrt(2) - 1),
            0.0,
        )
        residual = sigmas[0] * init_noise

        for idx in range(num_steps):
            sigma = float(sigmas[idx].item())
            sigma_next = float(sigmas[idx + 1].item())
            gamma = float(gammas[idx].item())

            sigma_hat = sigma + gamma * sigma
            if gamma > 0:
                eps = self.S_noise * torch.randn_like(residual)
                residual = residual + math.sqrt(max(sigma_hat ** 2 - sigma ** 2, 0.0)) * eps

            sig_t = torch.full((batch_eff,), sigma_hat, device=device)
            d_cur = (residual - self._denoise(residual, condition, sig_t)) / sigma_hat
            residual_next = residual + (sigma_next - sigma_hat) * d_cur

            if sigma_next > 0:
                sig_n = torch.full((batch_eff,), sigma_next, device=device)
                d_prime = (
                    residual_next - self._denoise(residual_next, condition, sig_n)
                ) / sigma_next
                residual_next = residual + 0.5 * (sigma_next - sigma_hat) * (d_cur + d_prime)

            residual = residual_next

        return residual
