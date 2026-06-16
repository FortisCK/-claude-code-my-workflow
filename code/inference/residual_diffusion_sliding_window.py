"""Full-volume sliding-window inference for posterior residual diffusion."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from monai.inferers import SlidingWindowInferer

from code.inference.unet_sliding_window import sliding_window_unet_correct
from code.models.residual_edm import ResidualEDM
from code.models.residual_refiner import ResidualRefinerUNet3D
from code.models.residual_unet import ResidualUNet3D

log = logging.getLogger(__name__)


def _as_3tuple(value: int | Sequence[int], name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    out = tuple(int(v) for v in value)
    if len(out) != 3:
        raise ValueError(f"{name} must contain exactly 3 values, got {value!r}")
    return out


@torch.inference_mode()
def sliding_window_residual_diffusion_correct(
    v_corrupted: torch.Tensor,
    initializer: ResidualUNet3D,
    engine: ResidualEDM,
    roi_size: int | Sequence[int] = (128, 128, 128),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    sw_device: torch.device | str = "cuda",
    output_device: torch.device | str = "cpu",
    blend_mode: str = "gaussian",
    sigma_scale: float = 0.125,
    num_steps: int = 32,
    n_samples: int = 1,
    residual_scale: float = 1.0,
    deterministic: bool = True,
    progress: bool = False,
    return_initial: bool = False,
    return_uncertainty: bool = False,
    return_single_sample: bool = False,
    single_sample_seed: int = 0,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Run frozen U-Net initializer followed by residual EDM sampling.

    When `return_single_sample` is True, a coherent single posterior sample
    (sample index 0, sharp — not averaged) is also stitched and returned, using a
    fixed `single_sample_seed` across all patches so overlapping windows share the
    same noise chain and the stitched single sample stays seam-consistent. This
    lets us compare the sharp single sample against the blurry posterior mean.
    """
    if v_corrupted.dim() != 5:
        raise ValueError(f"expected 5D `(B, C, D, H, W)` tensor, got {tuple(v_corrupted.shape)}")
    if v_corrupted.shape[1] != 1:
        raise ValueError(f"expected single-channel input, got C={v_corrupted.shape[1]}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if sw_batch_size < 1:
        raise ValueError(f"sw_batch_size must be >= 1, got {sw_batch_size}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    roi = _as_3tuple(roi_size, "roi_size")
    spatial = tuple(int(v) for v in v_corrupted.shape[-3:])
    log.info(
        "[residual diffusion sliding] shape=%s roi=%s overlap=%.2f steps=%d samples=%d",
        spatial,
        roi,
        overlap,
        num_steps,
        n_samples,
    )

    initial = sliding_window_unet_correct(
        v_corrupted,
        initializer,
        roi_size=roi,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        sw_device=sw_device,
        output_device=output_device,
        blend_mode=blend_mode,
        sigma_scale=sigma_scale,
        progress=progress,
    )
    condition = ResidualRefinerUNet3D.make_condition(v_corrupted.float(), initial.float())

    engine.eval()
    sw_dev = torch.device(sw_device)
    out_dev = torch.device(output_device)
    inferer = SlidingWindowInferer(
        roi_size=roi,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=blend_mode,
        sigma_scale=sigma_scale,
        sw_device=sw_dev,
        device=out_dev,
        progress=progress,
        cache_roi_weight_map=True,
    )

    def predictor(patches: torch.Tensor) -> torch.Tensor:
        cond = patches.to(sw_dev, non_blocking=True)
        batch = cond.shape[0]
        # Fixed seed only when we need a coherent single sample stitched across
        # overlapping windows; otherwise keep the mean path stochastic (seed=None).
        seed = single_sample_seed if return_single_sample else None
        residual_samples = engine.sample(
            cond,
            num_steps=num_steps,
            n_samples=n_samples,
            deterministic=deterministic,
            seed=seed,
        )
        residual_samples = residual_samples.view(
            batch,
            n_samples,
            engine.target_channels,
            *cond.shape[-3:],
        )
        residual_mean = residual_samples.mean(dim=1)
        corrected = cond[:, 1:2] + float(residual_scale) * residual_mean
        parts = [corrected]
        if return_uncertainty:
            residual_std = residual_samples.std(dim=1, unbiased=False) * abs(float(residual_scale))
            parts.append(residual_std)
        if return_single_sample:
            single = cond[:, 1:2] + float(residual_scale) * residual_samples[:, 0]
            parts.append(single)
        if len(parts) == 1:
            return corrected.float()
        return torch.cat(parts, dim=1).float()

    pred = inferer(condition.float(), predictor).float()
    # Channel order emitted by predictor: corrected, [residual_std], [single_sample]
    idx = 0
    corrected = pred[:, idx:idx + 1]
    idx += 1
    uncertainty = None
    single_sample = None
    if return_uncertainty:
        uncertainty = pred[:, idx:idx + 1]
        idx += 1
    if return_single_sample:
        single_sample = pred[:, idx:idx + 1]
        idx += 1

    # Return order: corrected, [initial], [uncertainty], [single_sample]
    outputs: list[torch.Tensor] = [corrected]
    if return_initial:
        outputs.append(initial.float())
    if return_uncertainty and uncertainty is not None:
        outputs.append(uncertainty.float())
    if return_single_sample and single_sample is not None:
        outputs.append(single_sample.float())
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)
