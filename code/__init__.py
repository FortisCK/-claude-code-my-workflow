"""Cardiac CT motion-artifact correction — root package.

Submodules:
    data        — dataset loaders, PAD pipeline, transform pipelines
    models      — VAE-KL, latent denoiser, condition modules
    training    — training entry-points + YAML configs
    evaluation  — PSNR / SSIM / UQ / downstream-task pipelines
    inference   — posterior sampling, N-sample ensembling, DPS guidance

See `code/README.md` for layout + conventions.
"""

__version__ = "0.1.0"
