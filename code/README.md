# `code/` — Project source

PyTorch + MONAI implementation of the 3D conditional latent-diffusion
motion-correction model.

> **Status:** skeleton only. Conventions below; nothing implemented yet.

## Stack

| Component | Choice                                                           |
| --------- | ---------------------------------------------------------------- |
| Language  | Python ≥ 3.10                                                    |
| DL        | PyTorch (version pinned at first install via `pyproject.toml`)   |
| Medical   | MONAI + `monai.generative` (`GenerativeModels`)                  |
| Config    | YAML configs under `code/<area>/configs/`, loaded via OmegaConf  |
| Logging   | Weights & Biases (`wandb`) — preferred. Run name = run-card slug. |
| Lint      | `ruff` + `black` + `mypy` (loose for now)                        |
| Tests     | `pytest`, scoped to library code (not training scripts)          |

## Layout

```
code/
├── data/           # ImageCAS loaders, PAD wrappers, transform pipelines
├── models/         # VAE-KL, latent denoiser, condition modules
├── training/       # train_vae.py, train_diffusion.py, configs/
├── evaluation/     # PSNR/SSIM, reliability diagrams, downstream tasks
├── inference/      # posterior sampling, N-sample ensembling, DPS guidance
└── notebooks/      # exploratory only — must not be imported by other code
```

## Conventions

- **Reproducibility.** Every entry-point script must call
  `monai.utils.set_determinism(seed=...)` exactly once at the top.
  The seed is recorded in the run card. See
  [`../.claude/rules/python-code-conventions.md`](../.claude/rules/python-code-conventions.md).
- **No absolute paths.** All paths are relative to the repo root. Data paths
  go through `code/data/paths.py` which reads from `data/.paths.local` (gitignored).
- **GPU-memory budget.** Single A6000, 48 GB. Default to FP16 + AMP via
  `torch.autocast`. Keep at least 4 GB headroom — heavy ops (DPS guidance,
  N-sample ensembling) burst above the steady-state. Document any approach
  that requires `torch.cuda.empty_cache()` — it usually means a leak.
- **Eval boundary.** All model `forward` calls inside training stay in
  `MetaTensor` space. Evaluation pipelines explicitly call
  `monai.data.decollate_batch(...)` before any per-sample metric.
- **Sliding-window inference.** Use `monai.inferers.SlidingWindowInferer` for
  full-volume eval. ProDM's 64×64×3 patch and HM-EDM's 128×128×50 patch are
  the literature anchors.
- **Run cards required.** Every training run must have a matching
  `experiments/runs/<timestamp>_<short-name>.md` per
  [`../.claude/rules/experiments-protocol.md`](../.claude/rules/experiments-protocol.md).
  No run card → not a real experiment.

## Entry-point convention

Every entry-point script (anything you'd run with `python ...`) starts with a
header docstring listing the canonical invocation:

```python
"""train_vae.py — pretrain the 3D KL-VAE on ImageCAS.

Usage:
    python -m code.training.train_vae --config code/training/configs/vae_v1.yaml
"""
```

## Cross-references

- [`../.claude/rules/python-code-conventions.md`](../.claude/rules/python-code-conventions.md) — full convention list (PyTorch / MONAI / reproducibility / GPU hygiene).
- [`../.claude/rules/experiments-protocol.md`](../.claude/rules/experiments-protocol.md) — run-card schema.
- [`../.claude/skills/review-python/SKILL.md`](../.claude/skills/review-python/SKILL.md) — how to invoke a Python code review.
- [`../experiments/README.md`](../experiments/README.md) — how runs are tracked.
- [`../data/README.md`](../data/README.md) — dataset acquisition + license notes.
