---
paths:
  - "code/**/*.py"
  - "scripts/python/**/*.py"
  - "**/*.ipynb"
---

# Python Code Standards (PyTorch + MONAI for cardiac CT diffusion)

**Standard:** Senior Principal ML Engineer + PhD researcher quality.

The project's primary code stack is Python 3.10+ / PyTorch / MONAI
`GenerativeModels`. These conventions apply to anything under `code/`,
`scripts/python/`, and any committed Jupyter notebook.

---

## 1. Reproducibility

- **Single seed at the top.** Every entry-point script calls
  `monai.utils.set_determinism(seed=...)` exactly once, immediately after imports.
  Never re-seed inside a training loop or function.
- **Record the seed in the run card.** Required by
  [`experiments-protocol.md`](experiments-protocol.md).
- **`cudnn` benchmark vs deterministic.** Default to `torch.backends.cudnn.benchmark = True`
  for training speed; switch to `cudnn.deterministic = True` only for
  numerical-replication runs (and document this in the run card).
- **Git SHA in artefacts.** Every checkpoint, every saved metric, every figure
  embeds the git SHA at training start (e.g., in the YAML config emitted to
  `experiments/runs/<...>/config_resolved.yaml`).
- **No `np.random.*` or `random.*` without going through `set_determinism`.**
  Use `torch.Generator` for explicit-stream randomness in dataloaders.

## 2. Path discipline

- **Relative paths only.** Never write `/Users/...`, `/home/...`, `C:\\...`
  in code. All filesystem paths originate from `code/data/paths.py`, which
  reads `data/.paths.local` (gitignored).
- **`pathlib.Path` everywhere.** No `os.path.join` in new code.
- **Output directories:** `Path(out_dir).mkdir(parents=True, exist_ok=True)`.

## 3. PyTorch idioms

- `model.train()` / `model.eval()` parity: every eval pass must `model.eval()`
  before, and the calling code restores `model.train()` after if applicable.
- Wrap every eval pass in `with torch.no_grad():` (or `torch.inference_mode()`
  for hot paths).
- AMP via `torch.autocast("cuda", dtype=torch.float16)` and
  `torch.cuda.amp.GradScaler`. Do not use the deprecated `apex` package.
- DataLoader: `pin_memory=True`, `num_workers >= 4` for ImageCAS, `persistent_workers=True`.
- `torch.compile` is opt-in per-script (not global); document the speedup or
  the failure mode in the run card.

## 4. MONAI idioms

- Inputs to model `forward` stay as `MetaTensor`; metadata propagates.
- Eval boundary: `from monai.data import decollate_batch` then iterate per-sample.
- Full-volume inference uses `monai.inferers.SlidingWindowInferer`; document
  the patch size, overlap, and gaussian-mask choice in the run card.
- Transforms: build pipelines with `monai.transforms.Compose([...])`. Avoid
  ad-hoc per-batch tensor shuffling — every preprocessing op should appear in
  a `Compose` for inspectability.
- Generative models: prefer `monai.networks.nets.AutoencoderKL` and
  `monai.networks.nets.DiffusionModelUNet` over rolling our own unless we
  have a specific reason (documented in code comments).

## 5. GPU-memory hygiene (single A6000 / 48 GB)

- Steady-state memory should leave at least 4 GB headroom (so DPS guidance,
  N-sample posterior ensembling, eval pipelines have burst capacity).
- `torch.cuda.empty_cache()` between phases is a smell — usually means a
  leak. If it's genuinely needed, write a comment explaining the leak path
  and why `del + gc.collect()` isn't enough.
- Activation checkpointing (`torch.utils.checkpoint.checkpoint`) is allowed
  but documented in the run card; it changes wall-clock vs memory tradeoff.
- Mixed precision is the default. FP32 must be justified.

## 6. Function & module design

- `snake_case` everywhere; modules and packages are short lowercase nouns.
- Public functions have type hints on parameters and return value.
- Public classes have a one-line docstring; complex ones get a Numpy-style
  block (`Parameters`, `Returns`, `Raises`).
- Default arguments are immutable (no mutable defaults like `def f(x=[])`).
- No magic numbers in function bodies — promote to module-level constants
  with `UPPER_SNAKE_CASE` and a comment about why this value.
- Configurable values live in YAML configs, loaded via OmegaConf, not in
  Python literals.

## 7. Entry-point script header

```python
"""train_vae.py — pretrain the 3D KL-VAE on ImageCAS.

Usage:
    python -m code.training.train_vae --config code/training/configs/vae_v1.yaml

Inputs:
    - ImageCAS preprocessed volumes (data/imagecas/processed/v1/)
    - YAML config

Outputs:
    - Checkpoint at experiments/checkpoints/<run-slug>/
    - Resolved config at experiments/runs/<run-slug>/config_resolved.yaml
    - WandB run named after the slug
"""
```

A header doc**is** the script's contract. The `python-reviewer` agent flags
training scripts whose header doesn't reference a run-card slug.

## 8. Console output discipline

- Use Python's `logging` module, not `print`. The training entry-point
  configures the root logger once.
- One log line per epoch boundary; per-step metrics go to WandB, not stdout.
- No banners, no decorative separators, no per-step `print` debugging
  committed to the repo.

## 9. Numerical discipline (PyTorch flavour)

- **No float `==`.** Use `torch.allclose(a, b, atol=...)`.
- **Clamp probabilities** before passing to `log` / `digamma` / etc.:
  `p = p.clamp(EPS, 1 - EPS)`. Project epsilon: `EPS = 1e-7` for FP32,
  `1e-4` for FP16 (the FP16 ε is the headline trap; document the choice).
- **`reduce` discipline.** `torch.mean`, `torch.sum` are unambiguous; avoid
  `np.mean` on detached tensors unless you've actually called `.cpu().numpy()`.
- **`einsum` over manual broadcasts** when the shape contract is non-obvious.
- **Stable softmax / logsumexp** — use the torch built-ins, never roll your own.

## 10. Code-quality tooling

| Tool   | Purpose                            | Config                         |
| ------ | ---------------------------------- | ------------------------------ |
| `black`| autoformat (line length 100)       | `pyproject.toml` (added at first install) |
| `ruff` | lint + import sort                 | `pyproject.toml`               |
| `mypy` | static types (loose mode)          | `pyproject.toml`               |
| `pytest`| tests for `code/` library         | `code/tests/`                  |

These are not enforced via pre-commit hook yet — the human (and the
`python-reviewer` agent) is the gate. Add a hook once the toolchain is pinned.

## 11. What this rule does NOT cover

- Specific architectures, loss functions, or hyperparameter choices — those
  belong in code, not in conventions.
- Notebook quality bar — notebooks under `code/notebooks/` are exploratory;
  if a notebook produces a run-worthy result, the result must be ported to a
  `code/training/` or `code/evaluation/` script before going in the run card.
- Data-format choices — see [`../../data/README.md`](../../data/README.md).

## 12. Code-quality checklist

```
[ ] set_determinism(seed=...) called once at top
[ ] All paths relative; no /Users/... in code
[ ] model.train()/eval() parity around eval passes
[ ] AMP via torch.autocast (not apex)
[ ] MetaTensor → decollate_batch at eval boundary
[ ] SlidingWindowInferer for full-volume eval
[ ] No torch.cuda.empty_cache() unless commented
[ ] Type hints on public functions
[ ] No magic numbers; configs in YAML
[ ] Logging, not print
[ ] No float == ; probabilities clamped with EPS
[ ] Header docstring lists run-card slug + canonical invocation
```
