"""check_env.py — smoke-test the Python environment on LTSI.

Usage (after `pip install -e .`):
    python scripts/python/check_env.py

Verifies:
    1. All required imports succeed (with installed versions printed)
    2. CUDA-capable GPU detected (A6000 expected, 48 GB)
    3. monai-generative `AutoencoderKL` instantiates (catches subtle install bugs)

Exit code 0 = all checks pass. Exit code 1 = at least one failed.

Pair with the run-card `experiments/runs/<timestamp>_env-setup.md` (Week 1
Day 2; record stdout there for reproducibility).
"""

from __future__ import annotations

import importlib
import platform
import sys

# (import_name, pip_package_label_for_print)
REQUIRED: list[tuple[str, str | None]] = [
    ("torch", None),
    ("torchvision", None),
    ("monai", None),
    ("generative", "monai-generative"),  # imported as `generative`
    ("nibabel", None),
    ("pydicom", None),
    ("SimpleITK", None),
    ("numpy", None),
    ("scipy", None),
    ("skimage", "scikit-image"),
    ("sklearn", "scikit-learn"),
    ("wandb", None),
    ("omegaconf", None),
    ("yaml", "pyyaml"),
    ("tqdm", None),
    ("matplotlib", None),
    ("einops", None),
    ("kaggle", None),
]


def _try_import(modname: str, pkg_name: str | None = None) -> tuple[bool, str]:
    pkg = pkg_name or modname
    try:
        m = importlib.import_module(modname)
        version = getattr(m, "__version__", "?")
        return True, f"  [OK]   {pkg:<22s} v{version}"
    except Exception as exc:  # noqa: BLE001 — broad on purpose for env-check
        return False, f"  [FAIL] {pkg:<22s} ({type(exc).__name__}: {exc})"


def _gpu_check() -> tuple[bool, list[str]]:
    lines: list[str] = []
    try:
        import torch
    except ImportError as exc:
        return False, [f"  [FAIL] torch import: {exc}"]

    if not torch.cuda.is_available():
        lines.append("  [FAIL] torch.cuda.is_available() = False")
        return False, lines

    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        mem_gb = p.total_memory / (1024**3)
        lines.append(
            f"  [OK]   cuda:{i}  {p.name}  {mem_gb:.1f} GB  CC{p.major}.{p.minor}"
        )
    lines.append(f"  CUDA runtime : {torch.version.cuda}")
    lines.append(f"  cuDNN        : {torch.backends.cudnn.version()}")
    return True, lines


def _autoencoderkl_check() -> tuple[bool, str]:
    try:
        from generative.networks.nets import AutoencoderKL  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return False, f"  [FAIL] cannot import AutoencoderKL from generative: {exc}"

    try:
        m = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64),
            latent_channels=4,
            num_res_blocks=1,
        )
        n_params = sum(p.numel() for p in m.parameters())
        return True, f"  [OK]   AutoencoderKL toy 2D model: {n_params:,} parameters"
    except Exception as exc:  # noqa: BLE001
        return False, f"  [FAIL] AutoencoderKL instantiation: {exc}"


def main() -> int:
    print("=" * 64)
    print("Cardiac-CT diffusion env smoke test")
    print("=" * 64)
    print()
    print(f"Python    : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Platform  : {platform.platform()}")
    print()

    failures = 0

    print("Required imports:")
    for modname, pkg_name in REQUIRED:
        ok, line = _try_import(modname, pkg_name)
        print(line)
        failures += 0 if ok else 1
    print()

    print("GPU detection:")
    ok, lines = _gpu_check()
    for line in lines:
        print(line)
    failures += 0 if ok else 1
    print()

    print("monai-generative AutoencoderKL instantiation:")
    ok, line = _autoencoderkl_check()
    print(line)
    failures += 0 if ok else 1
    print()

    print("=" * 64)
    if failures == 0:
        print("All checks PASS. Ready to proceed (Week 1-2 Day 3+).")
        return 0
    else:
        print(f"{failures} check(s) FAILED. Review above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
