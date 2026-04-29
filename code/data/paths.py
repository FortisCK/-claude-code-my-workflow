"""paths.py — central registry of dataset / output paths.

All filesystem paths originate here. Code never hard-codes
`/Users/...`, `/home/...`, `/data/...` etc. — it asks `paths` to resolve.

Defaults are repo-relative; the local override file
`data/.paths.local` (gitignored) supplies machine-specific absolute paths.

Local-override format (INI-like; `data/.paths.local`):

    # comments OK
    IMAGECAS_RAW=/scratch/cmz/imagecas/raw
    IMAGECAS_PROCESSED=/scratch/cmz/imagecas/processed/v1
    PAD_REPO=/home/cmz/code/TT-U-Net
    HM_EDM_REPO=/home/cmz/code/Diffusion_for_CT_motion
    XCAT_DATA=/home/cmz/code/xcat-license-pending
    EXPERIMENTS_OUTPUT=/scratch/cmz/cardiac-experiments

Per `.claude/rules/python-code-conventions.md` §2 (Path discipline).
"""

from __future__ import annotations

from pathlib import Path

# Repo root = parent of `code/`.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Repo-relative defaults — override via data/.paths.local.
_REPO_DEFAULTS: dict[str, Path] = {
    "IMAGECAS_RAW": REPO_ROOT / "data" / "imagecas" / "raw",
    "IMAGECAS_PROCESSED": REPO_ROOT / "data" / "imagecas" / "processed" / "v1",
    "IMAGECAS_SPLITS": REPO_ROOT / "data" / "imagecas" / "splits",
    "PAD_REPO": REPO_ROOT / "data" / "pad",
    "HM_EDM_REPO": REPO_ROOT / "data" / "hm-edm",
    "XCAT_DATA": REPO_ROOT / "data" / "xcat",
    "EXPERIMENTS_OUTPUT": REPO_ROOT / "experiments" / "outputs",
}


def _read_local_overrides() -> dict[str, Path]:
    """Parse `data/.paths.local` (INI-like); empty dict if file absent."""
    path = REPO_ROOT / "data" / ".paths.local"
    if not path.exists():
        return {}
    out: dict[str, Path] = {}
    for line_num, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(
                f"data/.paths.local line {line_num}: expected KEY=VALUE, got {raw!r}"
            )
        key, _, value = line.partition("=")
        out[key.strip()] = Path(value.strip()).expanduser()
    return out


_RESOLVED: dict[str, Path] = {**_REPO_DEFAULTS, **_read_local_overrides()}


def get(key: str) -> Path:
    """Look up a path by symbolic key (e.g., 'IMAGECAS_RAW').

    Raises KeyError if unknown.  Returned Path is absolute.
    """
    if key not in _RESOLVED:
        raise KeyError(
            f"Unknown path key {key!r}. Known keys: {sorted(_RESOLVED.keys())}"
        )
    return _RESOLVED[key].resolve()


def all_paths() -> dict[str, Path]:
    """Return a copy of the resolved path table — useful for debug printing."""
    return {k: v.resolve() for k, v in _RESOLVED.items()}


if __name__ == "__main__":
    # Quick CLI: `python -m code.data.paths` prints resolved table.
    for k, v in all_paths().items():
        marker = "✓" if v.exists() else "✗"
        print(f"  {marker}  {k:<22s} {v}")
