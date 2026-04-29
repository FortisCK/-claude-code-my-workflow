# `data/` — Datasets (content gitignored)

The repo carries only this README and `.gitignore`. All actual data lives on
disk under this directory but is excluded from version control.

## Datasets used

### ImageCAS — primary training data

- **What:** 1000 high-resolution coronary CT angiography (CCTA) volumes with
  paired coronary-artery segmentation labels.
- **License:** Apache 2.0.
- **Source:** [Kaggle: ImageCAS-Coronary-Artery-Segmentation](https://www.kaggle.com/datasets) (search "ImageCAS").
- **Local layout (proposed):**
  ```
  data/imagecas/
    raw/                # untouched downloads (DICOM / NIfTI as provided)
    processed/v1/       # resampled, intensity-windowed, paired clean+motion
    splits/             # train / val / test JSON files (seeded splits)
  ```
- **Preprocessing:** TBD — a `code/data/imagecas/preprocess.py` script will
  produce `processed/v1/` from `raw/` deterministically. The version tag
  (`v1`) bumps when the preprocessing changes.

### PAD — motion-simulation pipeline

- **What:** Patient-specific Anatomy Driven cardiac motion-simulation pipeline.
  Produces (clean, motion-corrupted) paired training data.
- **Source:** TT U-Net authors' GitHub repo
  ([Deng et al., TMI 2023](https://github.com/...) — link to be filled when cloned).
- **Local layout:** `data/pad/` — clone of the upstream repo + a project-local
  fork branch with our patches.
- **License:** as upstream (academic-research, generally permissive).
- **Note:** part of PAD is MATLAB. We may need an Octave fallback or a
  Python re-implementation of the shape-fitting step. Documented as a
  technical risk in
  [`../quality_reports/decisions/2026-04-29_research-direction-v2.md`](../quality_reports/decisions/2026-04-29_research-direction-v2.md).

### XCAT phantom — supporting

- **What:** 4D extended cardiac-torso phantom (Segars group, Duke).
- **Used for:** PAD's 4D statistical shape model construction.
- **License:** academic, requires a request to Duke.
- **Status:** need to ask Pascal / Carlos whether LTSI already has access.
- **Local layout:** `data/xcat/` — placeholder until license is in hand.

## What is and isn't gitignored

This directory's `.gitignore` excludes everything except `README.md` and
`.gitignore` itself. **Do not** check in DICOM, NIfTI, NPZ, model checkpoints,
or run outputs from here.

If you want a small reference fixture (e.g., for unit tests), put it under
`code/data/_fixtures/` (committed) — not under `data/`.

## Data paths in code

Code reads paths from `data/.paths.local` (gitignored), with sane defaults
in `code/data/paths.py`. Format:

```ini
# data/.paths.local — example, not committed
IMAGECAS_RAW=/Volumes/external/imagecas/raw
PAD_REPO=/Users/cmz/work/PAD
XCAT_DATA=/Users/cmz/work/xcat-license-pending
```

This keeps the code portable across machines (laptop / lab cluster / Rennes
GPU box) without committing absolute paths.
