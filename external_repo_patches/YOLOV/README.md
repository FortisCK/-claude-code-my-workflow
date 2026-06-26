# YOLOV patches for CATHACTION Task 2

The official YOLOV repo is **not** committed to this repository (`external_repos/` is
gitignored — vendored third-party code with its own license/git history). This directory
version-controls **our modifications** so the Task 2 YOLOV base detector remains reproducible.

## Upstream pin

| Field | Value |
| --- | --- |
| Repo | https://github.com/YuHengsss/YOLOV.git |
| Commit | `fe777cbeeff92d3340d0424a83a6a0e906b97f17` (`fe777cb`) |
| License | Apache-2.0 |

## What this directory contains

- `cathaction_yolov.patch` — our edits to 4 upstream files (97 insertions / 13 deletions):
  - `yolox/models/yolo_pafpn.py` — adds the `YOLOPAFPN_P2` class (stride-4 P2 level; finest-first outputs at strides 4/8/16/32). **Implemented but the P2 exp was not wired in** — kept for future use.
  - `yolox/data/datasets/vid.py` — two-class CATHACTION VID data handling.
  - `yolox/evaluators/coco_evaluator.py`, `yolox/evaluators/vid_evaluator_v2.py` — eval-path adjustments for the CATHACTION two-class setting.
- `exps/cathaction/` — our experiment configs:
  - `cathaction_yolov_s_two_class_mv.py` — **the config that produced the best result** (mv split, 576px). Base detector for the 0.278 per-case mAP50 panel result (YOLOV + Stage2U reranker, Arm A).
  - `cathaction_yolov_s_two_class.py` — single 32-video two-class variant.
  - `cathaction_yolov_s_two_class_mv_768.py` — 768px resolution probe (negative result; see `quality_reports/decisions/2026-06-18_task2_yolov_resolution_negative.md`).
  - `cathaction_yolov_s_agnostic.py` — class-agnostic variant.

## How to reconstruct

```bash
# 1. Clone the pinned upstream into external_repos/ (gitignored)
git clone https://github.com/YuHengsss/YOLOV.git external_repos/YOLOV
cd external_repos/YOLOV
git checkout fe777cbeeff92d3340d0424a83a6a0e906b97f17

# 2. Apply our edits to the upstream files
git apply ../../external_repo_patches/YOLOV/cathaction_yolov.patch

# 3. Drop our experiment configs in place
mkdir -p exps/cathaction
cp ../../external_repo_patches/YOLOV/exps/cathaction/*.py exps/cathaction/

# 4. Run from the repo root with PYTHONPATH set
cd ../..
PYTHONPATH=external_repos/YOLOV python external_repos/YOLOV/tools/train.py \
  -f external_repos/YOLOV/exps/cathaction/cathaction_yolov_s_two_class_mv.py ...
```

## Provenance

Best Task 2 result (validated 6-video phantom panel): **YOLOV-S two-class (mv split, 576px) +
Stage2U reranker (Arm A) = 0.278 per-case mAP50** (raw YOLOV = 0.252). See
`quality_reports/decisions/2026-06-18_task2_yolov_reranker_swap_armA.md` and
`quality_reports/decisions/2026-06-17_task2_eff_gap_calibration_result.md`.
