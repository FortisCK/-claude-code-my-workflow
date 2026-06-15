# Decision Record — External-Asset License Manifest (Submission Compliance)

**Date:** 2026-06-15
**Status:** ACTIVE — pre-submission compliance tracking
**Why:** The challenge permits public external data + pretrained models (INV-4), the dataset license is **CC-BY-NC-SA**, and a joint challenge publication is planned. Non-commercial pretrained weights are therefore generally acceptable; the real risks are **AGPL copyleft code** and **repos with no license**. This manifest pins what each asset is, its license, and whether it blocks submission/publication.

## Champion-path assets (what actually ships)

### Task 1 — Stage9A 7-model ensemble (champion, mean Dice 0.6552)

| Backbone (model) | Source | License | Risk | Note |
|---|---|---|---|---|
| `tu-convnext_tiny` (convnext640) | timm ImageNet, ConvNeXt-v1 | MIT code / permissive weights | LOW | |
| `efficientnet-b3` | smp/timm ImageNet | Apache-2.0 | LOW | (no imagenet norm in its config — preprocessing per-model) |
| **`tu-convnextv2_base`** (convnextv2_base, weight 0.0936) | timm ImageNet, **ConvNeXtV2** | **code MIT / pretrained weights CC-BY-NC 4.0** | **MEDIUM** | **In the champion ensemble.** CC-BY-NC is compatible with CC-BY-NC-SA challenge data and academic publication, but **must be disclosed** in the method report and bars commercial reuse. Resolved a survey disagreement: `run_stage9a_seven_model_postprocess.sh` line 58 confirms convnextv2_base IS in the frozen champion (one survey agent wrongly reported v1-only). |
| `tu-convnext_small` ×4 (small512/640/cldice/toolness) | timm ImageNet, ConvNeXt-v1 | MIT / permissive | LOW | bulk of the ensemble |

### Task 2 — Stage2 proposal + ranker pipeline (champion candidates)

| Asset | Source | License | Risk | Note |
|---|---|---|---|---|
| `yolo11s.pt`, finetuned `yolo_stage2l_proposal/.../best.pt` | Ultralytics YOLO11 | **AGPL-3.0** | **HIGH** | **The proposal stage imports `ultralytics.YOLO` (`export_yolo_nogt_proposals.py`) — AGPL exposure is real, not hypothetical.** AGPL copyleft can attach to the derived Docker image/code on distribution. Acceptable for the challenge run itself, but **flag for the joint publication / public code release**: either keep the YOLO component AGPL-compliant (publish that code under AGPL) or replace it with an Apache/MIT detector before any commercial path. |
| `weights/yolox_s.pth` | Megvii YOLOX-S | Apache-2.0 | LOW | matches YOLOV repo |
| stage2x tip localizer (`tu-convnext_tiny`) | timm ConvNeXt-v1 | permissive | LOW | |
| stage2u quality ranker | internal/derived | n/a | LOW | |

## `external_repos/` audit

| Repo | License | Risk |
|---|---|---|
| ConvNeXt | MIT | LOW |
| ConvNeXt-V2 | MIT code + **CC-BY-NC 4.0 weights** | MEDIUM (only if v2 weights ship — they do, via champion) |
| MSLNet | MIT | LOW |
| nnUNet | Apache-2.0 | LOW |
| TransUNet | Apache-2.0 | LOW |
| YOLOV | Apache-2.0 | LOW |
| SegFormer | **NVIDIA non-commercial (research/eval only)** | MEDIUM (not in champion) |
| **FGA-Net-Guidewire-Segmentation** | **NO LICENSE FILE** | **HIGH** |
| **Swin-Unet** | **NO LICENSE FILE** | **HIGH** |
| **WT-CMUNeXt** | **NO LICENSE FILE** | **HIGH** |

## Decisions

1. **SUBMISSION-BLOCKING to resolve before final upload / code release:**
   - **AGPL-3.0 (Ultralytics YOLO11)** in the Task 2 champion path — decide: ship AGPL-compliant, or swap the detector. Track as an open item; do NOT swap during pre-validation (separate gated follow-up).
   - **No-license repos** (`FGA-Net-Guidewire-Segmentation`, `Swin-Unet`, `WT-CMUNeXt`) — confirmed NOT reachable in either champion inference chain. Keep them OUT of the Docker build context (`.dockerignore` excludes `external_repos/`) and out of any released code. Do not import from them.
2. **Disclose in the method report:** ConvNeXtV2 (CC-BY-NC) and any NC weights actually used; the YOLO AGPL component.
3. **Acceptable as-is** (vs CC-BY-NC-SA data, academic use): ConvNeXtV2 CC-BY-NC, SegFormer NVIDIA-NC. Permissive (MIT/Apache): everything else in the champion paths.
4. **Env reproducibility:** `environment-task2-gpu.yml` adds the missing `ultralytics` dep; the active `cathaction-task1` env has a **CPU-only** torch build, so GPU runs require the `-gpu` env.

## References

- `environment-task2-gpu.yml`, `environment-task1-gpu.yml`
- `scripts/task1/run_stage9a_seven_model_postprocess.sh` (champion ensemble definition)
- `scripts/task2/export_yolo_nogt_proposals.py` (ultralytics import)
- `quality_reports/plans/2026-06-15_prevalidation_roadmap.md` — Tier A #7
