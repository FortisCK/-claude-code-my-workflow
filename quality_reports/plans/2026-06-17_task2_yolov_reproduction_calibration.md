# Plan — Reproduce YOLOV on CATHACTION + EFF-Gap Calibration

**Status:** APPROVED, executing. **Date:** 2026-06-17. **Env:** `cardiac-diffusion` (torch 2.11+cu128). **GPU:** RTX 6000 Ada 49GB.

## Why
Champion ~0.13 / RT-DETR 0.103 look below EFF 0.169, but that compares OUR thin single-video eval to the paper's OFFICIAL test set — two rulers. Reproduce YOLOV (paper baseline, 14.11) on OUR split+eval to get a proven anchor AND calibrate the gap.

## Verdict: reproduce YOLOV (not RTMDet)
YOLOV repo already torch-2.11-patched by the stage2n pilot; `weights/yolox_s.pth` on disk (no download). RTMDet needs an mmcv/cu128 build (worse). Ultralytics = AGPL (avoid as host).

## CRITICAL leakage fix
`prepare_yolov_cathaction.py` default `--train-split` is the stage2l set that **contains video_0** (the phantom eval video) → INV-2 violation. Override with `configs/task2/splits/train_clean_labels.txt` (35,079 frames, no video_0, no animal). Keep default valid args (reproduce the 824 phantom + 107 animal GT frames).

## Steps
0. Smoke: import `get_exp(cathaction_yolov_s_agnostic.py)` → num_classes ok; confirm evaluator COCOeval patches intact.
1. Build two-class dataset: `prepare_yolov_cathaction.py --class-mode two_class --train-split configs/task2/splits/train_clean_labels.txt ... --output-dir datasets/collision_detection_yolov_two_class`. VERIFY: no video_0/animal in train, phantom 824 / animal 107, cats {0:normal,1:collision}.
2. Author 2 exp files: `cathaction_yolox_s_base.py` (single-frame, num_classes=2) + `cathaction_yolov_s_two_class.py` (copy agnostic, set num_classes=2, data_dir, output_dir; keep gframe=8, 576px, data_num_workers=4).
3. Base finetune (accuracy lever, backbone NOT frozen so not strictly required): `tools/train.py -f base -c weights/yolox_s.pth -b 16 --fp16`.
4. Temporal train: `tools/vid_train.py -f two_class -c <base best_ckpt> -b 8 --fp16`. Early-stop on overfit.
5. New code: `scripts/task2/export_yolov_clean_predictions.py` — fork decode from `evaluate_yolov_proposals.py` (`scale=min(test/H,test/W); box=out[:,:4]/scale; score=out[:,4]*out[:,5]; cls=out[:,6]`), write the 14-col CLEAN schema to `{split}_predictions.csv` (NO gt_*), for valid_phantom + valid_animal.
6. Eval: `evaluate_candidate_honest_baseline.py --gt-run-dir <V's run>` → phantom per-case mAP50 = **Y**.

## CALIBRATION (headline deliverable)
Eval-transfer ratio from the one method on both rulers (YOLOV):
`EFF_ours = 0.1488 × (Y / 0.1411) = 1.0546·Y`
Cross-over: **champion 0.13 beats EFF ⟺ Y < 0.1233.**
The GT-from-candidates recall optimism inflates Y AND champion 0.13 by the same mechanism → cancels in the ratio (that's why the ratio, not Y, is the deliverable).

Report a BAND over: per-class vs mean; mAP50 vs mAP50-95 (axis can flip the verdict — RT-DETR was ~2× between them). Pin the paper IoU axis from `Cathaction.pdf` first. Phantom-only for r (don't use animal-inflated combined).

## Go/No-Go (what Y means)
- Y < 0.10 → champion beats YOLOV; via r champion ≥ EFF → GO.
- 0.10 ≤ Y < 0.123 → **win condition**: champion ≈ EFF on a common ruler; the EFF gap is an eval-mismatch artifact.
- 0.123 ≤ Y < 0.15 → EFF gap real on our substrate → lever is NWD/P2/high-res.
- Y > 0.15 → eval too optimistic to calibrate → thin-val blocker; wait for multi-video retrain / official evaluator.
- Step 4 ≤ Step 3 base AP → temporal head not helping → debug before quoting anchor.

## Budget / fallback
~8–18 GPU-hr, one overnight. Hard stop 24 GPU-hr → fall back to the already-trained RT-DETR-R18 (0.103) as the proven-floor anchor.
