# Task 1 Method Report — DRAFT (PROVISIONAL, pre-official-validation)

**Date:** 2026-06-15
**Status:** DRAFT. Every number below is **internal released-eval, frame-level, NOT the hidden test**, and is provisional pending the official validation package (2026-07-10). Target format: MICCAI 4-page methodology report.

## Task

Catheter and guidewire segmentation in X-ray fluoroscopy (CATHACTION Task 1). Primary ranking metric **DSC**; secondary IoU/Jaccard, mIoU, pixel accuracy. Pixel labels distinguish three classes: 0 = background, 1 = catheter, 2 = guidewire.

## Method (champion = Stage9A)

A **7-model softmax ensemble** with horizontal-flip TTA and a light connected-component postprocess.

| # | Arch | Backbone | Input | Norm | Weight |
|---|---|---|---|---|---:|
| 1 | FPN | convnext_tiny (v1) | 640, aspect_pad | imagenet | 0.260 |
| 2 | Unet | efficientnet-b3 | 512, direct | none | 0.260 |
| 3 | FPN | convnextv2_base | 512, direct | imagenet | 0.094 |
| 4 | FPN | convnext_small (v1) | 512, direct | imagenet | 0.094 |
| 5 | FPN | convnext_small (v1) | 640, aspect_pad | imagenet | 0.093 |
| 6 | FPN | convnext_small (v1) + clDice | 640, aspect_pad | imagenet | 0.100 |
| 7 | FPN | convnext_small (v1) + toolness-aux | 640, aspect_pad | imagenet | 0.100 |

All built with `segmentation_models_pytorch` (timm encoders, ImageNet init, raw logits). Per-model preprocessing differs (model 2 uses no ImageNet normalization; resize modes differ), so the inference path runs a per-model loader. Model 7 has a 4th auxiliary "toolness" channel sliced off before softmax. Per-model softmax probabilities are mapped back to original image size, weighted (weights re-normalized to sum 1), summed, and argmax'd. TTA = mean of {identity, hflip}. **Postprocess:** `remove_small_min32` (drop connected components < 32 px per class); no dilation/erosion.

## Results (internal released-eval, frame-level, N=4691, NOT hidden test)

| Metric | Value |
|---|---:|
| Mean Dice | **0.6552** |
| label_1 (catheter) Dice | 0.6340 |
| label_2 (guidewire) Dice | 0.6764 |
| mIoU | 0.5332 |
| Pixel accuracy | 0.9949 |
| Animal-domain Dice | 0.7423 |
| Phantom-domain Dice | 0.6314 |

Raw (pre-postprocess) mean Dice 0.6537 → +0.0015 from `remove_small_min32`.

## Positioning vs prior work (NOT a leaderboard comparison)

- **CathAction dataset paper (arXiv 2408.13126), Table IV:** best baseline SegViT Dice **63.47**, SwinUNet Jaccard 59.54. Our ensemble's mean Dice 0.6552 is **above the paper's SegViT** on our internal released-eval split. Caveat: different split + evaluator from the official hidden test; treat as indicative.
- **MSLNet (Barbu 2025):** a guidewire-localization method reporting parity with Res-UNet/nnU-Net on a 23,449-frame fluoroscopy set (same scale as our segmentation data). Our comparison is single-ensemble-vs-their-reported-means, not a head-to-head; MSLNet's strong binary guidewire Dice (~0.93) is on a binary task and a different evaluation, so it is NOT directly comparable to our 3-class mean Dice.

## Provenance

- Champion definition: `scripts/task1/run_stage9a_seven_model_postprocess.sh`; wrap-up `quality_reports/decisions/2026-06-04_task1_final_wrapup.md`.
- Frozen artifacts: `outputs/task1/stage9a_seven_model_add_both_010_010/remove_small_min32/{predictions.csv,eval.json,summary.json}`.
- 7 checkpoints sha256: `quality_reports/decisions/2026-06-15_task1_weight_manifest.json`.
- Submission entrypoint (no-GT): `scripts/task1/run_task1_submission_inference.py`; preflight `scripts/task1/check_task1_submission_package.py`.

## Disclosures (challenge policy)

- **External pretrained weights:** all encoders ImageNet-pretrained (public at launch, INV-4). The ensemble includes **ConvNeXtV2 (CC-BY-NC)** ImageNet weights — non-commercial, compatible with the CC-BY-NC-SA challenge data and academic use; disclosed per `quality_reports/decisions/2026-06-15_external_asset_license_manifest.md`. All other encoders are MIT/Apache.
- **No private/clinical data, no hidden-test tuning, fully automated inference.**

## Known limitations / open risks (must keep in final report)

1. **Human domain unevaluated.** The released-eval set is animal+phantom only; the champion has never been scored on the human domain (a stated CATHACTION domain). Run the human_holdout diagnostic before claiming domain generality.
2. **Split is frame-level, leak-assumed not leak-verified** for Task 1 (unlike Task 2, which is now video-disjoint). Disclose; do not claim case-level rigor for Task 1 numbers until audited.
3. All numbers are internal proxies; the official hidden-test DSC may differ.
