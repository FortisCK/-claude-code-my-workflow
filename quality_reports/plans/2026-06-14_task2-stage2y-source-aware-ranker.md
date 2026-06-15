# Task2 Stage2Y Source-Aware Ranker Plan

Date: 2026-06-14

## Motivation

Stage2X proved that better class1/collision candidates exist in the expanded five-source pool, but the current image-only Stage2U ranker cannot assign useful scores to the new candidate source. The failure mode is ranking/calibration, not just proposal recall.

Current evidence:

- five-source union valid_combined class1 R@0.75: `0.1766`;
- Stage2X-only source-filter valid_combined mAP50-95: `0.0046`;
- YOLO+Stage2X source-filter valid_combined mAP50-95: `0.0387`;
- current Stage2V champion valid_combined mAP50-95: `0.04924`.

## Objective

Test whether a source-aware residual metadata head can calibrate candidate scores across proposal sources without destroying the existing ROI image ranker.

## Design

1. Keep the existing ROI image model as the main scorer.
2. Add a zero-initialized residual metadata head:
   - output = `image_model(roi) + metadata_head(source_features)`;
   - zero initialization means the model starts exactly as the warm-started image-only checkpoint.
3. Use inference-safe metadata only:
   - source name one-hot;
   - source confidence;
   - source rank and source priority transforms;
   - normalized candidate box center, size, area, and aspect.
4. Do not use GT class or domain as features, because those may not be available or reliable on hidden test.

## Pilot

Run a bounded pilot:

- train CSV: `outputs/task2/stage2o_candidate_pool/stage2x_train_subset_yolo_geometry_stage2x_top50/valid_combined_candidates.csv`;
- valid CSVs: five-source valid pool;
- init checkpoint: `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt`;
- epochs: `2`;
- train limit: `60000`;
- valid limit: `256`;
- max candidates per sample: `0`;
- batch size: `128`;
- workers: `4`;
- log interval: `100`;
- primary metric: `valid_combined/blend_rank_decay_roi/mAP50-95`.

## Gate

Continue only if the pilot clearly beats the old five-source eval and is plausibly close to Stage2V:

- hard stop if full/sampled valid_combined mAP50-95 remains below `0.041`;
- continue to full-valid eval if sampled valid_combined mAP50-95 is at least `0.045`;
- promote only if full valid_combined mAP50-95 exceeds Stage2V `0.04924`.

If it fails, keep Stage2V as champion and move from model exploration to Task2 submission hardening/reporting.
