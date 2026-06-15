# Task2 Stage2V Plan: Source-Filtered Inference and Candidate Redesign

Date: 2026-06-13
Status: proposed

## Goal

Move Task2 optimization away from adding more unfiltered candidate sources and toward:

1. reproducible source-filtered inference;
2. better high-IoU candidate generation, especially for phantom.

## Current Evidence

Current best fair Stage2U result on the original YOLO+geometry pool:

- valid_combined `blend_rank_decay_roi`: mAP50 `0.1875`, mAP50-95 `0.0414`

Refined-source fine-tune result on the Stage2P augmented pool:

- valid_combined best: mAP50 `0.1501`, mAP50-95 `0.0318`

Source-filter diagnostic from saved Stage2U prediction rows:

- all augmented sources: `0.1493 / 0.0308`
- original sources: `0.1875 / 0.0414`
- YOLO only: `0.1883 / 0.0416`
- refined only: `0.1269 / 0.0341`

Interpretation:

- appending more candidates hurts AP unless ranking quality improves substantially;
- refined boxes add a little localization recall but too much ranking noise;
- source filtering can recover or slightly improve the baseline, but the gain is too small to be the main solution;
- candidate generation quality remains the bottleneck.

## Stage2V Work Items

### 1. Source-filtered official eval path

Add source-filter support to the Stage2U eval path or a wrapper script so the current `yolo_stage2l`-only diagnostic can be reproduced from candidate CSVs and checkpoints, not only from saved prediction rows.

Minimum output:

- full-valid metrics for `yolo_stage2l`-only using the Stage2U champion checkpoint;
- prediction rows usable for downstream submission/export checks;
- comparison against original mixed-source Stage2U.

Success threshold:

- reproduce the diagnostic direction: valid_combined mAP50-95 around `0.0416`;
- no regression below the current mixed-source `0.0414`.

### 2. Candidate source audit by domain and class

Quantify for each source:

- candidate count per sample;
- best-IoU distribution;
- recall at 0.50, 0.75, 0.90;
- contribution to AP true positives vs false positives;
- split by phantom/animal and normal/collision.

Success threshold:

- identify which source helps each domain/class;
- identify candidate generation failures that cannot be solved by ranking.

### 3. Phantom-specific candidate redesign

Prioritize phantom because current best phantom mAP50-95 is only about `0.0345`.

Candidate ideas:

- yolo-only high-confidence proposals plus local perturbation grid around YOLO boxes;
- Task1 mask/centerline-derived interaction proposals, but only if source audit shows a clear target region;
- heatmap/keypoint-style interaction localizer trained on GT box centers;
- temporal smoothing or center-track consistency for stable phantom frames.

Success threshold:

- improve phantom candidate-pool R@0.75 without doubling low-quality candidates;
- target valid_phantom mAP50-95 improvement of at least `+0.005`.

### 4. Stop criteria

Stop Stage2V source filtering if:

- yolo-only official eval does not reproduce the diagnostic improvement;
- improvements remain below `+0.001` mAP50-95 after full-valid evaluation.

Stop candidate-source additions if:

- they increase R@0.75 but lower AP after reasonable filtering;
- they mostly add duplicate low-IoU candidates.

## Proposed Immediate Next Step

Implement source-filtered Stage2U eval first. This is lower risk than a new model and will make the current best inference policy reproducible.

After that, start the candidate source audit and use it to choose one candidate redesign, rather than adding another detector blindly.
