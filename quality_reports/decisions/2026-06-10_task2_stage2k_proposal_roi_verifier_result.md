# Task 2 Stage2K Proposal + ROI Verifier Pilot Result

Date: 2026-06-10

## Objective

Test whether a two-stage design can improve Task 2 over direct YOLO:

1. Stage 1 proposes multiple candidate boxes.
2. Stage 2 scores candidate crops with an ROI verifier.

## Implemented Artifacts

- Plan:
  - `quality_reports/plans/2026-06-10_task2-stage2k-proposal-roi-verifier.md`
- Script:
  - `scripts/task2/train_roi_verifier.py`
- Smoke run:
  - `outputs/task2/roi_verifier/smoke_stage2k_bg_verifier`
- Main pilot:
  - `outputs/task2/roi_verifier/convnext_tiny_stage2h_proposals_v1_val_v2_spdc1000_e8`

## Stage2K-A: Existing GT-only ROI Classifier on Top-K Proposals

Run:

- proposal detector: Stage2H best detector
- ROI classifier: existing GT-ROI ConvNeXt binary classifier
- candidate mode: top-10
- animal validation: held-out `video_2_animal`
- phantom validation: balanced-small phantom panel

Results:

| Panel | two-class mAP50 | argmax mAP50 | loc recall@0.50 | Notes |
| --- | ---: | ---: | ---: | --- |
| animal `video_2` | 0.4896 | 0.0000 | 0.8598 | candidates often contain the collision target, but binary GT-only classifier cannot select a final class robustly |
| phantom balanced-small | 0.0486 | 0.0331 | 0.3786 | proposal quality and candidate ranking remain weak |

Interpretation:

The existing GT-only ROI classifier is not enough for two-stage detection. It
was trained only on true ROIs, so it lacks a background/rejection class and
cannot reliably choose the correct candidate from a Top-K proposal set.

## Stage2K-B: Background-Aware ROI Verifier

New verifier labels:

- `0`: background proposal
- `1`: normal
- `2`: collision

Candidate construction:

- GT boxes are added as positives for training.
- proposal boxes with IoU >= 0.50 are positive normal/collision examples.
- proposal boxes with IoU <= 0.20 are background examples.
- ambiguous proposal boxes are ignored for training.

Training setup:

- proposal detector: Stage2H best detector
- train split: `train_v1_val_v2_train_labels`
- train sample cap: 1000 per domain/class
- training samples used: 2,183
- verifier candidates: 13,704
- model: ConvNeXt-tiny, 3 classes
- epochs: 8

Candidate summary:

| Split | Candidates | Background | Normal | Collision | Ignore | loc recall@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 13,704 | 6,210 | 3,642 | 3,852 | 0 | not measured as validation |
| animal `video_2` | 714 | 149 | 0 | 196 | 369 | 0.8598 |
| phantom balanced-small | 8,029 | 3,813 | 579 | 203 | 3,434 | 0.3786 |

Best epoch:

- epoch: 8
- primary metric: `valid_animal/detection_argmax_mAP50`

Results:

| Panel | argmax mAP50 | argmax mAP50-95 | two-class mAP50 | two-class mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| animal `video_2` | 0.4892 | 0.1468 | 0.4892 | 0.1468 |
| phantom balanced-small | 0.0408 | 0.0138 | 0.0475 | 0.0155 |

Animal candidate-classification details at best epoch:

- candidate accuracy: 0.5681
- background recall: 0.0
- normal recall: not available because there were no valid normal positive candidates
- collision recall: 1.0
- predicted labels: background 0, normal 1, collision 344

## Interpretation

The two-stage idea is partially validated but not solved.

What worked:

- Stage 1 candidate localization on held-out animal collision is good enough:
  recall@0.50 is 0.8598 overall and 1.0 for collision samples in the earlier
  Top-K diagnostic.
- A background-aware verifier can be trained end to end and produces valid
  detection mAP outputs.

What failed:

- Animal normal remains unrecovered because Stage2H proposals do not produce
  useful normal candidates on `video_2_animal`.
- The verifier does not learn strong background rejection on animal candidates;
  background recall is 0.0.
- Phantom collision remains collapsed because the Stage2H proposal detector is
  animal-adapted and weak on phantom collision.

## Decision

Do not treat the current Stage2K verifier as a final solution.

The useful next direction is to improve the proposal stage before investing
more into the second-stage classifier:

1. train a proposal detector on a split that includes animal normal examples
   (`video_0_animal`) plus animal collision examples (`video_1_animal`), while
   validating on `video_2_animal`;
2. optimize proposal recall separately for normal and collision;
3. only then retrain the background-aware ROI verifier.

The current result supports the diagnosis: the bottleneck is still proposal
coverage and candidate distribution, especially for animal normal and phantom
collision.
