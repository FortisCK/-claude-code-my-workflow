# Task 2 Stage2J: Domain-Balanced Training and Validation Panels

Date: 2026-06-10

Status: approved by user request ("可以试试")

## Objective

Create reproducible Task 2 split panels that test whether domain-balanced
training improves animal collision detection without destroying phantom-domain
performance.

## Motivation

Stage2H showed that adding `video_1_animal` to training can produce strong
held-out collision AP on `video_2_animal`, but it also hurt phantom validation.
Stage2I showed the reverse fold is unstable. The issue is therefore not just
model architecture: the current training distribution is dominated by phantom
procedures, while animal data are tiny and procedure-specific.

The next experiment should change the data presentation before changing the
detector family again.

## Plan

1. Generate balanced training lists without moving raw files:
   - keep the original `train_clean` split intact;
   - oversample the animal training procedure by repeat-listing frames;
   - optionally undersample phantom at deterministic procedure/frame level for
     fast pilot runs.
2. Generate validation panels:
   - held-out animal procedure for animal-domain generalization;
   - full `valid_phantom` for compatibility with earlier checks;
   - a small class-balanced phantom panel for faster monitoring and to avoid a
     normal-dominated validation readout.
3. Add tests for:
   - no animal train/validation video overlap;
   - no exact sample overlap between train and validation lists;
   - expected animal oversampling;
   - class-balanced phantom validation panel construction.
4. Run a short pilot only after split generation and tests pass.

## Initial Experiment Design

Create two folds:

- `balanced_train_v1_val_v2`: train with `train_clean` plus repeated
  `video_1_animal`, validate on `video_2_animal`.
- `balanced_train_v2_val_v1`: train with `train_clean` plus repeated
  `video_2_animal`, validate on `video_1_animal`.

For both folds, also emit YAML files targeting:

- held-out animal validation;
- full phantom validation;
- balanced-small phantom validation.

## Success Criteria

This route is worth continuing if short-pilot training shows:

- animal held-out mAP50 improves relative to phantom-only YOLO;
- phantom validation does not collapse;
- class-wise metrics do not become a one-class detector.

If animal AP improves but phantom collapses, the next stage should consider
domain-specific inference or a two-stage proposal/classifier design.

If animal AP still does not improve, the next stage should de-emphasize YOLO as
the primary localizer and revisit dense/local ROI proposal strategies.
