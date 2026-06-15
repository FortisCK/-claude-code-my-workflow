# Task 1 Stage 9D Binary Hard-Negative ROI Refiner

Date: 2026-06-03

Status: stopped after full evaluation; negative result

## Objective

Beat the MSLNet CathAction binary-foreground diagnostic by adding a selective
coarse-to-fine foreground refiner on top of the Stage 9A seven-model champion.

The refiner should solve the specific gap observed in Stage 9C: phantom
false-positive line structures. It should not replace the current multiclass
ensemble.

## Why This Route

Stage 9C showed that global postprocessing cannot cross the MSLNet F1 r3 gap.
Hard erosion and probability thresholding both improved precision but lost
too much recall. Toolness gating also failed on phantom.

So the next model must make local decisions:

- keep real tool pixels near the coarse foreground;
- reject detached or parallel line-like false positives;
- preserve Stage 9A's label_1 / label_2 assignment for surviving foreground.

## Implementation Steps

1. Add binary refiner loss support.
   - `BCEWithLogits + soft Dice`
   - configurable `pos_weight`
   - monitor binary Dice on the balanced eval set first

2. Add hard-negative patch sampling.
   - Read a coarse prediction manifest when provided.
   - Sample some patches centered on `pred_foreground AND NOT dilated(gt_foreground)`.
   - Keep ordinary foreground-centered and random patches as controls.

3. Train a binary ConvNeXt/FPN patch refiner.
   - Label mode: `binary_foreground`
   - Input crop: 512 or 384 in original image space
   - Domain-balanced sampler remains enabled
   - Phantom false positives get explicit exposure through hard-negative patches

4. Gate Stage 9A predictions.
   - Generate coarse predictions on the train split for hard-negative mining.
   - At inference, run binary patch refiner over Stage 9A foreground ROIs.
   - Remove Stage 9A foreground pixels where the binary refiner is below a gate
     threshold.
   - Assign remaining foreground back to label_1 / label_2 from Stage 9A.

5. Verify.
   - Smoke-train on small samples.
   - Evaluate a mixed subset with MSLNet-style metrics.
   - Promote to full released-eval only if subset precision improves without
     collapsing recall.

## Promotion Criteria

Primary MSLNet-style target:

- F1 r3 > `0.9305`, or phantom F1 r3 > `0.916`;
- Dice > `0.6251`;
- IoU > `0.4658`.

Task 1 challenge target:

- official multiclass mean Dice should not drop below the current Stage 9A
  postprocessed champion by more than `0.003`.

## Stop Rule

Stop this route if the refiner behaves like another global shrink/gate:

- precision improves but recall loss keeps F1 r3 below `0.920`;
- phantom F1 r3 stays below `0.905` after a trained refiner;
- official multiclass mean Dice falls materially below Stage 9A.

## Outcome

The stop rule was triggered on 2026-06-04.

Stage9D binary gate result on released eval:

- Official multiclass Dice: `0.645414`, below Stage9A champion `0.655196`.
- MSLNet-style F1 r3: `0.916587`, below the MSLNet paper reference `0.9305`.
- Precision improved versus Stage9A, but recall dropped enough to hurt both
  official Dice and distance-tolerant F1.

Decision note: `quality_reports/decisions/2026-06-04_task1_stage9d_binary_refiner_result.md`.
