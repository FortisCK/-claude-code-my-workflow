# Task2 Stage2X Class1 Center Precision Plan

Date: 2026-06-13

## Motivation

Stage2W sequence-tip training improved overall proposal recall, especially after dense templates, but failed the key class1/collision high-IoU gate:

- dense Stage2W standalone valid_combined class1 top50 R@0.75: `0.0040`
- dense four-source union valid_combined class1 R@0.75: `0.1250`
- current augmented candidate-pool class1 R@0.75 reference: about `0.1310`

The dense template sweep showed that box-size/aspect templates can lift class1 IoU@0.50, but not IoU@0.75. This points to center precision as the bottleneck.

## Objective

Train a class1-only sequence localizer optimized for tighter collision center localization, then evaluate whether its proposals improve class1 recall@0.75.

## Design

1. **Split**
   - Build a class1-only training label list from the existing Stage2L train split.
   - Keep validation splits unchanged so metrics remain comparable.

2. **Model**
   - Reuse `scripts/task2/train_sequence_tip_localizer.py`.
   - Use `smp_fpn` with `tu-convnext_tiny`.
   - Keep 5-frame input at `384`.

3. **Loss**
   - Use CenterNet-style focal heatmap loss instead of weighted BCE.
   - Increase coordinate loss weight to directly target center precision.

4. **Primary metric**
   - Select by `valid_combined/class1_center_recall_10px`, not all-class `center@20`.
   - Secondary gate: exported proposal class1 R@0.75.

5. **Evaluation**
   - Export dense class1-oriented templates from best checkpoint.
   - Compare standalone and union oracle against:
     - Stage2W dense;
     - YOLO + geometry + old sequence-tip;
     - YOLO + geometry + old sequence-tip + Stage2W dense.

## Initial Run

Run a bounded pilot first:

- train split: generated class1-only Stage2L train split
- model: `smp_fpn`, `tu-convnext_tiny`
- train limit: `20000` if available
- epochs: `8`
- batch size: `16`
- AMP: off / fp32
- heatmap loss: `centernet`
- coord loss: `20`
- size loss: `4`
- offset loss: `1`
- primary metric: `valid_combined/class1_center_recall_10px`

## Go/Stop Gate

Continue only if either:

- valid_combined class1 center@10 improves meaningfully over Stage2W epoch-2 value `0.0992`; or
- dense proposal export improves class1 R@0.75 above `0.1310`.

Otherwise, stop localizer training and pivot to a different class1 candidate formulation.
