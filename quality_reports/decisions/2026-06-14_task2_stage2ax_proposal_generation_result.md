# Task2 Stage2AX Proposal Generation Result

Date: 2026-06-14

## Decision

Add no-GT proposal exporters for the Stage2AQ hidden-ready proposal sources:

`scripts/task2/export_yolo_nogt_proposals.py`

`scripts/task2/export_sequence_tip_nogt_proposals.py`

and a proposal-source planning manifest helper:

`scripts/task2/plan_stage2ax_proposal_generation.py`

## Why

Stage2AW can run the hidden-style pipeline from precomputed proposal CSVs. The
next gap is producing those proposal CSVs from raw official images without GT.

Reviewing the existing proposal scripts showed:

- `evaluate_yolo_proposals.py` is validation/evaluation-oriented and loads
  labeled Task2 samples to write GT diagnostics;
- `evaluate_task1_geometry_proposals.py` is also validation/evaluation-oriented
  and writes GT diagnostics;
- `evaluate_sequence_tip_proposals.py` uses labeled `SequenceTipDataset` fields
  and writes GT diagnostics.

So Stage2AX adds hidden-ready raw-image proposal paths for the two sources used
by the default Stage2AQ replacement policy:

- `yolo_stage2l`;
- `stage2x_class1`.

## Hidden-Ready Sources

`yolo_stage2l` is now hidden-ready through:

```bash
python3 scripts/task2/export_yolo_nogt_proposals.py \
  --weights outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt \
  --image-dir <official_hidden_images> \
  --split hidden \
  --output-dir <proposal_output_root> \
  --name hidden_yolo_stage2l
```

The exporter supports either:

- `--image-dir`; or
- `--image-list-csv` with `image_path` and optional `sample_id`, `video_id`,
  `frame_index`, `domain`.

It writes:

`<proposal_output_root>/hidden_yolo_stage2l/hidden_proposals.csv`

without any `gt_*` fields.

`stage2x_class1` is now hidden-ready through:

```bash
python3 scripts/task2/export_sequence_tip_nogt_proposals.py \
  --checkpoint outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt \
  --image-dir <official_hidden_images> \
  --split hidden \
  --output-dir <proposal_output_root> \
  --name hidden_stage2x_class1
```

It writes:

`<proposal_output_root>/hidden_stage2x_class1/hidden_proposals.csv`

without any `gt_*` fields.

## Stage2AX Manifest

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/plan_stage2ax_proposal_generation.py \
  --image-dir datasets/collision_detection/images \
  --manifest outputs/task2/stage2ax_proposal_generation/public_images_manifest.json \
  --device cuda
```

Output:

`outputs/task2/stage2ax_proposal_generation/public_images_manifest.json`

Preflight:

- YOLO Stage2L weights: found
- Stage2X class1 checkpoint: found
- image directory: found

Hidden-ready sources:

- `yolo_stage2l`
- `stage2x_class1`

Remaining optional no-GT adapters:

- `task1_geometry_rect`
- `sequence_tip`
- `stage2w_dense`

The most important Stage2AQ source policy is now covered. The remaining
adapters are needed only if we want to reproduce the full public-validation
five-source pool on hidden data.

## Stage2X Checkpoint Smoke

Command:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/export_sequence_tip_nogt_proposals.py \
  --image-dir datasets/collision_detection/images \
  --split hidden_smoke \
  --output-dir /tmp/cathaction_stage2x_nogt_smoke \
  --name stage2x \
  --checkpoint outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt \
  --device cpu \
  --workers 0 \
  --batch-size 1 \
  --limit 2 \
  --no-amp
```

Result:

- image count: 2
- proposal rows: 200
- no `gt_*` fields in output CSV

## Verification

Focused tests:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest \
  tests/test_task2_yolo_nogt_proposals.py \
  tests/test_task2_sequence_tip_nogt_proposals.py \
  tests/test_task2_stage2ax_proposal_generation.py
```

Result: 9 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_*.py
```

Result: 105 passed.

## Next

Wire the hidden proposal generation commands into `run_stage2aq_hidden_pipeline.py`
as an optional first phase. The remaining external gap is final challenge
result-file formatting.
