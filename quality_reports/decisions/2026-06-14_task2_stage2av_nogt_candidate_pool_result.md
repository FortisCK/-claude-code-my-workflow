# Task2 Stage2AV No-GT Candidate Pool Result

Date: 2026-06-14

## Decision

Add a hidden-test-safe no-GT candidate-pool normalizer:

`scripts/task2/build_nogt_candidate_pool.py`

The script combines multiple proposal-source CSVs into Stage2U-compatible
candidate CSVs without requiring labels, `gt_*` fields, or oracle IoU.

It writes one file per split:

`{split}_candidates.csv`

with the metadata needed by Stage2AU:

- `sample_id`, `video_id`, `frame_index`, `domain`;
- `image_path`, `image_width`, `image_height`;
- candidate box `x1`, `y1`, `x2`, `y2`;
- `source`, `source_priority`, `source_rank`, `source_conf`;
- source class/subtype diagnostics.

## Why

Stage2AU made ranker inference possible once no-GT candidate rows exist. Stage2AV
creates those rows from proposal sources. Together they close most of the
hidden-test gap between proposal CSVs and Stage2AE/Stage2AQ clean-prediction
export.

## Public-Validation Smoke

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/build_nogt_candidate_pool.py \
  --output-dir outputs/task2/stage2av_nogt_candidate_pool \
  --name public_valid_five_source_top50
```

Output:

`outputs/task2/stage2av_nogt_candidate_pool/public_valid_five_source_top50`

Rows:

| Split | Rows | Domain Counts |
| --- | ---: | --- |
| `valid_combined` | 206,635 | animal 22,166; phantom 184,469 |
| `valid_phantom` | 184,469 | phantom 184,469 |
| `valid_animal` | 22,166 | animal 22,166 |

The first smoke run exposed a real issue: generic phantom sample IDs like
`video_0_...` could not be mapped to `phantom` from the ID alone. The script now
also infers domain from split names and upgrades per-sample metadata when a
later source/split provides a more specific domain than `unknown`.

## Stage2AU End-to-End Smoke

A two-row candidate subset from the Stage2AV output was passed through the
frozen Stage2U checkpoint with the no-GT inference bridge:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/infer_stage2u_quality_ranker.py \
  --candidate-csv valid_phantom=/tmp/cathaction_stage2av_smoke/valid_phantom_candidates.csv \
  --checkpoint outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt \
  --output-dir /tmp/cathaction_stage2av_smoke/stage2u \
  --device cpu \
  --workers 0 \
  --batch-size 2 \
  --no-amp
```

Result:

- candidate rows: 2
- prediction rows: 2
- required score columns present:
  - `rank_decay_roi_score_collision`
  - `prob_iou75_rank_decay_roi_score_collision`

This smoke also identified and fixed a checkpoint backend problem. The frozen
Stage2U checkpoint uses timm-style ConvNeXt keys (`stem.*`, `stages.*`,
`head.*`). `infer_stage2u_quality_ranker.py` now inspects checkpoint keys and
selects the timm backend automatically instead of silently falling back to
torchvision under `--backend auto`.

## Environment Note

The `cathaction-task1` environment currently does not have `timm`. The
`cardiac-diffusion` environment does have `timm` and successfully ran the
checkpoint smoke. Any final Docker/environment file must include `timm` for the
Stage2U frozen ranker checkpoint.

## Verification

Focused tests:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest \
  tests/test_task2_stage2av_nogt_candidate_pool.py \
  tests/test_task2_stage2u_nogt_inference.py
```

Result: 5 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  -m pytest tests/test_task2_*.py
```

Result: 94 passed.

## Next

The remaining submission hardening step is a single hidden-test wrapper that
connects:

1. proposal generation or precomputed proposal CSV directories;
2. `build_nogt_candidate_pool.py`;
3. `infer_stage2u_quality_ranker.py`;
4. `run_stage2_champion_pipeline.py --variant stage2ae --variant stage2aq`;
5. final result-file formatting required by the challenge.
