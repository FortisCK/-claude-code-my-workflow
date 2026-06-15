# Task2 Stage2AK Bottleneck Attribution Result

Date: 2026-06-14

## Question

After Stage2AE/Stage2AI, should the next Task2 effort focus on:

- candidate generation;
- box calibration/refinement;
- ranking/verifier scoring?

## Diagnostic

Script added:

`scripts/task2/diagnose_stage2ak_bottleneck.py`

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/diagnose_stage2ak_bottleneck.py \
  --run-dir outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fullvalid_yolo_only_eval \
  --prediction-dir outputs/task2/stage2aj_champion_pipeline_with_ai/public_valid_ab_ad_ae_ai/stage2ae \
  --box-transform-json outputs/task2/stage2ae_calibration_strength_sweep/animal_only_strength_1.25_valid_combined.json \
  --output-dir outputs/task2/stage2ak_bottleneck_attribution/stage2ae_public_valid
```

Outputs:

- `outputs/task2/stage2ak_bottleneck_attribution/stage2ae_public_valid/stage2ak_bottleneck_attribution_manifest.json`
- `outputs/task2/stage2ak_bottleneck_attribution/stage2ae_public_valid/valid_phantom_candidate_recall.csv`
- `outputs/task2/stage2ak_bottleneck_attribution/stage2ae_public_valid/valid_phantom_ap_attribution.csv`
- `outputs/task2/stage2ak_bottleneck_attribution/stage2ae_public_valid/valid_phantom_source_recall.csv`

## Main Result

On the same Stage2AE candidate pool, replacing current scores with oracle IoU scores gives a much higher upper bound:

| Split | Current mAP50 | Current mAP50-95 | Oracle-score mAP50 | Oracle-score mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `valid_combined` | 0.21128164745000094 | 0.06327005557430047 | 0.4535740489666544 | 0.2587057365179862 |
| `valid_phantom` | 0.10462265654206386 | 0.04301139057740514 | 0.4003400993223441 | 0.23428344973161846 |
| `valid_animal` | 0.4997355309073923 | 0.1431702793939404 | 0.5 | 0.24801980198019802 |

This shows a large recoverable scoring/ranking gap, especially on phantom.

## Per-Class Evidence

For `valid_phantom`:

| Class | any-candidate recall@0.50 | top1-source recall@0.50 | current AP50 | oracle-score AP50 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7354368932038835 | 0.35436893203883496 | 0.16353168130818602 | 0.5603011295529466 |
| 1 | 0.27427184466019416 | 0.019417475728155338 | 0.045713631775941704 | 0.24037906909174153 |

Interpretation:

- `phantom class0`: candidate coverage is usable, but the current ranker/scoring does not promote the best candidates.
- `phantom class1`: candidate coverage is also weak, but the gap between any-candidate recall and top1-source recall is extreme; ranking/verifier still has immediate recoverable space.
- `animal class1`: AP50 is already near solved; remaining gains are mostly high-IoU geometry.

## Source-Level Check

The current Stage2AE run directory uses only `yolo_stage2l` candidates. A separate check of the prior three-source candidate pool showed:

| Group | yolo-only recall@0.50 | three-source recall@0.50 | yolo-only recall@0.75 | three-source recall@0.75 |
| --- | ---: | ---: | ---: | ---: |
| phantom class0 | 0.7354368932038835 | 0.7669902912621359 | 0.3422330097087379 | 0.4174757281553398 |
| phantom class1 | 0.27427184466019416 | 0.4393203883495146 | 0.14805825242718446 | 0.21601941747572814 |

The extra `stage2x_class1` candidates help phantom class1 localization, but the previous Stage2AA selector failed to turn that into AP. Therefore generic candidate expansion is not enough unless paired with a better verifier/ranker.

## Decision

Prioritize a Stage2AL ranking/verifier experiment.

The next experiment should use the existing `train_stage2u_quality_ranker.py` infrastructure and explicitly target phantom/class-aware quality ranking, rather than launching another generic detector or another box-calibration sweep.

Acceptance target for Stage2AL:

- improve `valid_phantom` class1 AP50 above Stage2AE's 0.045713631775941704;
- avoid reducing `valid_combined` mAP50 below Stage2AE's 0.21128164745000094 unless `mAP50-95` improves enough to justify an alternate Stage2AI-style candidate;
- compare against Stage2AE, Stage2AI, oracle IoU score, and top1 source rank using Stage2AK diagnostics.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile scripts/task2/diagnose_stage2ak_bottleneck.py
```

passed.

