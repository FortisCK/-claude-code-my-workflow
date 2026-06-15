# Task2 Current Status

Date: 2026-06-14

## Current Method

The current Task2 solution is a multi-candidate detection pipeline rather than a single plain YOLO detector.

The stable AB/AD/AE/AI/AN/AQ export wrapper is:

```bash
python3 scripts/task2/run_stage2_champion_pipeline.py \
  --run-dir <stage2u_eval_run_dir> \
  --output-dir <champion_pipeline_dir>
```

It produces six variants:

| Variant | Role |
| --- | --- |
| `stage2ab` | no-calibration fallback |
| `stage2ad` | conservative animal-only box calibration, strength 1.0 |
| `stage2ae` | balanced/default current champion, animal-only box calibration, strength 1.25 |
| `stage2ai` | high-IoU/high-`mAP50-95` candidate, Stage2AE boxes plus calibrated score policy |
| `stage2an` | mild global score calibration backup, Stage2AE boxes plus class/domain score scales |
| `stage2aq` | balanced/default current candidate, Stage2AE plus phantom-class1-only Stage2X replacement |

The latest AB/AD/AE/AI/AN/AQ public-validation export is:

`outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq`

The current best balanced candidate is Stage2AQ, which keeps Stage2AE
predictions everywhere except phantom class1, where it uses a controlled
`yolo_stage2l + stage2x_class1` multi-source replacement:

`outputs/task2/stage2ar_champion_pipeline_with_aq/public_valid_ab_ad_ae_ai_an_aq/stage2aq`

Stage2AQ upstream inputs can be checked with:

```bash
python3 scripts/task2/verify_stage2aq_upstream.py \
  --output-json outputs/task2/stage2as_aq_upstream_verify/public_valid_verify_summary.json
```

Official-style validation orchestration is now available as a dry-run-first
entry point:

```bash
python3 scripts/task2/run_stage2aq_official_pipeline.py \
  --manifest outputs/task2/stage2at_aq_official_orchestration/public_valid_manifest.json
```

The public-validation dry run found zero missing required inputs and records the
candidate-pool, frozen-ranker-eval, champion-export, and upstream-verification
commands in:

`outputs/task2/stage2at_aq_official_orchestration/public_valid_manifest.json`

Hidden/no-GT Stage2U inference now has a dedicated bridge:

```bash
python3 scripts/task2/infer_stage2u_quality_ranker.py \
  --candidate-csv <split>=<candidate_csv_without_gt> \
  --checkpoint outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt \
  --output-dir <stage2u_hidden_inference_dir>
```

This writes `{split}_candidates_used.csv` and
`{split}_eval_prediction_rows.csv` without requiring GT fields in the input.
It is the missing bridge between hidden-test proposal rows and the existing
Stage2AE/Stage2AQ clean-prediction exporter.

Hidden/no-GT candidate-pool normalization is now available:

```bash
python3 scripts/task2/build_nogt_candidate_pool.py \
  --source yolo_stage2l=<proposal_dir> \
  --source stage2x_class1=<proposal_dir> \
  --split <split_name> \
  --output-dir <candidate_pool_root> \
  --name <run_name>
```

The public-validation smoke output is:

`outputs/task2/stage2av_nogt_candidate_pool/public_valid_five_source_top50`

It produced 206,635 `valid_combined` candidates without GT requirements and
preserved domain metadata as animal 22,166 / phantom 184,469.

A hidden-test-style dry-run wrapper now connects the no-GT pieces:

```bash
python3 scripts/task2/run_stage2aq_hidden_pipeline.py \
  --generate-proposals \
  --raw-image-dir <official_hidden_images> \
  --split <split_name> \
  --work-dir <hidden_work_dir> \
  --manifest <manifest_json>
```

The wrapper can either start from precomputed proposal directories or generate
the default hidden-ready proposal directories from raw images first. In
raw-image mode it generates seven commands:

1. YOLO Stage2L no-GT proposal export;
2. Stage2X class1 no-GT proposal export;
3. yolo-only no-GT candidate pool;
4. multi-source no-GT candidate pool;
5. yolo-only Stage2U no-GT inference;
6. multi-source Stage2U no-GT inference;
7. Stage2AE/Stage2AQ clean-prediction export.

The raw-image public-validation dry run is:

`outputs/task2/stage2ay_hidden_raw_image_pipeline/public_images_dryrun_manifest.json`

It found zero missing required inputs using the current public-validation image
directory and records the full raw images -> proposals -> candidates -> ranker
-> predictions command chain. In raw-image mode, the candidate-pool fallback
image directory is the provided `--raw-image-dir`, so the wrapper is not tied to
the local default `datasets/collision_detection/images` path.

Raw-image proposal generation without GT now has dedicated exporters for the
two sources used by the default Stage2AQ replacement policy:

```bash
python3 scripts/task2/export_yolo_nogt_proposals.py \
  --weights outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt \
  --image-dir <official_hidden_images> \
  --split hidden \
  --output-dir <proposal_output_root> \
  --name hidden_yolo_stage2l
```

```bash
python3 scripts/task2/export_sequence_tip_nogt_proposals.py \
  --checkpoint outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt \
  --image-dir <official_hidden_images> \
  --split hidden \
  --output-dir <proposal_output_root> \
  --name hidden_stage2x_class1
```

The proposal generation status manifest is:

`outputs/task2/stage2ax_proposal_generation/public_images_manifest.json`

It marks `yolo_stage2l` and `stage2x_class1` as hidden-ready. The optional
five-source-public-validation extras `task1_geometry_rect`, `sequence_tip`, and
`stage2w_dense` still need no-GT adapters.

## Public-Validation Metrics

| Variant | valid_combined mAP50 | valid_combined mAP50-95 | Interpretation |
| --- | ---: | ---: | --- |
| `stage2ab` | 0.21128164745000094 | 0.050628135044670744 | safest no-calibration fallback |
| `stage2ad` | 0.21128164745000094 | 0.06251332119288261 | improves high-IoU animal boxes without hurting mAP50 |
| `stage2ae` | 0.21128164745000094 | 0.06327005557430047 | default candidate under AP50-like metric risk |
| `stage2ai` | 0.18439741617609534 | 0.08394956471533466 | best if official `mAP` is COCO-style/high-IoU weighted |
| `stage2an` | 0.20037493968130657 | 0.06478975652205522 | tiny high-IoU gain over Stage2AE, but loses mAP50 |
| `stage2aq` | 0.23661313056264216 | 0.07023557986011882 | current best balanced candidate |

Domain split:

| Variant | valid_phantom mAP50 | valid_phantom mAP50-95 | valid_animal mAP50 | valid_animal mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `stage2ab` | 0.10462265654206386 | 0.04301139057740514 | 0.4997355309073923 | 0.050008473801353336 |
| `stage2ad` | 0.10462265654206386 | 0.04301139057740514 | 0.4997355309073923 | 0.1269937435713481 |
| `stage2ae` | 0.10462265654206386 | 0.04301139057740514 | 0.4997355309073923 | 0.1431702793939404 |
| `stage2ai` | 0.09742357618031064 | 0.040369930409670254 | 0.48785015325856906 | 0.23876800871618054 |
| `stage2an` | 0.10462265654206386 | 0.04301139057740514 | 0.4997355309073923 | 0.1431702793939404 |
| `stage2aq` | 0.12983063980792314 | 0.04994025156087294 | 0.4997355309073923 | 0.1431702793939404 |

## Main Interpretation

The current bottleneck is no longer just "YOLO is bad." The evidence is more specific:

- The candidate-generation stage can produce many plausible boxes, but ranking and scoring are unstable across domains and classes.
- Animal boxes benefit strongly from box calibration, especially at stricter IoU thresholds.
- Phantom performance remains weak and dominates the combined score because the public validation set contains many more phantom rows.
- Stage2AP isolated the weakest failure mode: for `valid_phantom` class1, the current Stage2AE yolo-only pool has only 0.2743 oracle IoU50 sample recall and 0.1262 top-score IoU50 sample recall, so both candidate coverage and hard-negative ranking are insufficient.
- Stage2AQ validates the focused fix: adding Stage2X candidates only for phantom class1 raises phantom class1 AP50 from 0.0457 to 0.0961 and valid_combined mAP50 from 0.2113 to 0.2366.
- The official metric wording is still ambiguous: the PDF says primary ranking uses `mAP`, but it does not explicitly state whether this is COCO-style IoU 0.50:0.95. That is why both `stage2aq` and `stage2ai` must be kept.

## Current Selection Rule

Until official validation/evaluation code clarifies the exact mAP convention:

- Use `stage2aq` as the balanced/default candidate after export hardening.
- Use `stage2ai` if official validation rewards COCO-style/high-IoU mAP.
- Keep `stage2ae` as the simpler submission-safe fallback until Stage2AQ hidden-test export is hardened.
- Keep `stage2an` as a conservative high-IoU score-calibration backup.
- Keep `stage2ad` and `stage2ab` as fallbacks for calibration-risk control.

## Next Optimization Direction

The next useful work should not be another generic YOLO run. The highest-value directions are:

1. Create a single official-split orchestration command that generates Stage2X proposal rows, builds the Stage2O multi-source pool, runs Stage2U frozen eval, verifies inputs, and exports Stage2AQ.
2. Keep Stage2AQ in the champion pipeline as the default balanced candidate.
3. Continue phantom class1 hard-negative reranking only after hidden-test Stage2X candidate generation is reproducible end to end.
4. As soon as official validation is released, run the same wrapper and choose the variant by the official metric, not by current assumptions.

Stage2AT completed item 1 for labeled official-style validation splits.
Stage2AU adds the no-GT Stage2U inference bridge needed for hidden-test
candidate rows. Stage2AV adds the no-GT multi-source candidate-pool normalizer.
Stage2AW adds a dry-run-first wrapper that connects precomputed proposal
directories -> Stage2AV candidate pool -> Stage2AU inference -> Stage2AQ export.
Stage2AX adds raw-image no-GT proposal generation for `yolo_stage2l` and
`stage2x_class1`, which covers the default Stage2AQ source policy.
Stage2AY folds those proposal generators into the hidden wrapper, producing a
dry-run-verified raw-image-to-Stage2AQ command chain. The remaining Docker gap
is official result-file formatting; the 2026 challenge PDF states that a
predefined result file will be required but does not yet define the concrete
Task2 schema. Optional no-GT adapters for geometry/other sequence-tip sources
are only needed if we want the full public-validation five-source pool on
hidden data.

Stage2AZ then verified the raw-image hidden wrapper with a real tiny execute
smoke:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  --generate-proposals \
  --proposal-limit 2 \
  --raw-image-dir datasets/collision_detection/images \
  --split hidden_smoke \
  --work-dir outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom \
  --manifest outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom_manifest.json \
  --device cpu \
  --workers 0 \
  --batch-size 2 \
  --fallback-domain phantom \
  --execute
```

This produced the final internal Stage2AQ prediction CSV:

`outputs/task2/stage2az_hidden_execute_smoke/limit2_cpu_phantom/hidden_stage2ae_stage2aq_predictions/stage2aq/hidden_smoke_domain_policy_predictions.csv`

The file has 26 prediction rows for the 2-image smoke input. The smoke also
exposed and fixed two submission-readiness details: proposal exporters now
inherit wrapper `--batch-size` / `--workers`, and the final champion exporter
accepts a propagated `--fallback-domain` for hidden filenames that do not encode
domain metadata.

The current submission-facing Task2 inference entrypoint is:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-dir <official_input_images> \
  --output-dir <official_output_dir> \
  --fallback-domain phantom \
  --execute
```

This entrypoint wraps the raw-image Stage2AQ hidden pipeline and writes:

```text
<official_output_dir>/task2_submission_inference_manifest.json
<official_output_dir>/task2_predictions_internal.csv
```

The second file is a stable internal Stage2AQ prediction CSV, not the official
challenge result schema. Once the official validation package defines the
required result format, add one final formatter step after this internal CSV is
created.

Stage2BB verified this submission-facing entrypoint in execute mode on a
2-image smoke input:

```bash
/home/mingzhang/miniconda3/envs/cardiac-diffusion/bin/python \
  scripts/task2/run_task2_submission_inference.py \
  --image-dir datasets/collision_detection/images \
  --output-dir outputs/task2/stage2bb_submission_entrypoint_execute/limit2_cpu_phantom \
  --split hidden_smoke \
  --device cpu \
  --batch-size 2 \
  --workers 0 \
  --proposal-limit 2 \
  --fallback-domain phantom \
  --execute
```

Result:

- `task2_submission_inference_manifest.json` was written.
- `task2_predictions_internal.csv` was written.
- `task2_predictions_internal.csv` has 26 prediction rows.
- It is byte-identical to the source Stage2AQ CSV.
- The manifest reports `stable_internal_result_exists=true`.

Stage2BC adds a package-level preflight for the submission inference stack:

```bash
python scripts/task2/check_task2_submission_package.py \
  --image-dir <official_input_images> \
  --output-dir <official_output_dir> \
  --strict
```

The current local preflight manifest is:

`quality_reports/decisions/2026-06-14_task2_submission_package_preflight.json`

It checks the submission entrypoint, hidden wrapper, no-GT proposal exporters,
candidate-pool builder, Stage2U no-GT inference, champion exporter, and the
three required trained weights. The latest local run has `failure_count=0`.

Stage2BD adds checksum hardening for the three required trained artifacts.
Checksum manifest:

`quality_reports/decisions/2026-06-14_task2_weight_manifest.json`

Strict preflight with checksum manifest:

```bash
python scripts/task2/check_task2_submission_package.py \
  --image-dir datasets/collision_detection/images \
  --output-dir outputs/task2/stage2bd_checksum_preflight/example_output \
  --checksum-manifest quality_reports/decisions/2026-06-14_task2_weight_manifest.json \
  --manifest quality_reports/decisions/2026-06-14_task2_submission_package_preflight_with_checksums.json \
  --strict
```

Result: `failure_count=0`.

Stage2BE documents the Docker/entrypoint checklist:

`quality_reports/reports/2026-06-14_task2_docker_entrypoint_checklist.md`

Stage2BF documents metric alignment and the next high-IoU improvement route:

`quality_reports/reports/2026-06-14_task2_metric_alignment_high_iou_route.md`

Main Stage2BF conclusion: Stage2AQ looks competitive under AP50-like / loose
localization comparison, but remains below paper YOLOV/EFF under stricter
mAP50-95-style comparison. The next performance work should target high-IoU box
refinement and candidate ranking rather than another generic detector run.

## Verification

The latest code-level regression passed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/task2/export_stage2ab_domain_policy_predictions.py \
  scripts/task2/export_yolo_nogt_proposals.py \
  scripts/task2/export_sequence_tip_nogt_proposals.py \
  scripts/task2/build_nogt_candidate_pool.py \
  scripts/task2/infer_stage2u_quality_ranker.py \
  scripts/task2/plan_stage2ax_proposal_generation.py \
  scripts/task2/run_stage2_champion_pipeline.py \
  scripts/task2/run_stage2aq_official_pipeline.py \
  scripts/task2/run_stage2aq_hidden_pipeline.py \
  scripts/task2/verify_stage2aq_upstream.py \
  scripts/task2/sweep_stage2an_fast_global_score_scale.py \
  scripts/task2/sweep_stage2aq_phantom_class1_multisource.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 105 passed.

Latest Stage2AY regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 106 passed.

Latest Stage2AZ regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 106 passed.

Latest Stage2BA regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 109 passed.

Latest Stage2BC regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 112 passed.

Latest Stage2BD regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 113 passed.
