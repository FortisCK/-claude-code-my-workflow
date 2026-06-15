# Task2 Docker / Entrypoint Checklist

Date: 2026-06-14

## Current Entrypoint

The current Task2 submission-facing entrypoint is:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-dir <input_images> \
  --output-dir <output_dir> \
  --fallback-domain phantom \
  --execute
```

Alternative input mode:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-list-csv <input_images.csv> \
  --output-dir <output_dir> \
  --fallback-domain phantom \
  --execute
```

Current stable internal outputs:

```text
<output_dir>/task2_submission_inference_manifest.json
<output_dir>/task2_predictions_internal.csv
```

`task2_predictions_internal.csv` is not yet the official result file. It is the
internal Stage2AQ prediction CSV that the future official formatter should
consume.

## Current Pipeline

The entrypoint calls:

```text
run_stage2aq_hidden_pipeline.py
  -> export_yolo_nogt_proposals.py
  -> export_sequence_tip_nogt_proposals.py
  -> build_nogt_candidate_pool.py
  -> infer_stage2u_quality_ranker.py
  -> run_stage2_champion_pipeline.py
  -> task2_predictions_internal.csv
```

The active model policy is Stage2AQ:

- YOLO Stage2L proposal source.
- Stage2X class1 proposal source.
- Stage2U quality-ranker inference.
- Stage2AE baseline export plus Stage2AQ phantom-class1 replacement policy.

## Required Weights

| Artifact | Path | sha256 |
| --- | --- | --- |
| YOLO Stage2L | `outputs/task2/yolo_stage2l_proposal/yolo11s_1024_agnostic_train_v0_v1_val_v2_combined_bal_e20/weights/best.pt` | `b21e7b483485155641924df30304c43df21b9dae1bb6b212cd5fdc1d1d39af85` |
| Stage2X class1 | `outputs/task2/sequence_tip_localizer/stage2x_class1_tip384_convnext_centernet_coord20_e8/checkpoints/best.pt` | `c6490dfc991afe5943fdbb6f915421b0f25738a48ae61b786b308fa1e9cf81b4` |
| Stage2U ranker | `outputs/task2/stage2u_quality_ranker/stage2u_warm_stage2o_fulltrain_valid512_e6/checkpoints/best.pt` | `138b382650220972f06e7644f081a5240d8c8ec9b9e70050e655257ccb39d22c` |

Checksum manifest:

```text
quality_reports/decisions/2026-06-14_task2_weight_manifest.json
```

## Package Preflight

Before building/running a container, run:

```bash
python scripts/task2/check_task2_submission_package.py \
  --image-dir <input_images> \
  --output-dir <output_dir> \
  --checksum-manifest quality_reports/decisions/2026-06-14_task2_weight_manifest.json \
  --strict
```

Current local verified command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/check_task2_submission_package.py \
  --image-dir datasets/collision_detection/images \
  --output-dir outputs/task2/stage2bd_checksum_preflight/example_output \
  --checksum-manifest quality_reports/decisions/2026-06-14_task2_weight_manifest.json \
  --manifest quality_reports/decisions/2026-06-14_task2_submission_package_preflight_with_checksums.json \
  --strict
```

Result:

```text
failure_count=0
```

## Smoke Test

Fast CPU smoke:

```bash
python scripts/task2/run_task2_submission_inference.py \
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

Verified result:

- `task2_predictions_internal.csv` exists.
- It has 26 prediction rows for the 2-image smoke input.
- It is byte-identical to the source Stage2AQ CSV.
- Manifest has `stable_internal_result_exists=true`.

GPU/full-run command shape:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-dir <official_input_images> \
  --output-dir <official_output_dir> \
  --split hidden \
  --device cuda \
  --batch-size 256 \
  --workers 8 \
  --fallback-domain phantom \
  --execute
```

The Codex sandbox used for recent smoke tests could not access the NVIDIA
driver, so the verified smoke was CPU-only. This is an execution-context
limitation, not a conclusion that the workstation GPU is unavailable.

## Docker Packaging Checklist

- Include repository code required by `scripts/task2/` and `src/cathaction/`.
- Include or mount the three required weights listed above.
- Include Python dependencies used by the current pipeline:
  - PyTorch with CUDA support;
  - ultralytics;
  - segmentation-models-pytorch;
  - timm;
  - OpenCV / PIL image IO stack;
  - pandas/numpy/scipy/sklearn-style scientific stack as required by existing scripts.
- Set working directory to repository root.
- Make the container command call `scripts/task2/run_task2_submission_inference.py`.
- Map official platform input images to the `--image-dir` or `--image-list-csv` argument.
- Map official platform output directory to `--output-dir`.
- Run package preflight with checksum manifest before final container build.
- Run `--proposal-limit 2` smoke after container build.

## Remaining Official-Submission Blockers

The available 2026 CATHACTION PDF says participants must submit a Docker
container and a predefined result file, but it does not define the concrete
Task2 result-file schema. Missing details:

- CSV vs JSON;
- required filename;
- required columns/keys;
- box coordinate convention;
- class labels/class names;
- confidence field name;
- whether domain/video/frame metadata are required;
- exact input/output mount paths on the official platform.

Once official validation/platform instructions are released, add:

```text
scripts/task2/format_task2_official_submission.py
```

and call it after `task2_predictions_internal.csv` is generated.

