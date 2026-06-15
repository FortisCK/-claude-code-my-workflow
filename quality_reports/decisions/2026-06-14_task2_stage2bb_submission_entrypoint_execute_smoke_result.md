# Task2 Stage2BB Submission Entrypoint Execute Smoke Result

Date: 2026-06-14  
Status: passed

## Purpose

Stage2BA added a submission-facing Task2 inference entrypoint. Stage2BB verifies
that this entrypoint can execute a tiny raw-image no-GT pipeline and produce the
stable internal result CSV.

## Command

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

The command completed successfully.

## Output

Submission-facing manifest:

```text
outputs/task2/stage2bb_submission_entrypoint_execute/limit2_cpu_phantom/
  task2_submission_inference_manifest.json
```

Stable internal result CSV:

```text
outputs/task2/stage2bb_submission_entrypoint_execute/limit2_cpu_phantom/
  task2_predictions_internal.csv
```

Source Stage2AQ CSV:

```text
outputs/task2/stage2bb_submission_entrypoint_execute/limit2_cpu_phantom/
  stage2aq_work/hidden_stage2ae_stage2aq_predictions/stage2aq/
  hidden_smoke_domain_policy_predictions.csv
```

## Verification

Row counts including headers:

| File | Rows including header |
| --- | ---: |
| `task2_predictions_internal.csv` | 27 |
| source Stage2AQ CSV | 27 |

The stable internal result CSV and the source Stage2AQ CSV are byte-identical:

```text
cmp_exit=0
```

The final manifest contains:

```text
stable_internal_result_exists=true
```

## Interpretation

This confirms that the submission-facing entrypoint now executes the full
internal Task2 inference chain and exposes a stable output file. The official
challenge result schema remains unavailable in the current 2026 PDF, so this is
not yet a final official submission file. It is the stable internal result that
the future official formatter should consume.

## GPU Note

The smoke used CPU because the current Codex sandbox cannot access the NVIDIA
driver. This is an execution-context limitation, not a change to the project
assumption that the workstation GPU is usable from the appropriate environment.

