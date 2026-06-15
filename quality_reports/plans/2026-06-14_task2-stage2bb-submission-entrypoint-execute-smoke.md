# Task2 Stage2BB Submission Entrypoint Execute Smoke Plan

Date: 2026-06-14

## Goal

Run the Stage2BA submission-facing Task2 entrypoint in execute mode on a tiny
raw-image subset and verify that it writes the stable internal result CSV.

## Constraints

- Use `--proposal-limit 2` to keep the smoke fast.
- Use CPU in the current Codex execution context because the sandbox cannot
  access the NVIDIA driver.
- Do not claim official challenge format support; this smoke verifies the
  internal CSV output only.

## Steps

1. Run `scripts/task2/run_task2_submission_inference.py` with:
   - `--image-dir datasets/collision_detection/images`
   - `--output-dir outputs/task2/stage2bb_submission_entrypoint_execute/limit2_cpu_phantom`
   - `--proposal-limit 2`
   - `--fallback-domain phantom`
   - `--execute`
2. Check that the wrapper manifest and stable internal CSV exist.
3. Check row counts for the final copied result and source Stage2AQ CSV.
4. Record the result.

## Acceptance Criteria

- The command exits with status 0.
- `<output-dir>/task2_submission_inference_manifest.json` exists.
- `<output-dir>/task2_predictions_internal.csv` exists.
- The stable internal CSV has prediction rows.

