# Task2 Stage2BA Submission Inference Entrypoint Plan

Date: 2026-06-14

## Goal

Create a single Task2 inference entrypoint that can be used as the future Docker
command once official validation/hidden input paths are known.

Stage2AZ proved that `run_stage2aq_hidden_pipeline.py` can execute from raw
images to an internal Stage2AQ prediction CSV. Stage2BA should wrap that command
behind a cleaner submission-facing interface:

```bash
python scripts/task2/run_task2_submission_inference.py \
  --image-dir <official_input_images> \
  --output-dir <official_output_dir> \
  --execute
```

## Constraints

- Do not invent the final official result-file schema; it is not specified in
  the available 2026 PDF.
- Produce a stable internal prediction CSV and manifest.
- Dry-run by default.
- Preserve support for `--image-list-csv` as an alternative to an image
  directory.
- Keep Stage2AQ as the default balanced candidate.

## Steps

1. Add `scripts/task2/run_task2_submission_inference.py`.
2. Have it build and optionally execute the Stage2AQ hidden raw-image wrapper.
3. After execution, copy the internal Stage2AQ prediction CSV to a stable
   output filename such as `task2_predictions_internal.csv`.
4. Write a submission-facing manifest recording input path, output path,
   fallback domain, wrapper manifest, final internal CSV, and official-format
   caveat.
5. Add dry-run/command-generation tests.
6. Run focused tests and full Task2 regression.

## Acceptance Criteria

- The entrypoint dry-run writes a manifest without running inference.
- The entrypoint command contains `--generate-proposals`, the raw input source,
  output work directory, split, device, batch size, workers, and fallback domain.
- The script exposes a stable internal result CSV path.
- Tests pass.

