# Task2 Stage2BD Weight Checksum Manifest Result

Date: 2026-06-14  
Status: implemented and verified

## Purpose

Stage2BD makes the three trained artifacts required by the Task2 submission
inference package explicit and machine-checkable. Stage2BC verified that files
exist and are non-empty; Stage2BD additionally verifies their exact sha256 and
size.

## Implemented

Added:

```text
scripts/task2/write_task2_weight_manifest.py
```

Updated:

```text
scripts/task2/check_task2_submission_package.py
tests/test_task2_submission_package_preflight.py
```

The package preflight now accepts:

```bash
--checksum-manifest <path>
```

When provided, it verifies size and sha256 for the required Task2 weights.

## Weight Manifest

Generated:

```text
quality_reports/decisions/2026-06-14_task2_weight_manifest.json
```

Command:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python \
  scripts/task2/write_task2_weight_manifest.py \
  --output quality_reports/decisions/2026-06-14_task2_weight_manifest.json
```

Weights:

| Artifact | Size | sha256 |
| --- | ---: | --- |
| YOLO Stage2L | 19,255,962 | `b21e7b483485155641924df30304c43df21b9dae1bb6b212cd5fdc1d1d39af85` |
| Stage2X class1 | 358,019,609 | `c6490dfc991afe5943fdbb6f915421b0f25738a48ae61b786b308fa1e9cf81b4` |
| Stage2U quality ranker | 334,124,917 | `138b382650220972f06e7644f081a5240d8c8ec9b9e70050e655257ccb39d22c` |

## Strict Preflight With Checksums

Command:

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

Manifest:

```text
quality_reports/decisions/2026-06-14_task2_submission_package_preflight_with_checksums.json
```

## Verification

Focused checks:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m py_compile \
  scripts/task2/check_task2_submission_package.py \
  scripts/task2/write_task2_weight_manifest.py

/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest \
  tests/test_task2_submission_package_preflight.py
```

Result: 4 passed.

Full Task2 regression:

```bash
/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python -m pytest tests/test_task2_*.py
```

Result: 113 passed.

