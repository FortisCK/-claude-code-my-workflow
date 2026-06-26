# Session Log — 2026-06-15 — Task 2 split integrity + Tier A start

## Context

Resumed a long-running Codex thread ("检查 datasets 目录" → really the Task 2 collision-detection thread). Reviewed both supporting papers (CathAction arXiv + MSLNet) and the authoritative 2026-04-22 challenge PDF (source of truth). Ran a multi-agent pre-validation sweep (41 agents, 8 dimensions) that produced `quality_reports/plans/2026-06-15_prevalidation_roadmap.md`. User approved "开始A" (execute Tier A).

## Key finding (verified, not relayed)

The Task 2 self-built split had a **video-level (case-level) leak (INV-2)**:
- `scripts/task2/prepare_yolo_splits.py` deduped train vs valid by **frame name**, not video id.
- `valid_phantom.txt` = a single video (video_0); `train_phantom.txt` also contained video_0.
- After frame-level dedup, **5 phantom video_0 frames** remained in train while video_0 is 100% of valid_phantom.

Magnitude calibration: the train/valid leak itself is only **5 frames / 35,788 (0.01%)** — it did NOT materially inflate results. The substantive Task 2 methodology weaknesses are separate and remain (Tier B): single-video phantom validation, score-mode selection tuned on the reported eval set (selection-on-eval), and an animal-dominated combined headline (~107 frames).

## Tier A #1 + #2 done (verified)

- `prepare_yolo_splits.py`: frame-level → **video-level** dedup, hard `AssertionError` if `train_clean ∩ valid` videos ≠ ∅, plus a `video_level_disjointness` block in `summary.json`.
- New `scripts/task2/audit_split_disjointness.py`: reusable enforcement tool, non-zero exit on overlap; `--splits-dir` auto-discovery + explicit `--train/--valid` modes.
- New `tests/test_task2_split_disjointness.py` (5 tests) + updated `tests/test_task2_dataset.py` to video-level semantics. **9 tests pass** (env `cathaction-task1`, pytest 9.0.3).
- Regenerated all derived splits from fixed train_clean and audited: base, agnostic, collision_only, stage2j (2 folds), adapt_animal (3 folds), stage2l — **all PASS**. (stage2aa/2o/2x dirs hold export-only train subsets, no held-out lists; they read the now-fixed train_clean.)

Verification commands: `python -m pytest tests/test_task2_split_disjointness.py tests/test_task2_dataset.py` (9 passed); `python scripts/task2/audit_split_disjointness.py --splits-dir configs/task2/splits` (PASS, 32 train ⟂ 4 held-out).

## Open / next (Tier A remaining)

#3 per-case AP evaluator variant; #4 metric-interpretation decision record; #5 Task 1 submission entrypoint+preflight+manifest (priority 88, biggest packaging gap); #6 freeze AQ/AI/V symmetrically; #7 pin Task 2 deps + external-asset license manifest (ConvNeXtV2 CC-BY-NC flag); #8 Dockerfile (build blocked: `docker` not installed); #9 Task 1 method-report draft.

## Tier A COMPLETE (all 9 items, 2026-06-15)

#3 per-case AP evaluator (`compute_detection_map_per_case`) + divergence test. #4 metric-interpretation decision record + tie-break tree (per-case averaging confirmed by official ranking text; IoU axis still ambiguous → keep AQ + AI). #5 Task 1 no-GT submission entrypoint + preflight + weight manifest + dataset no-GT path (hermetic test passes; full ensemble inference needs the GPU env — smp absent from CPU env). #6 parameterized `freeze_task2_candidate.py`; V/AQ/AI frozen with no-leak schema validation (AQ canonical = stage2ae_stage2x_sweep, mAP50 0.2372). #7 `environment-task2-gpu.yml` (+ultralytics) + external-asset license manifest (AGPL YOLO + no-license repos flagged). #8 Dockerfile + .dockerignore + entrypoint (BUILD UNVERIFIED — docker not installed). #9 Task 1 method-report draft.

Final test status: **152 passed, 1 skipped, 1 pre-existing failure** (`test_task1_mslnet_style_metrics` — `evaluate_mslnet_style.py` f1_r2 KeyError, dated 2026-06-03, unrelated to Tier A).

Added `.gitignore` rules for generated split lists / `*.local.yaml` (machine-specific absolute paths, INV-18). Committing Tier A.

## Notes / blockers

- `docker` binary not installed on host → in-container smoke blocked (`apt install docker.io`).
- Base python lacks pytest; use `/home/mingzhang/miniconda3/envs/cathaction-task1/bin/python`.
- Whole repo (scripts, decisions, configs) is still **untracked in git** — recommend committing code+docs (data/outputs ignored) at a Tier A checkpoint.


---
**Context compaction (auto) at 10:47**
Check git log and quality_reports/plans/ for current state.
