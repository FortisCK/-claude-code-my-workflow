# CATHACTION Pre-Validation Roadmap (2026-06-15 → 2026-07-10)

**Status:** Tier A COMPLETE & verified 2026-06-15 (all 9 items). Tier B/C pending.

**Tier A completion (2026-06-15):**
1. Video-level split guard + assert — `prepare_yolo_splits.py`; all derived splits regenerated & audited PASS.
2. Disjointness auditor — `scripts/task2/audit_split_disjointness.py`; tests `test_task2_split_disjointness.py`.
3. Per-case AP evaluator — `compute_detection_map_per_case` in `detection.py`; divergence test `test_task2_per_case_map.py`.
4. Metric-interpretation decision + tie-break — `decisions/2026-06-15_task2_metric_interpretation.md`.
5. Task 1 no-GT entrypoint + preflight + weight manifest — `run_task1_submission_inference.py`, `check_task1_submission_package.py`, `write_task1_weight_manifest.py`; dataset no-GT path + test. Full ensemble inference needs the GPU env (smp absent from CPU env) — see #7.
6. Symmetric candidate freeze — `freeze_task2_candidate.py`; V/AQ/AI frozen with schema (no-leak) validation.
7. Task 2 deps env + license manifest — `environment-task2-gpu.yml`, `decisions/2026-06-15_external_asset_license_manifest.md`.
8. Dockerfile + .dockerignore + entrypoint — BUILD UNVERIFIED (docker not installed; setup gap).
9. Task 1 method-report draft — `reports/2026-06-15_task1_method_report_draft.md`.

Test status: 152 passed, 1 skipped, 1 PRE-EXISTING failure (`test_task1_mslnet_style_metrics` — f1_r2 KeyError in `evaluate_mslnet_style.py`, dated 2026-06-03, NOT touched by Tier A; follow-up).
**Author:** Claude (Opus 4.8), multi-agent sweep (41 agents, 8 survey dimensions, 32 candidate actions, 25 kept)
**Scope:** Everything worth doing before the official CATHACTION validation set + evaluator release (2026-07-10). Submission deadline 2026-08-23.

---

## Independently verified headline finding (CONFIRMED leak)

`scripts/task2/prepare_yolo_splits.py` builds `train_clean` by removing only exact frame label-names that appear in validation (`label_name not in valid_label_set`) — **frame-level dedup, not video-level**.

Direct evidence (verified 2026-06-15):
- `valid_phantom.txt` contains exactly **1 video: video_0**.
- `train_phantom.txt` contains **33 videos including video_0**.
- `train ∩ valid_phantom` video set = **{video_0}** → same-video frames in both train and eval.
- `valid_animal.txt` = 3 videos {video_0, video_1, video_2}; the combined headline is carried by ~107 animal frames.

**Consequence (INV-2 violation):** the Task 2 phantom/combined mAP numbers (Stage2AQ mAP50 0.2366, phantom 0.1298, etc.) and the entire Stage2AB/AQ domain-policy calibration were tuned and reported on a leaky, single-video phantom panel. Treat all Task 2 self-built rankings as **directional, not decisional**, until a video-level disjoint split lands and the phantom panel is rebuilt from multiple videos.

(Task 1's `released_eval` is leak-*assumed*, not leak-*verified*, and is frame-level not case-level — softer caveat, disclose in report, does not block work.)

---

## Situational read

- **Task 1 (DSC primary):** strong, frozen, near-finished. Stage9A 7-model ensemble mean Dice **0.6552** (> paper SegViT 63.47), MSLNet parity on strict Dice/IoU/AHD. But **zero submission packaging** (no entrypoint/preflight/manifest/Dockerfile) and **never evaluated on the human domain**.
- **Task 2 (mAP primary):** active, weaker, riskier. Best self-built Stage2AQ mAP50 0.2366 — but on the leaky split above; metric definition genuinely ambiguous (AP50-like vs COCO mAP50-95).
- **Environment:** GPU = RTX 6000 Ada (49 GB, ample). `docker` NOT installed (in-container smoke blocked → `apt install docker.io`). No Dockerfile for either task. Task 1 champion depends on **ConvNeXtV2 (CC-BY-NC)** backbone — possible submission license blocker. `ultralytics` dep unpinned.

## The ONE thing first

Video-level disjoint splits for Task 2, before any GPU work. Fix `prepare_yolo_splits.py` L39 (basename → video_id set subtraction), assert `train ∩ valid` video_ids = ∅, regenerate ALL derived splits (2j/2l/agnostic/collision_only/adapt_animal), rebuild a multi-video phantom panel.

## Tier A — Zero/low-GPU, do-now

| Action | Payoff | Effort | First step |
|---|---|---|---|
| Video-level disjoint split guard (Task 2) | High (closes −100 leak) | Low | `prepare_yolo_splits.py` L39 video_id subtraction + assert + regression test on video_0 frames 64/66/176/189/237; regenerate all derived splits |
| Disjointness audit script + per-domain headline | Medium | Low | `scripts/task2/audit_split_disjointness.py` (non-zero exit on overlap); quarantine unsafe `valid_animal`; restate Stage2AQ with explicit denominators |
| Faithful local evaluator: official-worded per-case AP | High (de-risks metric) | Medium | Add `compute_detection_map_per_case` sibling in `src/cathaction/metrics/detection.py`; current one is COCO global-pool, no per-case averaging; PDF p.35 says mAP = mean over cases |
| Metric-interpretation decision record + tie-break | High | Low | Quote PDF p.34/35; frame as "ambiguous on IoU axis"; AQ operational default, AI COCO hedge; binding revisit 2026-07-10 |
| Task 1 submission entrypoint + preflight + manifest | High (biggest packaging hole) | Medium | `run_task1_submission_inference.py` with `sys.executable`; full 7-model ensemble + remove_small_min32; no-mask inference path; GPU smoke → preflight failure_count=0 |
| Freeze + verify all 3 Task 2 champions (AQ/AI/V) symmetrically | Medium | Medium | Parameterized verify script (not a 3rd copy-paste); tie-break decision tree; wire into `check_task2_submission_package.py` |
| Pin Task 2 deps; provenance/license manifest | High | Low | `environment-task2-gpu.yml` (+ add `ultralytics`); external-asset manifest flagging ConvNeXtV2 CC-BY-NC / SegFormer research-only / 3 no-license repos as SUBMISSION-BLOCKING |
| Author Dockerfile + .dockerignore (text only) | High (central PDF req) | Medium | CUDA PyTorch base; un-ignore the 3 weight paths in `.dockerignore`; build BLOCKED (docker not installed) → request `apt install docker.io` |
| Draft Task 1 method-report section | Medium | Low | From frozen artifacts; label every number "released-eval, frame-level, human excluded, PROVISIONAL" |

## Tier B — GPU experiments now (only after split guard lands)

| Action | Expected payoff | Effort/risk | First step |
|---|---|---|---|
| Multi-video phantom validation panel (re-eval, no retrain) | High (foundation for all selection) | Med/low | Hold out 3–6 phantom videos with non-trivial class-1 count; re-run champion; report mAP ± per video |
| Source-aware class-balanced ranker on FULL valid | High (~2.1× mAP50 / ~4× mAP50-95 recoverable by ranking) | High/med | Stage2U-style + source-id + per-class calibration; evaluate ONLY on full 931-sample valid (128-subset 0.595→0.154 collapse is the #1 trap) |
| Box-refinement / regression head (mAP50→mAP50-95) | High (localization is dominant AP sink; mAP50 = 4.1× mAP50-95) | High/med | ROI box-delta refiner on train GT; REPLACE selected box (not flood); verify R@0.75 lift |
| Cap predictions per frame/class + score-floor trim | Medium (kills ~20k FP/class) | Low/low | Sweep `topk_per_sample_class` + score floor; pick from a plateau, not a peak; pure postproc |
| Diagnose dead animal-class0 (0 TP / 816 preds / 15 GT) | Medium | Med/low | Check source emission + class-confusion + GT convention; root-cause is the deliverable |
| Task 1 human_holdout diagnostic (read-only) | High (unmeasured domain shift) | Low/low | `predict_tta_ensemble_original_space.py` on human manifest; MSLNet-style binary evaluator; keep human OUT of training |
| Public guidewire/fluoroscopy data — Phase A only | High (only unused legit lever) | High/med | Download + license-check (MSLNet guidewire, Wen 2024 sim-to-real; skip Carvana); log URL/license/sha256; defer fine-tune + all metric claims until after Jul 10 |

**De-prioritized GPU work:** animal high-IoU localizer (n=107), YOLOV ImageNet-VID re-init (ranking not recall is binding), larger Task1 backbones, thin-line postproc (can't reach MSLNet 0.9305 by construction).

## Tier C — Submission insurance

- Docker build + in-container CPU smoke + preflight (both tasks) — BLOCKED on docker install; ship authored Dockerfile + hadolint + build-context tar-inventory meanwhile.
- Domain fallback locked: make **Stage2V (domain-agnostic) the primary default**, not Stage2AB (AB edge is calibration on a 3-video animal split = INV-3 risk). Drop auto-domain-classifier. Synthetic `domain=human/unknown` end-to-end test.
- Dual-metric candidate freeze: keep BOTH Stage2AQ (AP50) and Stage2AI (COCO 50-95) warm + packaged. Rank-stability matrix; carry both to Jul 10 unless one is top-2 under all readings.
- Schema adapter shells (`format_task2_official_submission.py` + Task 1) — config-driven, ONE PROVISIONAL profile, golden-file test, `xfail` schema test so the gap is visible.
- Method-report drafts both tasks, 4-page MICCAI, external-backbone disclosure.

## DO NOT do until official validation (Jul 10)

- Do not commit to a single Task 2 primary champion — freeze all three; the evaluator decides.
- Do not hard-tune Stage2AQ top-k/conf to chase mAP50-95 on the leaky single-video panel — plateau-picks only.
- Do not build a self-derived per-frame/event GT loader — sources confirm box-IoU object-detection AP; implement per-case averaging toggle only.
- Do not retrain the Task 2 champion with video deletion, nor fold human/external data into a CathAction fine-tune comparison.
- Do not promote any postproc/refinement variant over frozen Stage9A / Stage2V/AQ/AI on self-built numbers alone.
- Do not hardcode a guessed official result schema.
- Do not swap the ConvNeXtV2 backbone yet — audit + log license risk now; swap is a separate gated follow-up.

## Recommended 3-week sequence

**Jun 15 — integrity + packaging foundation (CPU/light GPU):** split guard + regression test + regenerate derived splits; audit script + quarantine unsafe animal panel; restate Stage2AQ with honest denominators; pin ultralytics/Task2 env; external-asset license manifest; author Dockerfile + .dockerignore + request docker.io; Task 1 entrypoint + preflight + manifest; metric-interpretation decision record.

**Jun 22 — GPU on the clean split:** multi-video phantom panel (re-eval); FP-cap/score-floor sweep; source-aware class-balanced ranker on FULL valid; Task 1 human_holdout diagnostic; animal-class0 root-cause; Phase-A public-data download + license; freeze + verify 3 Task 2 champions.

**Jun 29 → Jul 10 — refinement + insurance + evaluator-readiness:** box-refinement head (gate on R@0.75); rank-stability matrix; per-case-AP evaluator variant + divergence unit test (late, so fresh); schema-adapter shells + golden/xfail tests; lock Stage2V default + synthetic human test; method-report drafts; in-container smoke if docker installed.

**Land Jul 10 with:** a leak-free split, three frozen+verified Task 2 candidates, a packaged Task 1 champion, a buildable Docker image, faithful local evaluators for both metric readings, and a binding tie-break rule — so the first 48 h after release is pure re-ranking + schema-fill, not design under deadline.

## Key file paths

- Split-prep to fix: `scripts/task2/prepare_yolo_splits.py` (L39, basename→video_id)
- Evaluator to extend: `src/cathaction/metrics/detection.py` (`compute_detection_map`, global-pooled)
- Task 2 entrypoint: `scripts/task2/run_task2_submission_inference.py`; preflight `scripts/task2/check_task2_submission_package.py`
- Task 1 champion provenance: `quality_reports/decisions/2026-06-04_task1_final_wrapup.md`
- Stage2V manifest: `outputs/task2/stage2u_quality_ranker/stage2v_class0_yolo_iou75_class1_yolo_roi_export/champion_manifest.json`
