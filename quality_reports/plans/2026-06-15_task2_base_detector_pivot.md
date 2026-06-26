# Plan — Task 2 Base-Detector Pivot to YOLOV (temporal tiny-object)

**Status:** PROPOSED (proof-first; awaiting go-ahead to run the ~1 GPU-day experiment)
**Date:** 2026-06-15 · Deadline 2026-08-23 · Official evaluator 2026-07-10 · GPU env `cardiac-diffusion` (torch 2.11+cu128) on RTX 6000 Ada 49GB

## Verdict

The YOLO11-proposal + ROI-reranker stack is a **dead end at the base** (not downstream). Honest phantom per-case mAP50 ~0.10–0.13 is **below** the paper's tiny-object baselines (YOLOV 0.159 / EFF 0.169) — we lose **inside** the winning axis. The right pivot target is **tiny-object localization, with temporal added because it's cheap** — NOT "pure temporal." Paper Table V: temporal action detectors STEP/YOWO/YOWO-Plus/HIT (9.08/9.92/10.28/10.81) are the **bottom four** and collapse to ~1–3 under domain shift (Table VIII); tiny-object YOLOV (14.11) / EFF (14.88) lead and stay robust. **Target = YOLOV (Shi et al., AAAI 2023): simultaneously tiny-object AND temporal, the literal paper baseline, already vendored (`external_repos/YOLOV`, Apache-2.0), and it retires the AGPL-3.0 YOLO11 publication risk.** The user's "switch to temporal" instinct is right *in the YOLOV sense* (tiny-object backbone + frame aggregation); YOWO/STEP/HIT would chase the losing axis.

**Why Codex's dismissal was wrong:** the stage2n "do not promote YOLOV" verdict is **inconclusive** — it tested YOLOV as a 1k-sample, 3-epoch, class-agnostic PROPOSAL feeder with the **base-finetune skipped** (raw 80-class COCO weights → cls head dropped → epoch-1 AP=0). Pilot AP50 was still RISING (0→0.373→0.401) when stopped. YOLOV-as-end-detector was never actually tested.

## Recommended detector

| | Primary | Backup |
|---|---|---|
| Detector | **YOLOV-S** (YOLOX backbone), two-class | RT-DETR / RTMDet single-frame + small-object head (EFF *idea*, modern code) |
| Temporal / tiny-object | Yes / Yes | No / Yes |
| License | Apache-2.0 | Apache-2.0 |
| Feasible by deadline | Yes (harness half-built) | Maybe (MMCV/MMDet build friction vs torch 2.11) |

Rejected: original EFF repo (dead `maskrcnn_benchmark`, won't build on torch 2.11 — port the *idea* not the code); YOWO/STEP/HIT (losing axis + need contiguous clips we lack); StreamYOLO/TransVOD (large-object, weeks of work); staying on YOLO11 (AGPL-3.0 + weak high-IoU boxes).

## Keep vs discard from Codex's pipeline

Back half is **detector-agnostic** (consumes the 14-col CSV). A new base just emits that CSV.
- **Keep as-is:** metric layer (`detection.py` global + per-case), honest baseline evaluator, video-disjoint Tier A split + auditor + tests, candidate freeze, submission entrypoint/Docker/preflight, 42 detector-agnostic tests.
- **Keep as optional top layer:** Stage2U reranker + class/domain score-mode selection (only if it lifts full-valid per-case mAP on YOLOV boxes).
- **Discard/shelve:** multi-source candidate fusion (Stage2X/Y/Z/AA — raised oracle recall but hurt AP; a stronger base removes their motivation).

## Proof-first experiment (~1 GPU-day, go/no-go)

0. **Prereqs:** download `weights/yolox_s.pth` (Apache-2.0, NOT on disk); generate the **two-class** COCO set via `prepare_yolov_cathaction.py --class-mode two_class` from `configs/task2/splits/train_clean_*.txt` (only the agnostic set exists); re-audit disjointness.
1. **Stage-1:** finetune YOLOX-S **two-class base** from `yolox_s.pth` (the step the pilot skipped). ~1–1.5 GPU-hr.
2. **Stage-2:** train **YOLOV-S two-class** on full 35,079-frame `train_clean`, ~6–10 ep, tsize ≥576. ~1–1.5 GPU-hr.
3. **Exporter:** write a **gt-free** YOLOV→14-col CSV exporter (mirror `export_yolo_nogt_proposals.py`; the existing `evaluate_yolov_proposals.py` emits a gt_* diagnostic schema, unusable for freeze).
4. **Evaluate** via `evaluate_candidate_honest_baseline.py` (per-case phantom mAP50; also report combined/animal). Do NOT use YOLOV's internal COCOeval as the leaderboard number.

**Gate:** GO if per-case **phantom mAP50 > 0.13** (beats champion). NO-GO if ≤0.13 → escalate to YOLOV-L/X or larger tsize once, else fall back to RT-DETR/RTMDet. Watch class1 (collision) recall + animal class0 regardless — a good aggregate hiding class1/animal failure is not a real GO.

## If proof passes — milestones to 2026-08-23

| M | Date | Work |
|---|---|---|
| M1 Proof | 06-18 | §proof, gate |
| M2 Full converge + levers | 06-28 | 20-ep YOLOV-S two-class; sweep proposal hyperparams; test local-frame temporal (`lframe>0`) vs global; escalate to YOLOV-L/X/Swin-T if S plateaus <~14 |
| M3 Reattach top layer + imbalance | 07-08 | re-fit reranker + score-mode on YOLOV boxes; tackle collision<normal imbalance |
| M4 Official-evaluator align | 07-10→12 | re-validate on official val; keep loose+strict IoU candidates |
| M5 Docker + freeze | 08-10 | build heavier YOLOX/YOLOV image (Docker unverified); preflight; freeze |
| M6 Submit | 08-23 | final freeze + method report |

Total GPU: tens of GPU-hours (ample on 49GB).

## Honest risks
1. **Ceiling:** even converged YOLOV may only reach ~14–15 mAP — likely beats 0.13 but not a guaranteed win.
2. **Base swap is half the fix:** Stage2AK shows phantom class0 is a **ranking** problem (oracle-score lifts 0.21→0.45 on the *same* boxes) — a new detector won't fix that; only phantom class1 is genuine recall/localization. Pair pivot with ranker work.
3. **Class1/domain failure may persist:** train is frame-balanced but domain-skewed (35,084 phantom vs 291 animal); pilot class1 recall collapsed (0.015), animal class0 = 0.0.
4. **Validation thin (2 videos, animal-inflated):** judge the pivot on phantom per-case, not the combined headline.
5. **Metric-axis ambiguity** (AP50 vs COCO) until 07-10 can invert standing.
6. **Env/integration drift:** missing weights, two-class set not generated, no gt-free exporter yet, repo patched onto torch 2.11; Docker unverified. Re-smoke full-data path before the overnight job.
7. **Temporal on sparse data:** only 4/36 videos dense; index gaps (median delta 2–3) ≠ 24fps — use gap-aware, not fixed-stride, temporal windows.

**Pivot fails outright if:** trained YOLOV per-case phantom mAP50 ≤0.13 at M1 despite Stage-1 done right + full data (→ gap was never the base); or class1/animal failures survive 2-class + imbalance handling (→ illusory win on the hidden animal+human test).
