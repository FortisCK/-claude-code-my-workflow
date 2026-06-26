#!/usr/bin/env python3
"""Honest baseline eval for Task 2 candidates on the held-out validation.

Reuses the proven GT/prediction loaders from evaluate_clean_prediction_topk (so
the box coordinate convention matches the pipeline exactly), and reports BOTH the
global-pool mAP and the official-aligned per-case (per-video) mAP, broken down by
class / domain / video. This is a measurement baseline, not a re-ranking.

Caveats it surfaces (NOT bugs introduced here, but properties of the pipeline):
  - GT is derived from each candidate's `*_candidates_used.csv`, i.e. only frames
    that produced >=1 candidate carry GT; frames with no candidate are not counted,
    so reported recall is optimistic vs a full-frame-GT evaluation.
  - valid_phantom is a single video (video_0); per-video variance is therefore
    only observable on the animal side (3 videos).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

from cathaction.metrics.detection import (  # noqa: E402
    compute_detection_map,
    compute_detection_map_per_case,
)
from scripts.task2.evaluate_clean_prediction_topk import (  # noqa: E402
    ground_truths_from_candidates,
    read_prediction_rows,
    to_predictions,
)

SPLITS = ("valid_phantom", "valid_animal", "valid_combined")
CSV_PATTERNS = ("{split}_predictions.csv", "{split}_domain_policy_predictions.csv")


def find_pred_csv(prediction_dir: Path, split: str) -> Path | None:
    for pattern in CSV_PATTERNS:
        candidate = prediction_dir / pattern.format(split=split)
        if candidate.is_file():
            return candidate
    return None


def evaluate_split(gt_run_dir: Path, prediction_dir: Path, split: str) -> dict | None:
    gt_csv = gt_run_dir / f"{split}_candidates_used.csv"
    pred_csv = find_pred_csv(prediction_dir, split)
    if not gt_csv.is_file() or pred_csv is None:
        return None
    gts = ground_truths_from_candidates(gt_csv)
    preds = to_predictions(read_prediction_rows(pred_csv))
    glob = compute_detection_map(gts, preds, class_ids=(0, 1))
    per_case = compute_detection_map_per_case(gts, preds, class_ids=(0, 1))
    per_video = {
        vid: {"mAP50": round(m["mAP50"], 4), "mAP50-95": round(m["mAP50-95"], 4),
              "num_gt": m["num_gt"]}
        for vid, m in per_case["per_case"].items()
    }
    return {
        "split": split,
        "gt_samples": len({g.sample_id for g in gts}),
        "pred_rows": len(preds),
        "global": {"mAP50": round(glob["mAP50"], 4), "mAP50-95": round(glob["mAP50-95"], 4),
                   "class0_ap50": round(glob["classes"]["0"]["ap50"], 4),
                   "class1_ap50": round(glob["classes"]["1"]["ap50"], 4)},
        "per_case": {"mAP50": round(per_case["mAP50"], 4), "mAP50-95": round(per_case["mAP50-95"], 4),
                     "num_cases": per_case["num_cases"]},
        "per_video": per_video,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--gt-run-dir", type=Path, required=True,
                        help="Run dir holding {split}_candidates_used.csv with gt_* columns.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pred_dir = args.prediction_dir if args.prediction_dir.is_absolute() else (REPO_ROOT / args.prediction_dir)
    gt_dir = args.gt_run_dir if args.gt_run_dir.is_absolute() else (REPO_ROOT / args.gt_run_dir)

    splits = {}
    for split in SPLITS:
        result = evaluate_split(gt_dir, pred_dir, split)
        if result is not None:
            splits[split] = result
            g, pc = result["global"], result["per_case"]
            print(f"[{args.name}] {split}: global mAP50={g['mAP50']} mAP50-95={g['mAP50-95']} "
                  f"| per-case mAP50={pc['mAP50']} mAP50-95={pc['mAP50-95']} (n={pc['num_cases']}) "
                  f"| gt_samples={result['gt_samples']} pred_rows={result['pred_rows']}", flush=True)
            for vid, m in result["per_video"].items():
                print(f"    video {vid}: mAP50={m['mAP50']} mAP50-95={m['mAP50-95']} gt={m['num_gt']}")

    manifest = {"name": args.name, "prediction_dir": pred_dir.as_posix(),
                "gt_run_dir": gt_dir.as_posix(), "splits": splits}
    output = args.output or (pred_dir / "honest_baseline_metrics.json")
    output = output if output.is_absolute() else (REPO_ROOT / output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
