#!/usr/bin/env python3
"""Per-case (per-video) mAP of clean 14-col predictions vs FULL-FRAME COCO GT.

For the multi-video phantom validation panel: GT comes from the prepared COCO
json (every frame has its tip box — no GT-from-candidates recall optimism), and
predictions from the detector-agnostic clean CSVs. Reports per-video mAP50 /
mAP50-95 and the macro mean ± std across videos (the trustworthy panel number).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth, compute_detection_map, compute_detection_map_per_case,
)
from scripts.task2.evaluate_clean_prediction_topk import read_prediction_rows, to_predictions  # noqa: E402


def gt_from_coco(coco_json: Path) -> list[DetectionGroundTruth]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    img = {im["id"]: im for im in data["images"]}
    gts = []
    for ann in data["annotations"]:
        im = img[ann["image_id"]]
        sid = Path(im["file_name"]).stem
        x, y, w, h = ann["bbox"]
        gts.append(DetectionGroundTruth(sid, int(ann["category_id"]), (x, y, x + w, y + h)))
    return gts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--coco-gt", type=Path, required=True, help="Prepared cathaction_valid_*.json (full-frame GT).")
    p.add_argument("--pred-csv", type=Path, required=True, help="Clean 14-col predictions CSV for the same frames.")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gts = gt_from_coco(args.coco_gt if args.coco_gt.is_absolute() else REPO_ROOT / args.coco_gt)
    preds = to_predictions(read_prediction_rows(args.pred_csv if args.pred_csv.is_absolute() else REPO_ROOT / args.pred_csv))
    glob = compute_detection_map(gts, preds, class_ids=(0, 1))
    per = compute_detection_map_per_case(gts, preds, class_ids=(0, 1))
    pv = per["per_case"]
    map50s = [m["mAP50"] for m in pv.values()]
    map5095s = [m["mAP50-95"] for m in pv.values()]
    result = {
        "name": args.name,
        "num_videos": per["num_cases"],
        "gt_samples": len({g.sample_id for g in gts}),
        "global": {"mAP50": round(glob["mAP50"], 4), "mAP50-95": round(glob["mAP50-95"], 4)},
        "per_case_macro": {"mAP50": round(per["mAP50"], 4), "mAP50-95": round(per["mAP50-95"], 4)},
        "per_case_std": {"mAP50": round(st.pstdev(map50s), 4) if len(map50s) > 1 else 0.0,
                         "mAP50-95": round(st.pstdev(map5095s), 4) if len(map5095s) > 1 else 0.0},
        "per_video": {k: {"mAP50": round(v["mAP50"], 4), "mAP50-95": round(v["mAP50-95"], 4),
                          "num_gt": v["num_gt"]} for k, v in sorted(pv.items())},
    }
    print(f"[{args.name}] videos={result['num_videos']} gt={result['gt_samples']}")
    print(f"  GLOBAL    mAP50={result['global']['mAP50']} mAP50-95={result['global']['mAP50-95']}")
    print(f"  PER-CASE  mAP50={result['per_case_macro']['mAP50']} ± {result['per_case_std']['mAP50']} | "
          f"mAP50-95={result['per_case_macro']['mAP50-95']} ± {result['per_case_std']['mAP50-95']}")
    for vid, m in result["per_video"].items():
        print(f"    {vid}: mAP50={m['mAP50']} mAP50-95={m['mAP50-95']} gt={m['num_gt']}")
    if args.output:
        out = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
