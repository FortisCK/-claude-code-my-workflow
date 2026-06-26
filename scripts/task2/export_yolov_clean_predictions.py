#!/usr/bin/env python3
"""Export YOLOV predictions to the detector-agnostic 14-column Task 2 CSV.

Reuses the proven model-load + decode from evaluate_yolov_proposals.py (YOLOV
video forward, scale-back decode), but emits the gt-FREE clean schema accepted by
freeze_task2_candidate.py / evaluate_candidate_honest_baseline.py. Iterates
valid_phantom + valid_animal (the prepared two-class jsons, which align frame-for-
frame with the champion GT), and writes valid_combined = concat.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
YOLOV_ROOT = REPO_ROOT / "external_repos" / "YOLOV"
import sys
for p in (REPO_ROOT, REPO_ROOT / "src", YOLOV_ROOT):
    if p.as_posix() not in sys.path:
        sys.path.insert(0, p.as_posix())

from scripts.task2.evaluate_yolov_proposals import load_model, output_to_proposals  # noqa: E402

COLUMNS = ["sample_id", "video_id", "frame_index", "domain", "class_id", "score",
           "x1", "y1", "x2", "y2", "source", "source_rank", "score_mode", "policy_name"]
_STEM = re.compile(r"(.+)_([0-9]+)")
SPLITS = (("valid_phantom", "cathaction_valid_phantom.json"),
          ("valid_animal", "cathaction_valid_animal.json"))


def parse_stem(sid: str):
    m = _STEM.fullmatch(sid)
    return (m.group(1), int(m.group(2))) if m else (sid, 0)


@torch.no_grad()
def export_split(model, exp, *, split_name, annotation, device, half, max_proposals, workers):
    exp.val_name = split_name
    exp.val_ann = annotation
    loader = exp.get_eval_loader(batch_size=int(exp.lframe_val + exp.gframe_val), data_num_workers=workers)
    dtype = torch.float16 if half else torch.float32
    rows = []
    for imgs, _, info_imgs, _, paths, _ in loader:
        imgs = imgs.to(device=device, dtype=dtype)
        outputs, _ = model(imgs, lframe=exp.lframe_val, gframe=exp.gframe_val)
        for output, info_img, image_path in zip(outputs, info_imgs, paths):
            sid = Path(str(image_path)).stem
            vid, frame_idx = parse_stem(sid)
            domain = "animal" if "animal" in sid else "phantom"
            h, w = int(info_img[0]), int(info_img[1])
            for p in output_to_proposals(output, image_width=w, image_height=h,
                                         test_size=tuple(exp.test_size), max_proposals=max_proposals):
                x1, y1, x2, y2 = p["xyxy"]
                rows.append({
                    "sample_id": sid, "video_id": vid, "frame_index": frame_idx, "domain": domain,
                    "class_id": p["class_id"], "score": p["confidence"],
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "source": "yolov", "source_rank": p["rank"] - 1,
                    "score_mode": "raw", "policy_name": "yolov_two_class",
                })
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp-file", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-proposals", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--half", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from yolox.exp import get_exp
    device = torch.device(args.device)
    exp = get_exp(args.exp_file.as_posix(), None)
    model = load_model(exp, args.ckpt, device=device, half=args.half)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for split_name, annotation in SPLITS:
        rows = export_split(model, exp, split_name=split_name, annotation=annotation,
                            device=device, half=args.half, max_proposals=args.max_proposals, workers=args.workers)
        write_csv(args.output_dir / f"{split_name}_predictions.csv", rows)
        combined += rows
        print(f"{split_name}: pred_rows={len(rows)}", flush=True)
    write_csv(args.output_dir / "valid_combined_predictions.csv", combined)
    print(f"valid_combined: pred_rows={len(combined)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
