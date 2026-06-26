#!/usr/bin/env python3
"""Export RT-DETRv2 predictions to the detector-agnostic 14-column Task 2 CSV.

Runs the fine-tuned RT-DETRv2 on the SAME evaluation frames the champion GT
covers (read from a gt-run-dir's {split}_candidates_used.csv: sample_id +
image_path), so the result is apples-to-apples with the honest baseline
(evaluate_candidate_honest_baseline.py + the same --gt-run-dir). Emits the clean
gt-free schema accepted by freeze_task2_candidate.py.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("valid_phantom", "valid_animal", "valid_combined")
COLUMNS = ["sample_id", "video_id", "frame_index", "domain", "class_id", "score",
           "x1", "y1", "x2", "y2", "source", "source_rank", "score_mode", "policy_name"]
_STEM = re.compile(r"(.+)_([0-9]+)")


def parse_stem(sample_id: str) -> tuple[str, int]:
    m = _STEM.fullmatch(sample_id)
    return (m.group(1), int(m.group(2))) if m else (sample_id, 0)


def eval_frames(gt_run_dir: Path, split: str) -> list[dict]:
    """Unique (sample_id, image_path, domain) from candidates_used.csv."""
    path = gt_run_dir / f"{split}_candidates_used.csv"
    seen, rows = set(), []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        rows.append({"sample_id": sid, "image_path": r["image_path"], "domain": r.get("domain", "")})
    return rows


@torch.no_grad()
def predict_split(model, processor, device, frames, *, threshold, topk, batch_size) -> list[dict]:
    out_rows: list[dict] = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        images, sizes = [], []
        for f in chunk:
            ip = f["image_path"]
            ip = ip if Path(ip).is_absolute() else str(REPO_ROOT / ip)
            img = Image.open(ip).convert("RGB")
            images.append(img)
            sizes.append((img.size[1], img.size[0]))  # (H, W)
        inputs = processor(images=images, return_tensors="pt").to(device)
        outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor(sizes, device=device), threshold=threshold)
        for f, res in zip(chunk, results):
            vid, frame_idx = parse_stem(f["sample_id"])
            scores = res["scores"].tolist()
            labels = res["labels"].tolist()
            boxes = res["boxes"].tolist()
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
            for rank, i in enumerate(order):
                x1, y1, x2, y2 = boxes[i]
                out_rows.append({
                    "sample_id": f["sample_id"], "video_id": vid, "frame_index": frame_idx,
                    "domain": f["domain"], "class_id": int(labels[i]), "score": float(scores[i]),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "source": "rtdetrv2", "source_rank": rank, "score_mode": "raw",
                    "policy_name": "rtdetrv2_bare",
                })
    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--gt-run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.001)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=None, help="Override processor square resize (match training).")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    if args.image_size:
        processor.size = {"height": args.image_size, "width": args.image_size}
    model = RTDetrV2ForObjectDetection.from_pretrained(args.model_dir).to(device).eval()
    gt_dir = args.gt_run_dir if args.gt_run_dir.is_absolute() else (REPO_ROOT / args.gt_run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("valid_phantom", "valid_animal"):
        gt_csv = gt_dir / f"{split}_candidates_used.csv"
        if not gt_csv.is_file():
            print(f"skip {split}: no {gt_csv}")
            continue
        frames = eval_frames(gt_dir, split)
        rows = predict_split(model, processor, device, frames,
                             threshold=args.threshold, topk=args.topk, batch_size=args.batch_size)
        write_csv(args.output_dir / f"{split}_predictions.csv", rows)
        print(f"{split}: frames={len(frames)} pred_rows={len(rows)}", flush=True)

    # valid_combined = phantom + animal
    combined = []
    for split in ("valid_phantom", "valid_animal"):
        pcsv = args.output_dir / f"{split}_predictions.csv"
        if pcsv.is_file():
            combined += list(csv.DictReader(pcsv.open(encoding="utf-8")))
    if combined:
        write_csv(args.output_dir / "valid_combined_predictions.csv", combined)
        print(f"valid_combined: pred_rows={len(combined)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
