#!/usr/bin/env python3
"""Render qualitative PPT figures for the Task 2 best config (YOLOV-S mv 576 + Stage2U reranker Arm A).

Picks 3 representative phantom videos from the 6-video panel — HIGH (video_15),
MID (video_2), LOW (video_11) — and renders 2 evenly-spaced frames per video
with GT (green) + Arm A top-1 prediction (red, score and IoU) overlaid. Also
writes a 3x2 mosaic.

Output: Figures/task2_yolov_armA_examples/
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
GT_JSON = REPO / "datasets/collision_detection_yolov_mv/cathaction_valid_phantom.json"
ARMA_CSV = REPO / "outputs/task2/stage2v_yolov_armA/valid_phantom_predictions.csv"
RAW_CSV = REPO / "outputs/task2/yolov_mv_eval/run1/valid_phantom_predictions.csv"
IMG_DIR = REPO / "datasets/collision_detection/images"
OUT_DIR = REPO / "Figures/task2_yolov_armA_examples"

PICKS = {
    "video_15": ("HIGH", 0.708, 0.645),
    "video_2": ("MID", 0.425, 0.405),
    "video_11": ("LOW", 0.063, 0.031),
}

GREEN = (0, 220, 0)
RED = (255, 60, 60)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(path):
    coco = json.load(open(path))
    images = {im["id"]: im for im in coco["images"]}
    gt = defaultdict(list)
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        gt[images[ann["image_id"]]["file_name"]].append((x, y, x + w, y + h, ann["category_id"]))
    return coco, gt


def load_top1(csv_path):
    top1 = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            fname = r["sample_id"] + ".jpg"
            sc = float(r["score"])
            if fname in top1 and top1[fname][4] >= sc:
                continue
            top1[fname] = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), sc, int(r["class_id"]))
    return top1


def pick_frames(video, coco, gt, preds):
    candidates = sorted(
        im["file_name"]
        for im in coco["images"]
        if im["file_name"].startswith(video + "_") and im["file_name"] in gt and im["file_name"] in preds
    )
    if len(candidates) < 2:
        return candidates
    return [candidates[len(candidates) // 4], candidates[3 * len(candidates) // 4]]


def render_frame(fname, video, label, arma_per_case, raw_per_case, gt, preds, font, font_small):
    img = Image.open(IMG_DIR / fname).convert("RGB")
    draw = ImageDraw.Draw(img)

    iou_best = 0.0
    for gb in gt[fname]:
        draw.rectangle(gb[:4], outline=GREEN, width=3)
        if fname in preds:
            iou_best = max(iou_best, iou(gb[:4], preds[fname][:4]))

    if fname in preds:
        ab = preds[fname]
        draw.rectangle(ab[:4], outline=RED, width=2)
        tag = f"ArmA top-1  score={ab[4]:.2f}  IoU={iou_best:.2f}"
        tw = draw.textlength(tag, font=font_small)
        ty = max(0, ab[1] - 16)
        draw.rectangle([(ab[0], ty), (ab[0] + tw + 4, ty + 14)], fill=(0, 0, 0))
        draw.text((ab[0] + 2, ty), tag, fill=RED, font=font_small)

    cap = f"{video}  [{label}]   per-case mAP50: raw {raw_per_case:.3f} → ArmA {arma_per_case:.3f}"
    bar = 26
    bg = Image.new("RGB", (img.width, img.height + bar), (0, 0, 0))
    bg.paste(img, (0, bar))
    ImageDraw.Draw(bg).text((6, 5), cap, fill=(255, 255, 255), font=font)

    out = OUT_DIR / f"{video}_{label}_{fname}"
    bg.save(out)
    return bg, fname, iou_best


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coco, gt = load_gt(GT_JSON)
    arma = load_top1(ARMA_CSV)

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)

    panels = []
    for video, (label, a, r) in PICKS.items():
        frames = pick_frames(video, coco, gt, arma)
        if not frames:
            print(f"  [skip] {video}: no frames with both GT and ArmA pred")
            continue
        for fname in frames:
            bg, fn, iou_best = render_frame(fname, video, label, a, r, gt, arma, font, font_small)
            print(f"  wrote {bg.size}  {fn}  IoU={iou_best:.2f}")
            panels.append(bg)

    if panels:
        w = max(p.width for p in panels)
        h = max(p.height for p in panels)
        cols, rows = 3, 2
        mosaic = Image.new("RGB", (w * cols, h * rows), (15, 15, 15))
        for i, p in enumerate(panels[: cols * rows]):
            r, c = i // cols, i % cols
            if p.size != (w, h):
                scale = min(w / p.width, h / p.height)
                pp = p.resize((int(p.width * scale), int(p.height * scale)))
                bg = Image.new("RGB", (w, h), (15, 15, 15))
                bg.paste(pp, ((w - pp.width) // 2, (h - pp.height) // 2))
                p = bg
            mosaic.paste(p, (c * w, r * h))
        mosaic_path = OUT_DIR / "panel_mosaic.png"
        mosaic.save(mosaic_path)
        print(f"\nmosaic -> {mosaic_path}  ({mosaic.size})")
    print(f"\nall outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
