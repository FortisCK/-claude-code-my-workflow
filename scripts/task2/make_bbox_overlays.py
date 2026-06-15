#!/usr/bin/env python3
"""Create Task 2 bounding-box overlays for split sanity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cathaction.data.task2 import (
    Task2Sample,
    build_task2_index,
    label_name_to_sample,
    read_task2_split_label_names,
    repo_relative,
    samples_by_label_name,
)


COLORS = {
    0: (64, 220, 96),
    1: (255, 64, 64),
}


def make_overlays(
    *,
    data_root: Path,
    split_file: Path,
    output_dir: Path,
    repo_root: Path,
    max_samples: int,
    class_names: tuple[str, str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = build_task2_index(data_root)
    by_label = samples_by_label_name(samples)
    label_names = read_task2_split_label_names(data_root, split_file)
    selected = [label_name_to_sample(by_label, label_name) for label_name in label_names[:max_samples]]

    written: list[str] = []
    for sample in selected:
        output_path = output_dir / f"{sample.sample_id}_bbox.png"
        draw_overlay(sample, output_path, class_names=class_names)
        written.append(repo_relative(output_path, repo_root))

    summary = {
        "data_root": repo_relative(data_root, repo_root),
        "split_file": repo_relative(split_file, repo_root),
        "output_dir": repo_relative(output_dir, repo_root),
        "requested_samples": max_samples,
        "written": len(written),
        "class_mapping_assumption": {
            "0": class_names[0],
            "1": class_names[1],
        },
        "files": written,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def draw_overlay(sample: Task2Sample, output_path: Path, *, class_names: tuple[str, str]) -> None:
    with Image.open(sample.image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    color = COLORS.get(sample.class_id, (255, 255, 0))
    x1, y1, x2, y2 = sample.box.xyxy_pixels(canvas.width, canvas.height)
    box = (round(x1), round(y1), round(x2), round(y2))
    draw.rectangle(box, outline=color, width=max(2, round(min(canvas.size) / 180)))
    cx = round(sample.box.x_center * canvas.width)
    cy = round(sample.box.y_center * canvas.height)
    radius = max(3, round(min(canvas.size) / 160))
    draw.line((cx - radius, cy, cx + radius, cy), fill=color, width=2)
    draw.line((cx, cy - radius, cx, cy + radius), fill=color, width=2)

    class_name = class_names[sample.class_id] if sample.class_id in {0, 1} else f"class_{sample.class_id}"
    label = f"{sample.sample_id} | {sample.class_id}:{class_name}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    draw.rectangle((0, 0, text_w + 12, text_h + 12), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=color, font=font)
    canvas.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--class-0-name", default="normal")
    parser.add_argument("--class-1-name", default="collision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root
    split_file = (repo_root / args.split_file).resolve() if not args.split_file.is_absolute() else args.split_file
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    summary = make_overlays(
        data_root=data_root,
        split_file=split_file,
        output_dir=output_dir,
        repo_root=repo_root,
        max_samples=args.max_samples,
        class_names=(args.class_0_name, args.class_1_name),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
