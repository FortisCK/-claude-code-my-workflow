#!/usr/bin/env python3
"""Create visual overlays for Task 1 predicted segmentation masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    1: np.array([255, 64, 64], dtype=np.float32),
    2: np.array([64, 160, 255], dtype=np.float32),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_overlays(
    *,
    eval_manifest: Path,
    predictions_csv: Path,
    output_dir: Path,
    repo_root: Path,
    max_samples: int | None,
    alpha: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = {row["sample_id"]: row for row in read_rows(eval_manifest)}
    prediction_rows = read_rows(predictions_csv)
    if max_samples is not None:
        prediction_rows = prediction_rows[:max_samples]

    written: list[str] = []
    for row in prediction_rows:
        sample_id = row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            continue

        image_path = _resolve(eval_row["image_path"], repo_root)
        prediction_path = _resolve(row["prediction_path"], repo_root)
        overlay = make_overlay(image_path, prediction_path, alpha=alpha)
        output_path = output_dir / f"{_safe_filename(sample_id)}_overlay.png"
        overlay.save(output_path)
        written.append(_repo_relative(output_path, repo_root))

    summary = {
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "overlays": len(written),
        "overlay_files": written,
        "legend": {
            "label_1": "red",
            "label_2": "blue",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def make_overlay(image_path: Path, prediction_path: Path, *, alpha: float) -> Image.Image:
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    with Image.open(prediction_path) as prediction:
        mask = np.asarray(prediction.convert("L"), dtype=np.uint8)

    base = base.resize((mask.shape[1], mask.shape[0]), Image.Resampling.BILINEAR)
    base_array = np.asarray(base, dtype=np.float32)
    color_layer = base_array.copy()

    for label, color in COLORS.items():
        color_layer[mask == label] = color

    blended = np.where(
        np.isin(mask, list(COLORS)).reshape(mask.shape[0], mask.shape[1], 1),
        (1.0 - alpha) * base_array + alpha * color_layer,
        base_array,
    )
    result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, 238, 42), fill=(0, 0, 0))
    draw.text((8, 6), "label_1: red", fill=(255, 64, 64))
    draw.text((8, 24), "label_2: blue", fill=(64, 160, 255))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = create_overlays(
        eval_manifest=_resolve(args.eval_manifest, args.repo_root),
        predictions_csv=_resolve(args.predictions_csv, args.repo_root),
        output_dir=_resolve(args.output_dir, args.repo_root),
        repo_root=args.repo_root.resolve(),
        max_samples=args.max_samples,
        alpha=args.alpha,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _resolve(path: Path | str, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return base / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
