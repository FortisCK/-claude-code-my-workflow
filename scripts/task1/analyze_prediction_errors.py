#!/usr/bin/env python3
"""Create per-sample Task 1 error reports and worst-case overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cathaction.data.task1 import load_task1_mask
from cathaction.metrics.segmentation import dice_score, per_class_dice, per_class_iou, pixel_accuracy


LABELS = [1, 2]
COLORS = {
    1: np.array([255, 64, 64], dtype=np.float32),
    2: np.array([64, 160, 255], dtype=np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    eval_manifest = _resolve(args.eval_manifest, repo_root)
    predictions_csv = _resolve(args.predictions_csv, repo_root)
    output_dir = _resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = {row["sample_id"]: row for row in _read_rows(eval_manifest)}
    prediction_rows = _read_rows(predictions_csv)
    metrics = []
    for row in prediction_rows:
        sample_id = row["sample_id"]
        eval_row = eval_rows.get(sample_id)
        if eval_row is None:
            continue
        pred_path = _resolve(row["prediction_path"], repo_root)
        target = load_task1_mask(
            _resolve(eval_row["mask_path"], repo_root),
            eval_row["mask_encoding"],
        ).astype(np.uint8)
        with Image.open(pred_path) as image:
            prediction = np.asarray(image.convert("L"), dtype=np.uint8)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {sample_id}: prediction {prediction.shape}, "
                f"target {target.shape}"
            )
        per_label_dice = per_class_dice(prediction, target, labels=LABELS)
        per_label_iou = per_class_iou(prediction, target, labels=LABELS)
        metrics.append(
            {
                "sample_id": sample_id,
                "collection": eval_row["collection"],
                "domain": eval_row["domain"],
                "image_path": eval_row["image_path"],
                "mask_path": eval_row["mask_path"],
                "prediction_path": row["prediction_path"],
                "dice": dice_score(prediction, target, labels=LABELS),
                "label_1_dice": per_label_dice[1],
                "label_2_dice": per_label_dice[2],
                "label_1_iou": per_label_iou[1],
                "label_2_iou": per_label_iou[2],
                "pixel_accuracy": pixel_accuracy(prediction, target),
                "target_label_1_px": int(np.sum(target == 1)),
                "target_label_2_px": int(np.sum(target == 2)),
                "pred_label_1_px": int(np.sum(prediction == 1)),
                "pred_label_2_px": int(np.sum(prediction == 2)),
            }
        )

    metrics_path = output_dir / "per_sample_metrics.csv"
    _write_metrics_csv(metrics_path, metrics)
    buckets = _worst_buckets(metrics, top_k=int(args.top_k))
    overlay_summary = {}
    for bucket_name, bucket_rows in buckets.items():
        bucket_dir = output_dir / bucket_name
        overlay_summary[bucket_name] = _write_bucket_overlays(
            bucket_name=bucket_name,
            rows=bucket_rows,
            output_dir=bucket_dir,
            repo_root=repo_root,
            alpha=float(args.alpha),
        )

    summary = {
        "eval_manifest": _repo_relative(eval_manifest, repo_root),
        "predictions_csv": _repo_relative(predictions_csv, repo_root),
        "output_dir": _repo_relative(output_dir, repo_root),
        "samples": len(metrics),
        "top_k": int(args.top_k),
        "metrics_csv": _repo_relative(metrics_path, repo_root),
        "aggregate": _aggregate(metrics),
        "buckets": {
            name: {
                "count": len(rows),
                "worst": _public_metric_row(rows[0]) if rows else None,
                "contact_sheet": overlay_summary[name].get("contact_sheet"),
            }
            for name, rows in buckets.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No prediction rows were matched.")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _worst_buckets(rows: list[dict[str, object]], *, top_k: int) -> dict[str, list[dict[str, object]]]:
    return {
        "worst_overall": sorted(rows, key=lambda row: float(row["dice"]))[:top_k],
        "worst_label_1": sorted(rows, key=lambda row: float(row["label_1_dice"]))[:top_k],
        "worst_label_2": sorted(rows, key=lambda row: float(row["label_2_dice"]))[:top_k],
        "worst_animal": sorted(
            [row for row in rows if row["domain"] == "animal"],
            key=lambda row: float(row["dice"]),
        )[:top_k],
        "worst_phantom": sorted(
            [row for row in rows if row["domain"] == "phantom"],
            key=lambda row: float(row["dice"]),
        )[:top_k],
    }


def _write_bucket_overlays(
    *,
    bucket_name: str,
    rows: list[dict[str, object]],
    output_dir: Path,
    repo_root: Path,
    alpha: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for rank, row in enumerate(rows, start=1):
        panel = _make_panel(row, repo_root=repo_root, alpha=alpha)
        filename = (
            f"{rank:02d}_{_safe_filename(str(row['sample_id']))}"
            f"_dice{float(row['dice']):.3f}.png"
        )
        path = output_dir / filename
        panel.save(path)
        panels.append(path)
    contact_sheet = output_dir / "contact_sheet.png"
    _write_contact_sheet(panels, contact_sheet, title=bucket_name)
    return {
        "panels": [_repo_relative(path, repo_root) for path in panels],
        "contact_sheet": _repo_relative(contact_sheet, repo_root) if panels else None,
    }


def _make_panel(row: dict[str, object], *, repo_root: Path, alpha: float) -> Image.Image:
    image_path = _resolve(str(row["image_path"]), repo_root)
    target_path = _resolve(str(row["mask_path"]), repo_root)
    prediction_path = _resolve(str(row["prediction_path"]), repo_root)

    with Image.open(image_path) as image:
        base = image.convert("RGB")
    target = load_task1_mask(target_path, "npy_multiclass").astype(np.uint8)
    with Image.open(prediction_path) as image:
        prediction = np.asarray(image.convert("L"), dtype=np.uint8)

    base = base.resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
    plain = _fit_tile(base)
    target_overlay = _fit_tile(_overlay_mask(base, target, alpha=alpha))
    pred_overlay = _fit_tile(_overlay_mask(base, prediction, alpha=alpha))
    error_overlay = _fit_tile(_error_overlay(base, prediction, target))

    tile_w, tile_h = plain.size
    header_h = 58
    panel = Image.new("RGB", (tile_w * 4, tile_h + header_h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    title = (
        f"{row['sample_id']} | {row['domain']} | dice={float(row['dice']):.3f} "
        f"| l1={float(row['label_1_dice']):.3f} | l2={float(row['label_2_dice']):.3f}"
    )
    draw.text((8, 6), title, fill=(255, 255, 255))
    draw.text((8, 28), "red=label_1 blue=label_2 | error: green=TP red=FN blue=FP", fill=(220, 220, 220))
    labels = ["image", "target", "prediction", "error"]
    tiles = [plain, target_overlay, pred_overlay, error_overlay]
    for index, (label, tile) in enumerate(zip(labels, tiles)):
        x = index * tile_w
        panel.paste(tile, (x, header_h))
        draw.rectangle((x, header_h, x + 94, header_h + 22), fill=(0, 0, 0))
        draw.text((x + 6, header_h + 4), label, fill=(255, 255, 255))
    return panel


def _overlay_mask(base: Image.Image, mask: np.ndarray, *, alpha: float) -> Image.Image:
    base_array = np.asarray(base, dtype=np.float32)
    color_layer = base_array.copy()
    for label, color in COLORS.items():
        color_layer[mask == label] = color
    foreground = np.isin(mask, list(COLORS)).reshape(mask.shape[0], mask.shape[1], 1)
    blended = np.where(foreground, (1.0 - alpha) * base_array + alpha * color_layer, base_array)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _error_overlay(base: Image.Image, prediction: np.ndarray, target: np.ndarray) -> Image.Image:
    base_array = np.asarray(base, dtype=np.float32) * 0.55
    pred_fg = prediction > 0
    target_fg = target > 0
    tp = pred_fg & target_fg
    fn = (~pred_fg) & target_fg
    fp = pred_fg & (~target_fg)
    base_array[tp] = np.array([40, 220, 90], dtype=np.float32)
    base_array[fn] = np.array([255, 60, 60], dtype=np.float32)
    base_array[fp] = np.array([60, 160, 255], dtype=np.float32)
    return Image.fromarray(np.clip(base_array, 0, 255).astype(np.uint8))


def _fit_tile(image: Image.Image, *, size: tuple[int, int] = (320, 240)) -> Image.Image:
    tile = Image.new("RGB", size, (0, 0, 0))
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.BILINEAR)
    left = (size[0] - image.size[0]) // 2
    top = (size[1] - image.size[1]) // 2
    tile.paste(image, (left, top))
    return tile


def _write_contact_sheet(paths: list[Path], output_path: Path, *, title: str) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((420, 100), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (420, 100), (0, 0, 0))
            canvas.paste(thumb, ((420 - thumb.size[0]) // 2, (100 - thumb.size[1]) // 2))
            thumbs.append(canvas)
    columns = 2
    rows = int(np.ceil(len(thumbs) / columns))
    header_h = 28
    sheet = Image.new("RGB", (columns * 420, rows * 100 + header_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), title, fill=(255, 255, 255))
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 420
        y = header_h + (index // columns) * 100
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result = {
        "dice": _mean(rows, "dice"),
        "label_1_dice": _mean(rows, "label_1_dice"),
        "label_2_dice": _mean(rows, "label_2_dice"),
        "animal_dice": _mean([row for row in rows if row["domain"] == "animal"], "dice"),
        "phantom_dice": _mean([row for row in rows if row["domain"] == "phantom"], "dice"),
    }
    return result


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    if not rows:
        return None
    return float(np.mean([float(row[key]) for row in rows]))


def _public_metric_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "domain": row["domain"],
        "dice": row["dice"],
        "label_1_dice": row["label_1_dice"],
        "label_2_dice": row["label_2_dice"],
    }


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
