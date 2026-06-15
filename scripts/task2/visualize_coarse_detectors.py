#!/usr/bin/env python3
"""Visualize coarse Task 2 detector predictions against GT boxes."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    weights: Path
    class_names: dict[int, str]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Prediction:
    detector: str
    rank: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    gt_iou: float
    color: tuple[int, int, int]


GT_COLORS = {
    0: (70, 230, 95),
    1: (255, 60, 60),
}
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument("--valid-phantom-split", type=Path, default=Path("configs/task2/splits/valid_phantom_labels.txt"))
    parser.add_argument("--valid-animal-split", type=Path, default=Path("configs/task2/splits/valid_animal_labels.txt"))
    parser.add_argument(
        "--two-class-weights",
        type=Path,
        default=Path("outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt"),
    )
    parser.add_argument(
        "--agnostic-weights",
        type=Path,
        default=Path("outputs/task2/yolo_proposal/yolo11s_1024_agnostic_clean_combined_e100/weights/best.pt"),
    )
    parser.add_argument(
        "--collision-weights",
        type=Path,
        default=Path("outputs/task2/yolo_collision/yolo11s_1024_collision_only_clean_combined_e80/weights/best.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/diagnostics/coarse_detection_visuals"))
    parser.add_argument("--samples-per-group", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--thumb-width", type=int, default=720)
    parser.add_argument("--contact-cols", type=int, default=2)
    parser.add_argument("--layout", choices=("overlay", "clean"), default="overlay")
    parser.add_argument("--clean-panel-width", type=int, default=1080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = resolve(args.data_root)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = select_samples(
        data_root=data_root,
        phantom_split=resolve(args.valid_phantom_split),
        animal_split=resolve(args.valid_animal_split),
        samples_per_group=args.samples_per_group,
        seed=args.seed,
    )
    detectors = build_detectors(args)
    predictions = run_detectors(
        detectors=detectors,
        samples=[item[1] for item in selected],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
    )

    summary_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for group_name, sample in selected:
        sample_predictions = predictions.get(sample.sample_id, [])
        panel_path = output_dir / f"{group_name}_{sample.sample_id}.png"
        if args.layout == "clean":
            row = draw_clean_panel(
                sample=sample,
                group_name=group_name,
                predictions=sample_predictions,
                output_path=panel_path,
                panel_width=args.clean_panel_width,
            )
        else:
            row = draw_panel(
                sample=sample,
                group_name=group_name,
                predictions=sample_predictions,
                output_path=panel_path,
                thumb_width=args.thumb_width,
            )
        panel_paths.append(panel_path)
        summary_rows.append(row)

    contact_path = output_dir / "contact_sheet.png"
    make_contact_sheet(panel_paths, contact_path, cols=args.contact_cols)
    index_path = output_dir / "index.html"
    write_html_index(index_path, contact_path, panel_paths, summary_rows)

    summary = {
        "output_dir": repo_relative(output_dir, REPO_ROOT),
        "contact_sheet": repo_relative(contact_path, REPO_ROOT),
        "index_html": repo_relative(index_path, REPO_ROOT),
        "samples": len(selected),
        "samples_per_group": args.samples_per_group,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "layout": args.layout,
        "detectors": [
            {
                "name": detector.name,
                "weights": repo_relative(detector.weights, REPO_ROOT),
            }
            for detector in detectors
        ],
        "rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(summary), indent=2), flush=True)
    return 0


def build_detectors(args: argparse.Namespace) -> list[DetectorSpec]:
    return [
        DetectorSpec(
            name="two_class",
            weights=resolve(args.two_class_weights),
            class_names={0: "normal", 1: "collision"},
            color=(55, 130, 255),
        ),
        DetectorSpec(
            name="agnostic",
            weights=resolve(args.agnostic_weights),
            class_names={0: "tool_roi"},
            color=(0, 210, 230),
        ),
        DetectorSpec(
            name="collision_only",
            weights=resolve(args.collision_weights),
            class_names={0: "collision_roi"},
            color=(255, 85, 230),
        ),
    ]


def select_samples(
    *,
    data_root: Path,
    phantom_split: Path,
    animal_split: Path,
    samples_per_group: int,
    seed: int,
) -> list[tuple[str, Task2Sample]]:
    split_specs = [
        ("valid_phantom", phantom_split),
        ("valid_animal", animal_split),
    ]
    rng = random.Random(seed)
    selected: list[tuple[str, Task2Sample]] = []
    for split_name, split_file in split_specs:
        samples = load_task2_samples_from_split(data_root, split_file)
        for class_id in (0, 1):
            class_samples = [sample for sample in samples if sample.class_id == class_id]
            if not class_samples:
                continue
            picked = stable_spread_sample(class_samples, samples_per_group, rng=rng)
            for sample in picked:
                selected.append((f"{split_name}_class{class_id}", sample))
    return selected


def stable_spread_sample(samples: list[Task2Sample], count: int, *, rng: random.Random) -> list[Task2Sample]:
    if count >= len(samples):
        return samples
    if count <= 1:
        return [samples[len(samples) // 2]]
    # Keep temporal spread, but jitter the starting order deterministically so we
    # do not always inspect only the earliest easy frames.
    shuffled = list(samples)
    rng.shuffle(shuffled)
    shuffled = sorted(shuffled[: max(count * 8, count)], key=lambda item: (item.video_id, item.frame_index))
    if len(shuffled) <= count:
        return shuffled
    indices = np.linspace(0, len(shuffled) - 1, num=count)
    return [shuffled[int(round(index))] for index in indices]


def run_detectors(
    *,
    detectors: list[DetectorSpec],
    samples: list[Task2Sample],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> dict[str, list[Prediction]]:
    from ultralytics import YOLO

    predictions_by_sample: dict[str, list[Prediction]] = {sample.sample_id: [] for sample in samples}
    image_paths = [sample.image_path.as_posix() for sample in samples]
    for detector in detectors:
        model = YOLO(detector.weights.as_posix())
        results = model.predict(
            source=image_paths,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch=min(16, max(1, len(image_paths))),
            device=device,
            stream=True,
            verbose=False,
        )
        for sample, result in zip(samples, results):
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            gt_box = sample.box.xyxy_pixels(image_width, image_height)
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            confidences = boxes.conf.detach().cpu().numpy()
            order = np.argsort(-confidences)
            for rank, box_index in enumerate(order.tolist(), start=1):
                class_id = int(boxes.cls[box_index].detach().cpu().item())
                xyxy = tuple(float(value) for value in boxes.xyxy[box_index].detach().cpu().numpy().tolist())
                predictions_by_sample[sample.sample_id].append(
                    Prediction(
                        detector=detector.name,
                        rank=rank,
                        class_id=class_id,
                        class_name=detector.class_names.get(class_id, f"class_{class_id}"),
                        confidence=float(confidences[box_index]),
                        xyxy=xyxy,  # type: ignore[arg-type]
                        gt_iou=float(iou_xyxy(xyxy, gt_box)),  # type: ignore[arg-type]
                        color=detector.color,
                    )
                )
    return predictions_by_sample


def draw_panel(
    *,
    sample: Task2Sample,
    group_name: str,
    predictions: list[Prediction],
    output_path: Path,
    thumb_width: int,
) -> dict[str, Any]:
    with Image.open(sample.image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(canvas.size) / 220))
    gt_box = sample.box.xyxy_pixels(canvas.width, canvas.height)
    gt_color = GT_COLORS.get(sample.class_id, (255, 255, 0))
    draw_box(draw, gt_box, gt_color, line_width + 1)
    draw_cross(draw, sample.box.x_center * canvas.width, sample.box.y_center * canvas.height, gt_color, line_width)
    draw_label(draw, (gt_box[0], gt_box[1]), f"GT c{sample.class_id}", gt_color, font)

    sorted_predictions = sorted(predictions, key=lambda item: (item.detector, item.rank))
    best_iou_by_detector: dict[str, float] = {}
    for prediction in sorted_predictions:
        best_iou_by_detector[prediction.detector] = max(
            best_iou_by_detector.get(prediction.detector, 0.0),
            prediction.gt_iou,
        )
        width = max(1, line_width - prediction.rank + 1)
        draw_box(draw, prediction.xyxy, prediction.color, width)
        label = (
            f"{short_detector_name(prediction.detector)}#{prediction.rank} "
            f"c{prediction.class_id} {prediction.confidence:.2f} "
            f"IoU {prediction.gt_iou:.2f}"
        )
        draw_label(draw, (prediction.xyxy[0], prediction.xyxy[1]), label, prediction.color, font)

    header_lines = [
        f"{group_name} | {sample.sample_id} | GT class {sample.class_id}",
        "GT: green=class0, red=class1 | two_class=blue | agnostic=cyan | collision_only=magenta",
    ]
    draw_header(draw, header_lines, font)

    if canvas.width > thumb_width:
        scale = thumb_width / canvas.width
        canvas = canvas.resize((thumb_width, int(round(canvas.height * scale))), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return {
        "group": group_name,
        "sample_id": sample.sample_id,
        "gt_class": sample.class_id,
        "image": repo_relative(sample.image_path, REPO_ROOT),
        "panel": repo_relative(output_path, REPO_ROOT),
        "prediction_count": len(sorted_predictions),
        "best_iou_by_detector": best_iou_by_detector,
    }


def draw_clean_panel(
    *,
    sample: Task2Sample,
    group_name: str,
    predictions: list[Prediction],
    output_path: Path,
    panel_width: int,
) -> dict[str, Any]:
    """Draw a human-readable three-panel diagnostic figure.

    The clean view intentionally shows only top-1 predictions in the visual
    panels. The row summary still reports the best IoU among all retained
    predictions, which is useful for diagnosing whether top-k contains a good
    candidate even if top-1 is wrong.
    """

    with Image.open(sample.image_path) as image:
        base = image.convert("RGB")
    font = ImageFont.load_default()
    gt_box = sample.box.xyxy_pixels(base.width, base.height)
    gt_color = GT_COLORS.get(sample.class_id, (255, 255, 0))
    top1_predictions = sorted(
        [prediction for prediction in predictions if prediction.rank == 1],
        key=lambda item: item.detector,
    )
    best_iou_by_detector: dict[str, float] = {}
    top1_iou_by_detector: dict[str, float] = {}
    for prediction in predictions:
        best_iou_by_detector[prediction.detector] = max(
            best_iou_by_detector.get(prediction.detector, 0.0),
            prediction.gt_iou,
        )
    for prediction in top1_predictions:
        top1_iou_by_detector[prediction.detector] = prediction.gt_iou

    col_width = max(240, panel_width // 3)
    line_width = max(2, round(min(base.size) / 220))

    gt_full = base.copy()
    draw_gt = ImageDraw.Draw(gt_full)
    draw_box(draw_gt, gt_box, gt_color, line_width + 1)
    draw_cross(draw_gt, sample.box.x_center * base.width, sample.box.y_center * base.height, gt_color, line_width)
    draw_header(draw_gt, [f"GT only | c{sample.class_id}", sample.sample_id], font)

    pred_full = base.copy()
    draw_pred = ImageDraw.Draw(pred_full)
    draw_box(draw_pred, gt_box, gt_color, max(1, line_width))
    draw_label(draw_pred, (gt_box[0], gt_box[1]), f"GT c{sample.class_id}", gt_color, font)
    for prediction in top1_predictions:
        draw_box(draw_pred, prediction.xyxy, prediction.color, line_width)
        label = (
            f"{short_detector_name(prediction.detector)} "
            f"c{prediction.class_id} conf {prediction.confidence:.2f} IoU {prediction.gt_iou:.2f}"
        )
        draw_label(draw_pred, (prediction.xyxy[0], prediction.xyxy[1]), label, prediction.color, font)
    draw_header(draw_pred, ["Top-1 predictions + GT", "blue=2cls cyan=agn magenta=col"], font)

    zoom_full = base.copy()
    draw_zoom = ImageDraw.Draw(zoom_full)
    draw_box(draw_zoom, gt_box, gt_color, line_width + 1)
    for prediction in top1_predictions:
        draw_box(draw_zoom, prediction.xyxy, prediction.color, line_width)
    zoom_crop = crop_around_box(zoom_full, gt_box, scale=7.5, min_side=240, max_side=520)
    draw_zoom_crop = ImageDraw.Draw(zoom_crop)
    draw_header(draw_zoom_crop, ["GT-centered zoom", "top-1 boxes only"], font)

    columns = [
        resize_to_width(gt_full, col_width),
        resize_to_width(pred_full, col_width),
        resize_to_width(zoom_crop, col_width),
    ]
    legend_lines = [
        f"{group_name} | {sample.sample_id} | GT class {sample.class_id}",
        format_prediction_summary("top1", top1_iou_by_detector),
        format_prediction_summary("best_topk", best_iou_by_detector),
    ]
    legend_height = 54
    panel_height = max(image.height for image in columns)
    output = Image.new("RGB", (col_width * len(columns), panel_height + legend_height), color=(18, 18, 18))
    x = 0
    for column in columns:
        output.paste(column, (x, 0))
        x += col_width
    draw_output = ImageDraw.Draw(output)
    draw_output.rectangle((0, panel_height, output.width, output.height), fill=(0, 0, 0))
    text_y = panel_height + 6
    for line in legend_lines:
        draw_output.text((8, text_y), line, fill=TEXT_FG, font=font)
        text_y += 15

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)

    return {
        "group": group_name,
        "sample_id": sample.sample_id,
        "gt_class": sample.class_id,
        "image": repo_relative(sample.image_path, REPO_ROOT),
        "panel": repo_relative(output_path, REPO_ROOT),
        "prediction_count": len(predictions),
        "top1_iou_by_detector": top1_iou_by_detector,
        "best_iou_by_detector": best_iou_by_detector,
    }


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    scale = width / image.width
    return image.resize((width, int(round(image.height * scale))), Image.Resampling.LANCZOS)


def crop_around_box(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    scale: float,
    min_side: int,
    max_side: int,
) -> Image.Image:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    side = min(float(max_side), max(float(min_side), side))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(left + side))
    bottom = int(round(top + side))
    return image.crop((max(0, left), max(0, top), min(image.width, right), min(image.height, bottom)))


def format_prediction_summary(label: str, values: dict[str, float]) -> str:
    if not values:
        return f"{label}: no predictions"
    parts = [f"{short_detector_name(key)}={value:.2f}" for key, value in sorted(values.items())]
    return f"{label} IoU: " + " | ".join(parts)


def draw_box(draw: ImageDraw.ImageDraw, xyxy: tuple[float, float, float, float], color: tuple[int, int, int], width: int) -> None:
    x1, y1, x2, y2 = xyxy
    draw.rectangle((round(x1), round(y1), round(x2), round(y2)), outline=color, width=width)


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], width: int) -> None:
    radius = max(4, width * 3)
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x = max(0, int(round(xy[0])))
    y = max(0, int(round(xy[1])) - 14)
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=TEXT_BG)
    draw.text((x, y), text, fill=color, font=font)


def draw_header(draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont) -> None:
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1] + 4)
    total_h = sum(line_heights) + 8
    total_w = max(widths) + 12 if widths else 12
    draw.rectangle((0, 0, total_w, total_h), fill=TEXT_BG)
    y = 6
    for line, line_h in zip(lines, line_heights):
        draw.text((6, y), line, fill=TEXT_FG, font=font)
        y += line_h


def make_contact_sheet(panel_paths: list[Path], output_path: Path, *, cols: int) -> None:
    if not panel_paths:
        return
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    cols = max(1, cols)
    rows = int(math.ceil(len(panels) / cols))
    cell_w = max(image.width for image in panels)
    cell_h = max(image.height for image in panels)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), color=(24, 24, 24))
    for index, panel in enumerate(panels):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(panel, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for panel in panels:
        panel.close()


def write_html_index(index_path: Path, contact_path: Path, panel_paths: list[Path], rows: list[dict[str, Any]]) -> None:
    def rel(path: Path) -> str:
        return path.relative_to(index_path.parent).as_posix()

    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Task2 coarse detection visuals</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee} img{max-width:100%;height:auto} table{border-collapse:collapse}td,th{border:1px solid #555;padding:4px 6px}</style>",
        "<h1>Task2 coarse detection visuals</h1>",
        f"<p>Contact sheet: <a href='{rel(contact_path)}'>{rel(contact_path)}</a></p>",
        f"<img src='{rel(contact_path)}'>",
        "<h2>Panels</h2>",
        "<table><tr><th>group</th><th>sample</th><th>gt</th><th>best IoU by detector</th><th>file</th></tr>",
    ]
    by_panel = {row["panel"]: row for row in rows}
    for panel_path in panel_paths:
        row = by_panel.get(repo_relative(panel_path, REPO_ROOT), {})
        parts.append(
            "<tr>"
            f"<td>{row.get('group', '')}</td>"
            f"<td>{row.get('sample_id', '')}</td>"
            f"<td>{row.get('gt_class', '')}</td>"
            f"<td>{json.dumps(row.get('best_iou_by_detector', {}), sort_keys=True)}</td>"
            f"<td><a href='{rel(panel_path)}'>{panel_path.name}</a></td>"
            "</tr>"
        )
    parts.append("</table>")
    index_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def short_detector_name(name: str) -> str:
    return {
        "two_class": "2cls",
        "agnostic": "agn",
        "collision_only": "col",
    }.get(name, name)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_ready(value.item())
    return value


if __name__ == "__main__":
    raise SystemExit(main())
