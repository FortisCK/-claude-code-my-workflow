#!/usr/bin/env python3
"""Evaluate YOLO-localized ROI classifier fusion for CATHACTION Task 2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import (  # noqa: E402
    RoiCropConfig,
    class_counts,
    crop_roi_with_padding,
    load_task2_samples_from_split,
)
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
    iou_xyxy,
)
from scripts.task2.train_roi_classifier import (  # noqa: E402
    binary_classification_metrics,
    build_transform,
    create_model,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class SelectedBox:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits/valid_combined_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits/valid_phantom_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits/valid_animal_labels.txt"),
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("outputs/task2/yolo/yolo11s_1024_clean_combined_e100/weights/best.pt"),
    )
    parser.add_argument(
        "--roi-checkpoint",
        type=Path,
        default=Path(
            "outputs/task2/roi_classifier/convnext_tiny_gtroi224_scale8_e30/checkpoints/best.pt"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/yolo_roi_fusion"))
    parser.add_argument("--name", default="yolo11s1024_convnext_tiny_gtroi224_stage2b")

    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--det-conf", type=float, default=0.001)
    parser.add_argument("--det-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument("--det-batch-size", type=int, default=32)
    parser.add_argument(
        "--source-chunk-size",
        type=int,
        default=64,
        help="Number of image paths passed to each Ultralytics predict call.",
    )
    parser.add_argument("--agnostic-nms", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--candidate-mode",
        choices=("top1", "topk", "all"),
        default="top1",
        help="Which YOLO boxes are passed to the ROI classifier.",
    )
    parser.add_argument("--top-k", type=int, default=5)

    parser.add_argument("--score-mode", choices=("roi", "det_roi"), default="roi")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--roi-batch-size", type=int, default=256)
    parser.add_argument("--amp", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    yolo_weights = resolve_path(args.yolo_weights)
    roi_checkpoint = resolve_path(args.roi_checkpoint)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    device_arg = "0" if device.type == "cuda" and device.index in (None, 0) else str(device)

    roi_model, roi_args = load_roi_model(roi_checkpoint, device)
    crop_config = RoiCropConfig(**roi_args["crop_config"])
    transform = build_transform(input_size=int(roi_args["input_size"]), training=False)

    args_record = json_ready(
        {
            **vars(args),
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "yolo_weights": repo_relative(yolo_weights, REPO_ROOT),
            "roi_checkpoint": repo_relative(roi_checkpoint, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "roi_model_args": roi_args,
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

    from ultralytics import YOLO

    yolo_model = YOLO(yolo_weights.as_posix())
    split_specs = [
        SplitSpec("valid_combined", resolve_path(args.valid_combined_split)),
        SplitSpec("valid_phantom", resolve_path(args.valid_phantom_split)),
        SplitSpec("valid_animal", resolve_path(args.valid_animal_split)),
    ]

    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        samples = load_task2_samples_from_split(data_root, split_spec.path)
        if args.limit is not None:
            samples = samples[: args.limit]
        print(f"Evaluating {split_spec.name}: {len(samples)} samples", flush=True)
        metrics, rows = evaluate_split(
            samples=samples,
            split_name=split_spec.name,
            yolo_model=yolo_model,
            roi_model=roi_model,
            transform=transform,
            crop_config=crop_config,
            device=device,
            device_arg=device_arg,
            args=args,
        )
        all_metrics[split_spec.name] = metrics
        metrics_path = run_dir / f"{split_spec.name}_metrics.json"
        predictions_path = run_dir / f"{split_spec.name}_predictions.csv"
        metrics_path.write_text(json.dumps(json_ready(metrics), indent=2) + "\n", encoding="utf-8")
        write_predictions_csv(predictions_path, rows)
        print(
            (
                f"{split_spec.name}: "
                f"two_cls_mAP50={metrics['detection_two_class_scores']['mAP50']:.6f} "
                f"two_cls_mAP50-95={metrics['detection_two_class_scores']['mAP50-95']:.6f} "
                f"argmax_mAP50={metrics['detection_argmax']['mAP50']:.6f} "
                f"loc_r50={metrics['localization']['recall_iou_0.50']:.4f} "
                f"cls_ap={metrics['classification_all_samples']['average_precision']:.4f}"
            ),
            flush=True,
        )

    summary_path = run_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(json_ready(all_metrics), indent=2) + "\n", encoding="utf-8")
    print(f"Saved Stage2B fusion results to {run_dir}", flush=True)
    return 0


def evaluate_split(
    *,
    samples: list[Task2Sample],
    split_name: str,
    yolo_model: Any,
    roi_model: torch.nn.Module,
    transform: Any,
    crop_config: RoiCropConfig,
    device: torch.device,
    device_arg: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ground_truths: list[DetectionGroundTruth] = []
    rows: list[dict[str, Any]] = []
    roi_tensors: list[torch.Tensor] = []
    pending_rows: list[dict[str, Any]] = []

    for chunk_start in range(0, len(samples), args.source_chunk_size):
        chunk_samples = samples[chunk_start : chunk_start + args.source_chunk_size]
        image_paths = [sample.image_path.as_posix() for sample in chunk_samples]
        results = yolo_model.predict(
            source=image_paths,
            imgsz=args.imgsz,
            conf=args.det_conf,
            iou=args.det_iou,
            max_det=args.max_det,
            batch=args.det_batch_size,
            device=device_arg,
            stream=True,
            verbose=False,
            agnostic_nms=args.agnostic_nms,
        )

        for chunk_offset, (sample, result) in enumerate(zip(chunk_samples, results)):
            index = chunk_start + chunk_offset + 1
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
            gt_box = tuple(float(value) for value in sample.box.xyxy_pixels(image_width, image_height))
            ground_truths.append(
                DetectionGroundTruth(sample.sample_id, sample.class_id, gt_box)  # type: ignore[arg-type]
            )

            selected_boxes = select_candidate_boxes(
                result,
                candidate_mode=args.candidate_mode,
                top_k=args.top_k,
            )
            if not selected_boxes:
                row = make_base_row(
                    split_name=split_name,
                    sample=sample,
                    image_width=image_width,
                    image_height=image_height,
                    gt_box=gt_box,  # type: ignore[arg-type]
                    selected_box=None,
                    candidate_count=0,
                )
                row.update(
                    {
                        "prob_collision": 0.0,
                        "score_normal": 0.0,
                        "score_collision": 0.0,
                        "pred_class": -1,
                        "pred_score": 0.0,
                    }
                )
                rows.append(row)
            else:
                with Image.open(sample.image_path) as image:
                    for selected_box in selected_boxes:
                        row = make_base_row(
                            split_name=split_name,
                            sample=sample,
                            image_width=image_width,
                            image_height=image_height,
                            gt_box=gt_box,  # type: ignore[arg-type]
                            selected_box=selected_box,
                            candidate_count=len(selected_boxes),
                        )
                        rows.append(row)
                        bounds = compute_roi_bounds_from_xyxy(selected_box.xyxy, crop_config)
                        roi = crop_roi_with_padding(image, bounds, fill=crop_config.fill)
                        roi_tensors.append(transform(roi))
                        pending_rows.append(row)
                        if len(roi_tensors) >= args.roi_batch_size:
                            flush_roi_predictions(
                                roi_model=roi_model,
                                roi_tensors=roi_tensors,
                                pending_rows=pending_rows,
                                device=device,
                                use_amp=args.amp,
                                score_mode=args.score_mode,
                                threshold=args.threshold,
                            )

            if args.progress_every > 0 and index % args.progress_every == 0:
                print(f"{split_name}: processed {index}/{len(samples)}", flush=True)

    flush_roi_predictions(
        roi_model=roi_model,
        roi_tensors=roi_tensors,
        pending_rows=pending_rows,
        device=device,
        use_amp=args.amp,
        score_mode=args.score_mode,
        threshold=args.threshold,
    )

    metrics = summarize_rows(
        split_name=split_name,
        samples=samples,
        rows=rows,
        ground_truths=ground_truths,
    )
    return metrics, rows


def select_candidate_boxes(result: Any, *, candidate_mode: str, top_k: int) -> list[SelectedBox]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    confidences = boxes.conf.detach().cpu().numpy()
    order = np.argsort(-confidences)
    if candidate_mode == "top1":
        order = order[:1]
    elif candidate_mode == "topk":
        order = order[: max(top_k, 1)]

    selected: list[SelectedBox] = []
    for rank, box_index in enumerate(order.tolist(), start=1):
        xyxy = tuple(float(value) for value in boxes.xyxy[box_index].detach().cpu().numpy().tolist())
        selected.append(
            SelectedBox(
                xyxy=xyxy,  # type: ignore[arg-type]
                confidence=float(confidences[box_index]),
                class_id=int(boxes.cls[box_index].detach().cpu().item()),
                rank=rank,
            )
        )
    return selected


def make_base_row(
    *,
    split_name: str,
    sample: Task2Sample,
    image_width: int,
    image_height: int,
    gt_box: tuple[float, float, float, float],
    selected_box: SelectedBox | None,
    candidate_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split_name,
        "sample_id": sample.sample_id,
        "image_path": repo_relative(sample.image_path, REPO_ROOT),
        "label_path": repo_relative(sample.label_path, REPO_ROOT),
        "image_width": image_width,
        "image_height": image_height,
        "gt_class": sample.class_id,
        "gt_x1": gt_box[0],
        "gt_y1": gt_box[1],
        "gt_x2": gt_box[2],
        "gt_y2": gt_box[3],
        "has_detection": selected_box is not None,
        "candidate_count": candidate_count,
    }
    if selected_box is None:
        row.update(
            {
                "candidate_rank": "",
                "det_class": "",
                "det_conf": 0.0,
                "det_x1": "",
                "det_y1": "",
                "det_x2": "",
                "det_y2": "",
                "det_gt_iou": 0.0,
            }
        )
        return row

    row.update(
        {
            "candidate_rank": selected_box.rank,
            "det_class": selected_box.class_id,
            "det_conf": selected_box.confidence,
            "det_x1": selected_box.xyxy[0],
            "det_y1": selected_box.xyxy[1],
            "det_x2": selected_box.xyxy[2],
            "det_y2": selected_box.xyxy[3],
            "det_gt_iou": iou_xyxy(selected_box.xyxy, gt_box),
        }
    )
    return row


def flush_roi_predictions(
    *,
    roi_model: torch.nn.Module,
    roi_tensors: list[torch.Tensor],
    pending_rows: list[dict[str, Any]],
    device: torch.device,
    use_amp: bool,
    score_mode: str,
    threshold: float,
) -> None:
    if not roi_tensors:
        return
    batch = torch.stack(roi_tensors).to(device, non_blocking=True)
    roi_model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
        logits = roi_model(batch)
        probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    for row, prob_collision in zip(pending_rows, probabilities):
        prob_collision_float = float(prob_collision)
        det_conf = float(row["det_conf"])
        if score_mode == "det_roi":
            score_normal = det_conf * (1.0 - prob_collision_float)
            score_collision = det_conf * prob_collision_float
        else:
            score_normal = 1.0 - prob_collision_float
            score_collision = prob_collision_float
        pred_class = int(score_collision >= threshold)
        pred_score = score_collision if pred_class == 1 else score_normal
        row.update(
            {
                "prob_collision": prob_collision_float,
                "score_normal": score_normal,
                "score_collision": score_collision,
                "pred_class": pred_class,
                "pred_score": pred_score,
            }
        )

    roi_tensors.clear()
    pending_rows.clear()


def summarize_rows(
    *,
    split_name: str,
    samples: list[Task2Sample],
    rows: list[dict[str, Any]],
    ground_truths: list[DetectionGroundTruth],
) -> dict[str, Any]:
    two_class_predictions: list[DetectionPrediction] = []
    argmax_predictions: list[DetectionPrediction] = []
    labels: list[int] = []
    probs: list[float] = []
    rows_by_sample = group_rows_by_sample(rows)

    for sample in samples:
        labels.append(sample.class_id)
        sample_rows = rows_by_sample.get(sample.sample_id, [])
        detected_rows = [row for row in sample_rows if row["has_detection"]]
        if detected_rows:
            probs.append(max(float(row.get("prob_collision", 0.0)) for row in detected_rows))
        else:
            probs.append(0.0)

    for row in rows:
        if not row["has_detection"]:
            continue
        box = (
            float(row["det_x1"]),
            float(row["det_y1"]),
            float(row["det_x2"]),
            float(row["det_y2"]),
        )
        two_class_predictions.append(
            DetectionPrediction(row["sample_id"], 0, float(row["score_normal"]), box)
        )
        two_class_predictions.append(
            DetectionPrediction(row["sample_id"], 1, float(row["score_collision"]), box)
        )
        argmax_predictions.append(
            DetectionPrediction(row["sample_id"], int(row["pred_class"]), float(row["pred_score"]), box)
        )

    class_counts_value = class_counts(samples)
    candidate_count = sum(1 for row in rows if row["has_detection"])
    detected_sample_count = sum(
        1 for sample_id in {sample.sample_id for sample in samples}
        if any(row["has_detection"] for row in rows_by_sample.get(sample_id, []))
    )
    top1_rows = [
        row
        for row in rows
        if (not row["has_detection"]) or str(row.get("candidate_rank", "")) == "1"
    ]
    return {
        "split": split_name,
        "samples": len(samples),
        "class_counts": class_counts_value,
        "candidate_count": candidate_count,
        "detected_sample_count": detected_sample_count,
        "no_detection_count": len(samples) - detected_sample_count,
        "detection_rate": detected_sample_count / max(len(samples), 1),
        "localization": summarize_best_localization(samples, rows),
        "top1_localization": summarize_localization(top1_rows),
        "candidate_localization": summarize_localization(rows),
        "localization_by_gt_class": {
            str(class_id): summarize_best_localization(
                [sample for sample in samples if sample.class_id == class_id],
                [row for row in rows if int(row["gt_class"]) == class_id],
            )
            for class_id in sorted(class_counts_value)
        },
        "classification_all_samples": binary_classification_metrics(labels, probs, [], threshold=0.5),
        "detection_two_class_scores": compute_detection_map(
            ground_truths,
            two_class_predictions,
            class_ids=(0, 1),
        ),
        "detection_argmax": compute_detection_map(
            ground_truths,
            argmax_predictions,
            class_ids=(0, 1),
        ),
    }


def group_rows_by_sample(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    return grouped


def summarize_best_localization(
    samples: Iterable[Task2Sample],
    rows: Iterable[dict[str, Any]],
) -> dict[str, float | int]:
    samples_list = list(samples)
    rows_by_sample = group_rows_by_sample(rows)
    best_ious: list[float] = []
    best_detected_ious: list[float] = []
    detected_count = 0
    for sample in samples_list:
        detected_ious = [
            float(row.get("det_gt_iou", 0.0))
            for row in rows_by_sample.get(sample.sample_id, [])
            if row["has_detection"]
        ]
        if detected_ious:
            detected_count += 1
            best_iou = max(detected_ious)
            best_ious.append(best_iou)
            best_detected_ious.append(best_iou)
        else:
            best_ious.append(0.0)

    result: dict[str, float | int] = {
        "samples": len(samples_list),
        "detected_count": detected_count,
        "mean_iou_all": float(np.mean(best_ious)) if best_ious else float("nan"),
        "mean_iou_detected": float(np.mean(best_detected_ious)) if best_detected_ious else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return result


def summarize_localization(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows_list = list(rows)
    ious_all = [float(row.get("det_gt_iou", 0.0)) for row in rows_list]
    ious_detected = [float(row["det_gt_iou"]) for row in rows_list if row["has_detection"]]
    result: dict[str, float | int] = {
        "samples": len(rows_list),
        "detected_count": len(ious_detected),
        "mean_iou_all": float(np.mean(ious_all)) if ious_all else float("nan"),
        "mean_iou_detected": float(np.mean(ious_detected)) if ious_detected else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in ious_all])) if ious_all else float("nan")
        )
    return result


def compute_roi_bounds_from_xyxy(
    box_xyxy: tuple[float, float, float, float],
    config: RoiCropConfig,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    side = max(box_width, box_height) * config.crop_scale
    side = max(side, float(config.min_crop_size))
    side = min(side, float(config.max_crop_size))
    side_int = max(1, int(math.ceil(side)))
    left = int(round(center_x - side_int / 2.0))
    top = int(round(center_y - side_int / 2.0))
    return left, top, left + side_int, top + side_int


def load_roi_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    model_name = checkpoint_args.get("model", "convnext_tiny")
    backend = checkpoint_args.get("backend", "auto")
    model = create_model(backend, model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    roi_args = {
        "backend": backend,
        "model": model_name,
        "input_size": int(checkpoint_args.get("input_size", 224)),
        "crop_config": checkpoint_args.get(
            "crop_config",
            asdict(RoiCropConfig()),
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metric": checkpoint.get("metrics", {}).get("valid_combined/average_precision"),
    }
    return model, roi_args


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
