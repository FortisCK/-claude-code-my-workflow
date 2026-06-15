#!/usr/bin/env python3
"""Evaluate Task1-segmentation geometry proposals for CATHACTION Task 2."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
except ImportError:  # pragma: no cover - exercised only in lighter test envs.
    _skimage_skeletonize = None

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402
from cathaction.training.task1_baseline import (  # noqa: E402
    _image_normalization_tensors,
    _image_size_tuple,
    _image_to_tensor,
    _normalize_image_tensor,
    build_model,
    load_checkpoint,
    primary_logits_for_label_mode,
    select_device,
)
from scripts.task2.evaluate_yolo_proposals import (  # noqa: E402
    format_summary,
    parse_top_k_values,
    summarize_proposals,
    write_rows_csv,
)


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class GeometryProposal:
    xyxy: tuple[float, float, float, float]
    source: str
    score: float


@dataclass(frozen=True)
class ResizeMeta:
    original_width: int
    original_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets/collision_detection"))
    parser.add_argument(
        "--valid-combined-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_combined_balanced_labels.txt"),
    )
    parser.add_argument(
        "--valid-phantom-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/valid_phantom_balanced_small_labels.txt"),
    )
    parser.add_argument(
        "--valid-animal-split",
        type=Path,
        default=Path("configs/task2/splits_stage2l_proposal/train_v0_v1_val_v2_valid_animal_labels.txt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/task1/smp_fpn_convnext_small_640_stage5/best_checkpoint.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/geometry_proposals"))
    parser.add_argument("--name", default="task1_stage5_geometry_d3_stage2l_panels")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dilation-px", type=int, default=3)
    parser.add_argument("--box-sizes", default="22,32,40,58,70,96")
    parser.add_argument(
        "--box-templates",
        default="22x22,32x32,40x40,58x58,70x70,96x96,20x58,24x58,30x70,36x70,58x20,70x30",
    )
    parser.add_argument("--max-proposals", type=int, default=120)
    parser.add_argument("--max-skeleton-foreground-fraction", type=float, default=0.03)
    parser.add_argument("--top-k", default="1,3,5,10,20,50")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--save-visuals", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    checkpoint_path = resolve_path(args.checkpoint)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_top_k_values(args.top_k)
    box_sizes = parse_int_list(args.box_sizes)
    box_templates = parse_box_templates(args.box_templates, fallback_sizes=box_sizes)

    device = select_device(args.device)
    payload = load_checkpoint(checkpoint_path, device)
    config = payload["config"]
    if isinstance(config.get("model"), dict) and config["model"].get("name") == "smp":
        config["model"] = dict(config["model"])
        config["model"]["encoder_weights"] = "none"
    data_cfg = config["data"]
    image_size = _image_size_tuple(data_cfg["image_size"])
    resize_mode = str(data_cfg.get("resize_mode", "direct"))
    label_mode = str(data_cfg["label_mode"])
    normalization = _image_normalization_tensors(data_cfg.get("normalization"))

    model = build_model(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    args_record = json_ready(
        {
            **vars(args),
            "repo_root": REPO_ROOT.as_posix(),
            "data_root": repo_relative(data_root, REPO_ROOT),
            "checkpoint": repo_relative(checkpoint_path, REPO_ROOT),
            "run_dir": repo_relative(run_dir, REPO_ROOT),
            "device": str(device),
            "task1_config_name": config.get("name"),
            "image_size": list(image_size),
            "resize_mode": resize_mode,
            "label_mode": label_mode,
            "box_sizes": box_sizes,
            "box_templates": box_templates,
            "top_k_values": top_k_values,
        }
    )
    (run_dir / "args.json").write_text(json.dumps(args_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(args_record, indent=2), flush=True)

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
        rows, visual_items = evaluate_split(
            samples=samples,
            split_name=split_spec.name,
            model=model,
            device=device,
            image_size=image_size,
            resize_mode=resize_mode,
            label_mode=label_mode,
            normalization=normalization,
            batch_size=int(args.batch_size),
            dilation_px=int(args.dilation_px),
            box_sizes=box_sizes,
            box_templates=box_templates,
            max_proposals=int(args.max_proposals),
            max_skeleton_foreground_fraction=float(args.max_skeleton_foreground_fraction),
            progress_every=int(args.progress_every),
            save_visuals=int(args.save_visuals),
        )
        metrics = summarize_proposals(samples, rows, top_k_values=top_k_values)
        all_metrics[split_spec.name] = metrics
        (run_dir / f"{split_spec.name}_metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        write_rows_csv(run_dir / f"{split_spec.name}_proposals.csv", rows)
        if args.save_visuals > 0:
            write_visuals(run_dir / "visuals" / split_spec.name, visual_items)
        print(format_summary(split_spec.name, metrics, top_k_values), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved geometry proposal evaluation to {run_dir}", flush=True)
    return 0


def evaluate_split(
    *,
    samples: list[Task2Sample],
    split_name: str,
    model: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    resize_mode: str,
    label_mode: str,
    normalization: tuple[torch.Tensor, torch.Tensor] | None,
    batch_size: int,
    dilation_px: int,
    box_sizes: list[int],
    box_templates: list[tuple[int, int]],
    max_proposals: int,
    max_skeleton_foreground_fraction: float,
    progress_every: int,
    save_visuals: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    visual_items: list[dict[str, Any]] = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        metas: list[ResizeMeta] = []
        for sample in batch_samples:
            with Image.open(sample.image_path) as image:
                image = image.convert("RGB")
                metas.append(compute_resize_meta(image.width, image.height, image_size, resize_mode))
                tensor = _image_to_tensor(image, image_size, resize_mode=resize_mode)
                tensor = _normalize_image_tensor(tensor, normalization)
                tensors.append(tensor)

        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = primary_logits_for_label_mode(model(batch), label_mode)
            predictions = torch.argmax(logits, dim=1).to("cpu").numpy().astype(np.uint8)

        for sample, pred_resized, meta in zip(batch_samples, predictions, metas):
            mask = restore_prediction_to_original(pred_resized, meta)
            proposals = generate_geometry_proposals(
                mask,
                dilation_px=dilation_px,
                box_sizes=box_sizes,
                box_templates=box_templates,
                max_proposals=max_proposals,
                max_skeleton_foreground_fraction=max_skeleton_foreground_fraction,
            )
            sample_rows = rows_for_sample(sample, proposals, split_name=split_name)
            rows.extend(sample_rows)
            if len(visual_items) < save_visuals:
                visual_items.append(
                    {
                        "sample": sample,
                        "mask": mask,
                        "rows": sample_rows,
                    }
                )
        current = start + len(batch_samples)
        if progress_every > 0 and (current % progress_every == 0 or current >= len(samples)):
            print(
                f"{split_name}: processed {current}/{len(samples)} samples",
                flush=True,
            )
    return rows, visual_items


def generate_geometry_proposals(
    mask: np.ndarray,
    *,
    dilation_px: int = 3,
    box_sizes: list[int] | None = None,
    box_templates: list[tuple[int, int]] | None = None,
    max_proposals: int = 120,
    max_skeleton_foreground_fraction: float = 1.0,
) -> list[GeometryProposal]:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D class mask, got shape={mask.shape}")
    if box_sizes is None:
        box_sizes = [22, 32, 40, 58, 70, 96]
    if box_templates is None:
        box_templates = [(size, size) for size in box_sizes]
    height, width = mask.shape
    class1 = mask == 1
    class2 = mask == 2
    foreground = class1 | class2
    proposals: list[GeometryProposal] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_box(
        xyxy: tuple[float, float, float, float],
        *,
        source: str,
        score: float,
    ) -> None:
        clipped = clip_box(xyxy, width=width, height=height)
        if clipped is None:
            return
        key = tuple(int(round(value)) for value in clipped)
        if key in seen:
            return
        seen.add(key)
        proposals.append(GeometryProposal(clipped, source=source, score=float(score)))

    def add_point_boxes(
        points: list[tuple[float, float, float]],
        *,
        source: str,
        score: float,
        templates: list[tuple[int, int]] | None = None,
    ) -> None:
        selected_templates = templates if templates is not None else box_templates
        for x, y, point_score in points:
            for box_width, box_height in selected_templates:
                add_box(
                    rect_box(x, y, box_width, box_height),
                    source=source,
                    score=score
                    + 0.000001 * float(point_score)
                    - 0.0001 * float(box_width + box_height) / 2.0,
                )

    if np.any(class1) and np.any(class2):
        class1_d = dilate_mask(class1, dilation_px)
        class2_d = dilate_mask(class2, dilation_px)
        overlap = class1_d & class2_d
        overlap_components = component_points(overlap, max_points=16)
        add_point_boxes(
            overlap_components,
            source=f"overlap_d{dilation_px}",
            score=1.00,
            templates=[(22, 22), (32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70)],
        )

        near_union = overlap | (dilate_mask(class1, dilation_px * 2) & dilate_mask(class2, dilation_px * 2))
        near_components = component_points(near_union, max_points=16)
        add_point_boxes(
            near_components,
            source=f"near_union_d{dilation_px * 2}",
            score=0.92,
            templates=[(32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70), (96, 96)],
        )

    if np.any(foreground):
        fg_d = dilate_mask(foreground, max(dilation_px, 2))
        for bbox, area in component_boxes(fg_d, max_components=12):
            x1, y1, x2, y2 = bbox
            add_box(expand_box_to_min_side(bbox, min_side=40, pad=8), source="component_union", score=0.82 + area * 1e-7)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            add_point_boxes(
                [(cx, cy, area)],
                source="component_centroid",
                score=0.78,
                templates=[(40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70), (96, 96)],
            )

        for label_value, binary in ((1, class1), (2, class2)):
            if not np.any(binary):
                continue
            for bbox, area in component_boxes(dilate_mask(binary, dilation_px), max_components=8):
                x1, y1, x2, y2 = bbox
                add_box(
                    expand_box_to_min_side(bbox, min_side=32, pad=6),
                    source=f"class{label_value}_component",
                    score=0.74 + area * 1e-7,
                )
                add_point_boxes(
                    [((x1 + x2) / 2.0, (y1 + y2) / 2.0, area)],
                    source=f"class{label_value}_centroid",
                    score=0.70,
                    templates=[(32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70)],
                )

        if float(np.mean(foreground)) <= float(max_skeleton_foreground_fraction):
            skeleton = skeletonize_mask(foreground)
            branch_points, end_points = skeleton_keypoints(skeleton)
            add_point_boxes(
                branch_points[:24],
                source="skeleton_branch",
                score=0.68,
                templates=[(32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70)],
            )
            add_point_boxes(
                end_points[:32],
                source="skeleton_endpoint",
                score=0.64,
                templates=[(32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70), (96, 96)],
            )
            sampled = sample_skeleton_points(skeleton, max_points=36)
            add_point_boxes(sampled, source="skeleton_sample", score=0.58)

        fg_points = foreground_grid_points(foreground, max_points=24)
        add_point_boxes(
            fg_points,
            source="foreground_grid",
            score=0.52,
            templates=[(32, 32), (40, 40), (20, 58), (24, 58), (30, 70), (58, 58), (70, 70)],
        )

    proposals = sorted(proposals, key=lambda item: item.score, reverse=True)
    return proposals[:max_proposals]


def rows_for_sample(
    sample: Task2Sample,
    proposals: list[GeometryProposal],
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    with Image.open(sample.image_path) as image:
        image_width, image_height = image.size
    gt_box = tuple(float(value) for value in sample.box.xyxy_pixels(image_width, image_height))
    if not proposals:
        return [
            {
                "split": split_name,
                "sample_id": sample.sample_id,
                "gt_class": sample.class_id,
                "has_proposal": False,
                "proposal_rank": "",
                "proposal_source": "",
                "proposal_score": 0.0,
                "proposal_gt_iou": 0.0,
                "proposal_x1": "",
                "proposal_y1": "",
                "proposal_x2": "",
                "proposal_y2": "",
            }
        ]
    rows: list[dict[str, Any]] = []
    for rank, proposal in enumerate(proposals, start=1):
        rows.append(
            {
                "split": split_name,
                "sample_id": sample.sample_id,
                "gt_class": sample.class_id,
                "has_proposal": True,
                "proposal_rank": rank,
                "proposal_source": proposal.source,
                "proposal_score": proposal.score,
                "proposal_gt_iou": iou_xyxy(proposal.xyxy, gt_box),
                "proposal_x1": proposal.xyxy[0],
                "proposal_y1": proposal.xyxy[1],
                "proposal_x2": proposal.xyxy[2],
                "proposal_y2": proposal.xyxy[3],
            }
        )
    return rows


def compute_resize_meta(
    original_width: int,
    original_height: int,
    image_size: tuple[int, int],
    resize_mode: str,
) -> ResizeMeta:
    target_height, target_width = image_size
    if resize_mode == "direct":
        return ResizeMeta(
            original_width=original_width,
            original_height=original_height,
            target_width=target_width,
            target_height=target_height,
            resized_width=target_width,
            resized_height=target_height,
            left=0,
            top=0,
        )
    if resize_mode not in {"aspect_pad", "letterbox"}:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    scale = min(target_width / float(original_width), target_height / float(original_height))
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    return ResizeMeta(
        original_width=original_width,
        original_height=original_height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        left=left,
        top=top,
    )


def restore_prediction_to_original(prediction: np.ndarray, meta: ResizeMeta) -> np.ndarray:
    cropped = prediction[
        meta.top : meta.top + meta.resized_height,
        meta.left : meta.left + meta.resized_width,
    ]
    image = Image.fromarray(cropped.astype(np.uint8), mode="L")
    restored = image.resize((meta.original_width, meta.original_height), Image.Resampling.NEAREST)
    return np.asarray(restored, dtype=np.uint8)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndi.binary_dilation(mask.astype(bool), structure=structure)


def component_boxes(mask: np.ndarray, *, max_components: int) -> list[tuple[tuple[float, float, float, float], float]]:
    labels, count = ndi.label(mask.astype(bool))
    if count == 0:
        return []
    objects = ndi.find_objects(labels)
    components: list[tuple[tuple[float, float, float, float], float]] = []
    for index, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        ys, xs = slc
        area = float(np.sum(labels[ys, xs] == index))
        if area <= 0:
            continue
        bbox = (float(xs.start), float(ys.start), float(xs.stop), float(ys.stop))
        components.append((bbox, area))
    components.sort(key=lambda item: item[1], reverse=True)
    return components[:max_components]


def component_points(mask: np.ndarray, *, max_points: int) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for bbox, area in component_boxes(mask, max_components=max_points):
        x1, y1, x2, y2 = bbox
        points.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, area))
    return points


def skeleton_keypoints(skeleton: np.ndarray) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    if not np.any(skeleton):
        return [], []
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), kernel, mode="constant", cval=0) - skeleton.astype(np.uint8)
    y_branch, x_branch = np.where(skeleton & (neighbors >= 3))
    y_end, x_end = np.where(skeleton & (neighbors == 1))
    branch = [(float(x), float(y), float(neighbors[y, x])) for y, x in zip(y_branch, x_branch)]
    ends = [(float(x), float(y), 1.0) for y, x in zip(y_end, x_end)]
    return farthest_subsample(branch, 64), farthest_subsample(ends, 64)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    if _skimage_skeletonize is not None:
        return _skimage_skeletonize(mask.astype(bool))
    return mask.astype(bool)


def sample_skeleton_points(skeleton: np.ndarray, *, max_points: int) -> list[tuple[float, float, float]]:
    y, x = np.where(skeleton)
    points = [(float(px), float(py), 1.0) for py, px in zip(y, x)]
    return farthest_subsample(points, max_points)


def foreground_grid_points(mask: np.ndarray, *, max_points: int) -> list[tuple[float, float, float]]:
    y, x = np.where(mask)
    if len(x) == 0:
        return []
    if len(x) <= max_points:
        return [(float(px), float(py), 1.0) for py, px in zip(y, x)]
    order = np.linspace(0, len(x) - 1, num=max_points, dtype=int)
    return [(float(x[idx]), float(y[idx]), 1.0) for idx in order]


def farthest_subsample(
    points: list[tuple[float, float, float]],
    max_points: int,
) -> list[tuple[float, float, float]]:
    if len(points) <= max_points:
        return points
    coords = np.asarray([(x, y) for x, y, _ in points], dtype=np.float32)
    selected = [0]
    distances = np.linalg.norm(coords - coords[0], axis=1)
    while len(selected) < max_points:
        index = int(np.argmax(distances))
        if index in selected:
            break
        selected.append(index)
        distances = np.minimum(distances, np.linalg.norm(coords - coords[index], axis=1))
    return [points[index] for index in selected]


def square_box(center_x: float, center_y: float, side: int | float) -> tuple[float, float, float, float]:
    half = float(side) / 2.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def rect_box(
    center_x: float,
    center_y: float,
    width: int | float,
    height: int | float,
) -> tuple[float, float, float, float]:
    half_width = float(width) / 2.0
    half_height = float(height) / 2.0
    return center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height


def expand_box_to_min_side(
    bbox: tuple[float, float, float, float],
    *,
    min_side: float,
    pad: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(float(min_side), x2 - x1 + 2.0 * pad, y2 - y1 + 2.0 * pad)
    return square_box(cx, cy, side)


def clip_box(
    xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), float(x1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y1 = max(0.0, min(float(height), float(y1)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def write_visuals(path: Path, items: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(items):
        sample: Task2Sample = item["sample"]
        mask: np.ndarray = item["mask"]
        proposal_rows: list[dict[str, Any]] = item["rows"]
        with Image.open(sample.image_path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        image_width, image_height = canvas.size
        gt = sample.box.xyxy_pixels(image_width, image_height)
        draw.rectangle(gt, outline=(255, 255, 0), width=3)
        for row in proposal_rows[:20]:
            if not row["has_proposal"]:
                continue
            color = (255, 0, 0) if int(row["proposal_rank"]) <= 5 else (0, 180, 255)
            draw.rectangle(
                (
                    float(row["proposal_x1"]),
                    float(row["proposal_y1"]),
                    float(row["proposal_x2"]),
                    float(row["proposal_y2"]),
                ),
                outline=color,
                width=1,
            )
        overlay = Image.fromarray(mask_to_rgb_overlay(mask, canvas.size), mode="RGB")
        blended = Image.blend(canvas, overlay, alpha=0.25)
        out_name = f"{index:03d}_{sample.sample_id}_class{sample.class_id}.jpg"
        blended.save(path / out_name, quality=92)
        rows.append({"sample_id": sample.sample_id, "class_id": str(sample.class_id), "visual": out_name})
    if rows:
        with (path / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "class_id", "visual"])
            writer.writeheader()
            writer.writerows(rows)


def mask_to_rgb_overlay(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[mask == 1] = (255, 0, 0)
    rgb[mask == 2] = (0, 120, 255)
    if (width, height) != size:
        rgb_image = Image.fromarray(rgb, mode="RGB").resize(size, Image.Resampling.NEAREST)
        return np.asarray(rgb_image, dtype=np.uint8)
    return rgb


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"Expected positive comma-separated integers, got {value!r}")
    return sorted(set(result))


def parse_box_templates(value: str, *, fallback_sizes: list[int]) -> list[tuple[int, int]]:
    if not value.strip():
        return [(size, size) for size in fallback_sizes]
    result: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" in item:
            width_text, height_text = item.split("x", maxsplit=1)
            width, height = int(width_text), int(height_text)
        else:
            width = height = int(item)
        if width <= 0 or height <= 0:
            raise ValueError(f"Box template dimensions must be positive: {item!r}")
        result.append((width, height))
    if not result:
        raise ValueError(f"Expected at least one box template, got {value!r}")
    return sorted(set(result), key=lambda pair: (pair[0] * pair[1], pair[0], pair[1]))


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
