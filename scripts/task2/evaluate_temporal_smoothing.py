#!/usr/bin/env python3
"""Evaluate temporal smoothing over Task 2 proposal CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import parse_task2_sample_stem, repo_relative  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    sample_id: str
    split: str
    video_id: str
    frame_index: int
    gt_class: int
    image_width: int
    image_height: int
    gt_xyxy: tuple[float, float, float, float]
    rank: int
    proposal_class: int
    confidence: float
    xyxy: tuple[float, float, float, float]
    source_iou: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def size(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1), max(0.0, y2 - y1)

    @property
    def gt_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.gt_xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass(frozen=True)
class DecodedFrame:
    sample_id: str
    split: str
    video_id: str
    frame_index: int
    gt_class: int
    image_width: int
    image_height: int
    gt_xyxy: tuple[float, float, float, float]
    selected: Candidate | None

    @property
    def iou(self) -> float:
        if self.selected is None:
            return 0.0
        return float(iou_xyxy(self.selected.xyxy, self.gt_xyxy))

    @property
    def center_distance(self) -> float:
        if self.selected is None:
            return float("inf")
        cx, cy = self.selected.center
        gx, gy = self.selected.gt_center
        return math.hypot(cx - gx, cy - gy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal-dir",
        type=Path,
        default=Path("outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels"),
    )
    parser.add_argument("--splits", default="valid_animal,valid_phantom,valid_combined")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/temporal_smoothing"))
    parser.add_argument("--name", default="stage2m_yolo_stage2l_viterbi")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--distance-weights", default="0,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--size-weights", default="0,0.005,0.01")
    parser.add_argument("--class-switch-weights", default="0")
    parser.add_argument("--distance-scale", type=float, default=25.0)
    parser.add_argument("--size-scale", type=float, default=20.0)
    parser.add_argument("--gap-power", type=float, default=1.0)
    parser.add_argument("--max-transition-cost", type=float, default=25.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proposal_dir = resolve_path(args.proposal_dir)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        **vars(args),
        "proposal_dir": repo_relative(proposal_dir, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "distance_weights": parse_float_list(args.distance_weights),
        "size_weights": parse_float_list(args.size_weights),
        "class_switch_weights": parse_float_list(args.class_switch_weights),
    }
    (run_dir / "args.json").write_text(json.dumps(json_ready(config), indent=2) + "\n", encoding="utf-8")

    all_summary: dict[str, Any] = {}
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        csv_path = proposal_dir / f"{split}_proposals.csv"
        candidates_by_sample = load_proposal_csv(csv_path, top_k=args.top_k)
        sample_frames = make_sample_frames(candidates_by_sample)
        split_summary = evaluate_split_sweep(
            sample_frames=sample_frames,
            candidates_by_sample=candidates_by_sample,
            args=args,
            run_dir=run_dir,
            split=split,
        )
        all_summary[split] = split_summary
        print(format_split_summary(split, split_summary), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_summary), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved temporal smoothing evaluation to {run_dir}", flush=True)
    return 0


def evaluate_split_sweep(
    *,
    sample_frames: list[DecodedFrame],
    candidates_by_sample: dict[str, list[Candidate]],
    args: argparse.Namespace,
    run_dir: Path,
    split: str,
) -> dict[str, Any]:
    distance_weights = parse_float_list(args.distance_weights)
    size_weights = parse_float_list(args.size_weights)
    class_switch_weights = parse_float_list(args.class_switch_weights)

    raw_top1 = evaluate_decoded_frames(raw_top1_frames(sample_frames, candidates_by_sample))
    raw_topk = evaluate_topk_oracle(sample_frames, candidates_by_sample, top_k=args.top_k)

    sweep_rows: list[dict[str, Any]] = []
    decoded_by_key: dict[str, list[DecodedFrame]] = {}
    for distance_weight in distance_weights:
        for size_weight in size_weights:
            for class_switch_weight in class_switch_weights:
                decoded = decode_all_videos(
                    sample_frames=sample_frames,
                    candidates_by_sample=candidates_by_sample,
                    distance_weight=distance_weight,
                    size_weight=size_weight,
                    class_switch_weight=class_switch_weight,
                    distance_scale=args.distance_scale,
                    size_scale=args.size_scale,
                    gap_power=args.gap_power,
                    max_transition_cost=args.max_transition_cost,
                    eps=args.eps,
                )
                metrics = evaluate_decoded_frames(decoded)
                key = make_sweep_key(distance_weight, size_weight, class_switch_weight)
                decoded_by_key[key] = decoded
                sweep_rows.append(
                    {
                        "key": key,
                        "distance_weight": distance_weight,
                        "size_weight": size_weight,
                        "class_switch_weight": class_switch_weight,
                        **flatten_metrics(metrics),
                    }
                )

    sweep_rows.sort(
        key=lambda row: (
            float(row["overall_center_recall_20px"]),
            float(row["overall_recall_iou_0.50"]),
            float(row["overall_mean_iou"]),
        ),
        reverse=True,
    )
    best_key = str(sweep_rows[0]["key"]) if sweep_rows else ""
    best_decoded = decoded_by_key[best_key] if best_key else []
    write_decoded_csv(run_dir / f"{split}_best_decoded.csv", best_decoded)
    write_sweep_csv(run_dir / f"{split}_sweep.csv", sweep_rows)

    summary = {
        "samples": len(sample_frames),
        "top_k": args.top_k,
        "raw_top1": raw_top1,
        "raw_topk_oracle": raw_topk,
        "best_temporal_key": best_key,
        "best_temporal": evaluate_decoded_frames(best_decoded),
        "best_sweep_row": sweep_rows[0] if sweep_rows else None,
        "sweep_rows": sweep_rows,
    }
    (run_dir / f"{split}_metrics.json").write_text(
        json.dumps(json_ready(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_proposal_csv(path: Path, *, top_k: int) -> dict[str, list[Candidate]]:
    candidates_by_sample: dict[str, list[Candidate]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_id = row["sample_id"]
            video_id, frame_index = parse_task2_sample_stem(sample_id)
            if parse_bool(row["has_proposal"]):
                rank = int(row["proposal_rank"])
                if rank > top_k:
                    continue
                candidate = Candidate(
                    sample_id=sample_id,
                    split=row["split"],
                    video_id=video_id,
                    frame_index=frame_index,
                    gt_class=int(row["gt_class"]),
                    image_width=int(row["image_width"]),
                    image_height=int(row["image_height"]),
                    gt_xyxy=parse_xyxy(row, "gt"),
                    rank=rank,
                    proposal_class=int(float(row["proposal_class"])),
                    confidence=float(row["proposal_conf"]),
                    xyxy=parse_xyxy(row, "proposal"),
                    source_iou=float(row["proposal_gt_iou"]),
                )
                candidates_by_sample.setdefault(sample_id, []).append(candidate)
            else:
                candidates_by_sample.setdefault(sample_id, [])
    for sample_id in candidates_by_sample:
        candidates_by_sample[sample_id].sort(key=lambda item: item.rank)
    return candidates_by_sample


def make_sample_frames(candidates_by_sample: dict[str, list[Candidate]]) -> list[DecodedFrame]:
    frames: list[DecodedFrame] = []
    for sample_id, candidates in candidates_by_sample.items():
        if candidates:
            first = candidates[0]
            frames.append(
                DecodedFrame(
                    sample_id=sample_id,
                    split=first.split,
                    video_id=first.video_id,
                    frame_index=first.frame_index,
                    gt_class=first.gt_class,
                    image_width=first.image_width,
                    image_height=first.image_height,
                    gt_xyxy=first.gt_xyxy,
                    selected=None,
                )
            )
        else:
            # No-proposal rows in the current CSV still include sample metadata,
            # but the Stage2L CSVs used here always have candidates. Keeping this
            # explicit makes failure easier to diagnose if a future CSV differs.
            video_id, frame_index = parse_task2_sample_stem(sample_id)
            raise ValueError(f"Cannot reconstruct metadata for no-proposal sample: {video_id} {frame_index}")
    return sorted(frames, key=lambda item: (item.video_id, item.frame_index))


def raw_top1_frames(
    sample_frames: list[DecodedFrame],
    candidates_by_sample: dict[str, list[Candidate]],
) -> list[DecodedFrame]:
    decoded: list[DecodedFrame] = []
    for frame in sample_frames:
        candidates = candidates_by_sample.get(frame.sample_id, [])
        decoded.append(replace_selected(frame, candidates[0] if candidates else None))
    return decoded


def decode_all_videos(
    *,
    sample_frames: list[DecodedFrame],
    candidates_by_sample: dict[str, list[Candidate]],
    distance_weight: float,
    size_weight: float,
    class_switch_weight: float,
    distance_scale: float,
    size_scale: float,
    gap_power: float,
    max_transition_cost: float,
    eps: float,
) -> list[DecodedFrame]:
    by_video: dict[str, list[DecodedFrame]] = {}
    for frame in sample_frames:
        by_video.setdefault(frame.video_id, []).append(frame)

    decoded: list[DecodedFrame] = []
    for video_frames in by_video.values():
        video_frames = sorted(video_frames, key=lambda item: item.frame_index)
        decoded.extend(
            decode_video(
                video_frames=video_frames,
                candidates_by_sample=candidates_by_sample,
                distance_weight=distance_weight,
                size_weight=size_weight,
                class_switch_weight=class_switch_weight,
                distance_scale=distance_scale,
                size_scale=size_scale,
                gap_power=gap_power,
                max_transition_cost=max_transition_cost,
                eps=eps,
            )
        )
    return sorted(decoded, key=lambda item: (item.video_id, item.frame_index))


def decode_video(
    *,
    video_frames: list[DecodedFrame],
    candidates_by_sample: dict[str, list[Candidate]],
    distance_weight: float,
    size_weight: float,
    class_switch_weight: float,
    distance_scale: float,
    size_scale: float,
    gap_power: float,
    max_transition_cost: float,
    eps: float,
) -> list[DecodedFrame]:
    candidate_lists = [candidates_by_sample.get(frame.sample_id, []) for frame in video_frames]
    if not candidate_lists:
        return []
    if any(not candidates for candidates in candidate_lists):
        return [
            replace_selected(frame, candidates[0] if candidates else None)
            for frame, candidates in zip(video_frames, candidate_lists)
        ]

    scores: list[list[float]] = []
    backptrs: list[list[int]] = []
    scores.append([node_score(candidate, eps=eps) for candidate in candidate_lists[0]])
    backptrs.append([-1 for _ in candidate_lists[0]])

    for frame_index in range(1, len(candidate_lists)):
        previous_candidates = candidate_lists[frame_index - 1]
        current_candidates = candidate_lists[frame_index]
        previous_scores = scores[-1]
        gap = max(
            1,
            video_frames[frame_index].frame_index - video_frames[frame_index - 1].frame_index,
        )
        current_scores: list[float] = []
        current_backptrs: list[int] = []
        for current in current_candidates:
            best_score = -float("inf")
            best_previous_index = 0
            for previous_index, previous in enumerate(previous_candidates):
                transition = transition_cost(
                    previous,
                    current,
                    gap=gap,
                    distance_weight=distance_weight,
                    size_weight=size_weight,
                    class_switch_weight=class_switch_weight,
                    distance_scale=distance_scale,
                    size_scale=size_scale,
                    gap_power=gap_power,
                    max_transition_cost=max_transition_cost,
                )
                total = previous_scores[previous_index] + node_score(current, eps=eps) - transition
                if total > best_score:
                    best_score = total
                    best_previous_index = previous_index
            current_scores.append(best_score)
            current_backptrs.append(best_previous_index)
        scores.append(current_scores)
        backptrs.append(current_backptrs)

    last_index = int(np.argmax(scores[-1]))
    path_indices = [last_index]
    for frame_index in range(len(candidate_lists) - 1, 0, -1):
        last_index = backptrs[frame_index][last_index]
        path_indices.append(last_index)
    path_indices.reverse()
    return [
        replace_selected(frame, candidates[index])
        for frame, candidates, index in zip(video_frames, candidate_lists, path_indices)
    ]


def node_score(candidate: Candidate, *, eps: float) -> float:
    rank_penalty = 0.03 * math.log1p(max(0, candidate.rank - 1))
    return math.log(max(candidate.confidence, eps)) - rank_penalty


def transition_cost(
    previous: Candidate,
    current: Candidate,
    *,
    gap: int,
    distance_weight: float,
    size_weight: float,
    class_switch_weight: float,
    distance_scale: float,
    size_scale: float,
    gap_power: float,
    max_transition_cost: float,
) -> float:
    pcx, pcy = previous.center
    ccx, ccy = current.center
    distance = math.hypot(ccx - pcx, ccy - pcy) / max(1.0, float(gap) ** gap_power)
    pwidth, pheight = previous.size
    cwidth, cheight = current.size
    size_delta = math.hypot(cwidth - pwidth, cheight - pheight) / max(1.0, float(gap) ** gap_power)
    class_delta = 1.0 if previous.proposal_class != current.proposal_class else 0.0
    cost = (
        distance_weight * min(distance / distance_scale, max_transition_cost)
        + size_weight * min(size_delta / size_scale, max_transition_cost)
        + class_switch_weight * class_delta
    )
    return cost


def evaluate_topk_oracle(
    sample_frames: list[DecodedFrame],
    candidates_by_sample: dict[str, list[Candidate]],
    *,
    top_k: int,
) -> dict[str, Any]:
    decoded: list[DecodedFrame] = []
    for frame in sample_frames:
        candidates = candidates_by_sample.get(frame.sample_id, [])[:top_k]
        best = max(candidates, key=lambda item: item.source_iou, default=None)
        decoded.append(replace_selected(frame, best))
    return evaluate_decoded_frames(decoded)


def evaluate_decoded_frames(frames: list[DecodedFrame]) -> dict[str, Any]:
    result = summarize_frame_subset(frames)
    by_class: dict[str, Any] = {}
    for class_id in sorted({frame.gt_class for frame in frames}):
        by_class[str(class_id)] = summarize_frame_subset(
            [frame for frame in frames if frame.gt_class == class_id]
        )
    result["by_gt_class"] = by_class
    return result


def summarize_frame_subset(frames: list[DecodedFrame]) -> dict[str, Any]:
    ious = [frame.iou for frame in frames]
    center_distances = [frame.center_distance for frame in frames]
    finite_center_distances = [value for value in center_distances if math.isfinite(value)]
    summary: dict[str, Any] = {
        "samples": len(frames),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "median_iou": float(np.median(ious)) if ious else float("nan"),
        "mean_center_distance_px": (
            float(np.mean(finite_center_distances)) if finite_center_distances else float("nan")
        ),
        "median_center_distance_px": (
            float(np.median(finite_center_distances)) if finite_center_distances else float("nan")
        ),
    }
    for threshold in (0.25, 0.50, 0.75):
        summary[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in ious])) if ious else float("nan")
        )
    for threshold in (5, 10, 20):
        summary[f"center_recall_{threshold}px"] = (
            float(np.mean([distance <= threshold for distance in center_distances]))
            if center_distances
            else float("nan")
        )
    return summary


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "by_gt_class":
            for class_id, class_metrics in value.items():
                for metric_name, metric_value in class_metrics.items():
                    result[f"class{class_id}_{metric_name}"] = metric_value
        else:
            result[f"overall_{key}"] = value
    return result


def write_decoded_csv(path: Path, frames: list[DecodedFrame]) -> None:
    rows = [decoded_frame_to_row(frame) for frame in frames]
    write_dict_rows(path, rows)


def decoded_frame_to_row(frame: DecodedFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": frame.split,
        "sample_id": frame.sample_id,
        "video_id": frame.video_id,
        "frame_index": frame.frame_index,
        "gt_class": frame.gt_class,
        "image_width": frame.image_width,
        "image_height": frame.image_height,
        "gt_x1": frame.gt_xyxy[0],
        "gt_y1": frame.gt_xyxy[1],
        "gt_x2": frame.gt_xyxy[2],
        "gt_y2": frame.gt_xyxy[3],
        "selected": frame.selected is not None,
        "selected_rank": "",
        "selected_class": "",
        "selected_conf": 0.0,
        "selected_x1": "",
        "selected_y1": "",
        "selected_x2": "",
        "selected_y2": "",
        "selected_iou": frame.iou,
        "selected_center_distance_px": frame.center_distance,
    }
    if frame.selected is not None:
        row.update(
            {
                "selected_rank": frame.selected.rank,
                "selected_class": frame.selected.proposal_class,
                "selected_conf": frame.selected.confidence,
                "selected_x1": frame.selected.xyxy[0],
                "selected_y1": frame.selected.xyxy[1],
                "selected_x2": frame.selected.xyxy[2],
                "selected_y2": frame.selected.xyxy[3],
            }
        )
    return row


def write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_dict_rows(path, rows)


def write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replace_selected(frame: DecodedFrame, selected: Candidate | None) -> DecodedFrame:
    return DecodedFrame(
        sample_id=frame.sample_id,
        split=frame.split,
        video_id=frame.video_id,
        frame_index=frame.frame_index,
        gt_class=frame.gt_class,
        image_width=frame.image_width,
        image_height=frame.image_height,
        gt_xyxy=frame.gt_xyxy,
        selected=selected,
    )


def parse_xyxy(row: dict[str, str], prefix: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}_x1"]),
        float(row[f"{prefix}_y1"]),
        float(row[f"{prefix}_x2"]),
        float(row[f"{prefix}_y2"]),
    )


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def make_sweep_key(distance_weight: float, size_weight: float, class_switch_weight: float) -> str:
    return f"d{distance_weight:g}_s{size_weight:g}_c{class_switch_weight:g}"


def format_split_summary(split: str, summary: dict[str, Any]) -> str:
    raw = summary["raw_top1"]
    topk = summary["raw_topk_oracle"]
    best = summary["best_temporal"]
    return (
        f"{split} "
        f"raw_top1:r50={raw['recall_iou_0.50']:.4f},c20={raw['center_recall_20px']:.4f} "
        f"topk_oracle:r50={topk['recall_iou_0.50']:.4f},c20={topk['center_recall_20px']:.4f} "
        f"temporal[{summary['best_temporal_key']}]:"
        f"r50={best['recall_iou_0.50']:.4f},c20={best['center_recall_20px']:.4f},"
        f"miou={best['mean_iou']:.4f}"
    )


def resolve_path(path: Path) -> Path:
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
