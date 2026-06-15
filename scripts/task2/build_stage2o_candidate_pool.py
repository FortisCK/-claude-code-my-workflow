#!/usr/bin/env python3
"""Build and audit a unified multi-source candidate pool for Task 2 Stage2O."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402
from cathaction.metrics.detection import iou_xyxy  # noqa: E402


DEFAULT_SOURCES = (
    "yolo_stage2l=outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels",
    "task1_geometry_rect=outputs/task2/geometry_proposals/task1_stage5_geometry_d3_rect_stage2l_panels",
    "sequence_tip=outputs/task2/sequence_tip_proposals/convnext_tip384_lr1e4_noamp_e3_top20_templates",
    "yolov_pilot=outputs/task2/yolov_proposal_eval/yolov_s_576_agnostic_pilot_e3_fullval",
)

VERIFIER_LABEL_NAMES = {
    -1: "ignore",
    0: "background",
    1: "normal",
    2: "collision",
}


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    proposal_dir: Path
    priority: int


@dataclass(frozen=True)
class SampleMeta:
    sample: Task2Sample
    domain: str
    image_width: int
    image_height: int
    gt_xyxy: tuple[float, float, float, float]


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
        "--source",
        action="append",
        default=None,
        help="Proposal source as name=directory. Defaults to the frozen Stage2O sources.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Optional split as name=label_list. If set, overrides the default valid panels.",
    )
    parser.add_argument("--source-top-k", type=int, default=50)
    parser.add_argument("--global-top-k", default="1,5,10,20,50,100,150,200")
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.20)
    parser.add_argument("--baseline-source", default="yolo_stage2l")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2o_candidate_pool"))
    parser.add_argument("--name", default="stage2o_aprime_default_sources_top50")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_root = resolve_path(args.data_root)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    top_k_values = parse_top_k_values(args.global_top_k)
    source_specs = [
        parse_source_spec(item, priority=index)
        for index, item in enumerate(args.source or DEFAULT_SOURCES)
    ]
    split_specs = parse_split_specs(args)

    args_record = {
        **vars(args),
        "repo_root": REPO_ROOT.as_posix(),
        "data_root": repo_relative(data_root, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "sources": [
            {
                "name": source.name,
                "proposal_dir": repo_relative(source.proposal_dir, REPO_ROOT),
                "priority": source.priority,
            }
            for source in source_specs
        ],
        "top_k_values": top_k_values,
        "verifier_label_names": VERIFIER_LABEL_NAMES,
    }
    (run_dir / "args.json").write_text(json.dumps(json_ready(args_record), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(args_record), indent=2), flush=True)

    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        samples = load_task2_samples_from_split(data_root, split_spec.path)
        meta_by_id = build_sample_meta(samples)
        rows, load_report = load_split_candidates(
            split_spec.name,
            meta_by_id=meta_by_id,
            sources=source_specs,
            source_top_k=int(args.source_top_k),
            positive_iou=float(args.positive_iou),
            background_iou=float(args.background_iou),
        )
        metrics = summarize_candidate_pool(
            samples=samples,
            rows=rows,
            sources=source_specs,
            top_k_values=top_k_values,
            baseline_source=str(args.baseline_source),
        )
        metrics["load_report"] = load_report
        all_metrics[split_spec.name] = metrics

        write_rows_csv(run_dir / f"{split_spec.name}_candidates.csv", rows)
        (run_dir / f"{split_spec.name}_audit.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        print(format_split_summary(split_spec.name, metrics), flush=True)

    gates = (
        evaluate_go_gates(all_metrics, baseline_source=str(args.baseline_source))
        if {"valid_combined", "valid_phantom", "valid_animal"}.issubset(all_metrics)
        else {"phase3_ranker_go": None, "reason": "go gates require valid_combined, valid_phantom, and valid_animal"}
    )
    all_metrics["go_gates"] = gates
    (run_dir / "summary_audit.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    split_names = [split.name for split in split_specs]
    write_markdown_report(run_dir / "candidate_pool_audit.md", all_metrics, args_record, split_names=split_names)
    print(format_gate_summary(gates), flush=True)
    print(f"Saved Stage2O candidate pool audit to {run_dir}", flush=True)
    return 0


def load_split_candidates(
    split_name: str,
    *,
    meta_by_id: dict[str, SampleMeta],
    sources: list[SourceSpec],
    source_top_k: int,
    positive_iou: float,
    background_iou: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    load_report: dict[str, Any] = {}
    for source in sources:
        csv_path = source.proposal_dir / f"{split_name}_proposals.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing proposal CSV for source {source.name}: {csv_path}")
        source_rows = load_source_csv(
            csv_path,
            split_name=split_name,
            source=source,
            meta_by_id=meta_by_id,
            source_top_k=source_top_k,
            positive_iou=positive_iou,
            background_iou=background_iou,
        )
        rows.extend(source_rows)
        load_report[source.name] = {
            "csv_path": repo_relative(csv_path, REPO_ROOT),
            "loaded_rows": len(source_rows),
            "unique_samples": len({row["sample_id"] for row in source_rows}),
        }
    rows.sort(key=candidate_sort_key_source_order)
    return rows, load_report


def load_source_csv(
    csv_path: Path,
    *,
    split_name: str,
    source: SourceSpec,
    meta_by_id: dict[str, SampleMeta],
    source_top_k: int,
    positive_iou: float,
    background_iou: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sample_id = str(raw["sample_id"])
            if sample_id not in meta_by_id:
                continue
            if not parse_bool(raw.get("has_proposal", "true")):
                continue
            rank_text = first_non_empty(raw, "proposal_rank", "rank", default="")
            if rank_text == "":
                continue
            source_rank = int(float(rank_text))
            if source_rank > source_top_k:
                continue
            candidate_xyxy = parse_candidate_xyxy(raw)
            if candidate_xyxy is None:
                continue

            meta = meta_by_id[sample_id]
            candidate_xyxy = clip_xyxy(candidate_xyxy, meta.image_width, meta.image_height)
            candidate_iou = parse_float(raw.get("proposal_gt_iou"), default=None)
            if candidate_iou is None:
                candidate_iou = iou_xyxy(candidate_xyxy, meta.gt_xyxy)
            source_conf = proposal_confidence(raw)
            source_class = first_non_empty(raw, "proposal_class", "source_class", default="")
            source_subtype = first_non_empty(raw, "proposal_source", "source", default="")
            verifier_label = verifier_label_from_iou(
                gt_class=meta.sample.class_id,
                candidate_iou=candidate_iou,
                positive_iou=positive_iou,
                background_iou=background_iou,
            )
            rows.append(
                make_candidate_row(
                    split_name=split_name,
                    source=source,
                    source_rank=source_rank,
                    source_conf=source_conf,
                    source_class=source_class,
                    source_subtype=source_subtype,
                    sample_meta=meta,
                    candidate_xyxy=candidate_xyxy,
                    candidate_iou=candidate_iou,
                    verifier_label=verifier_label,
                )
            )
    return rows


def make_candidate_row(
    *,
    split_name: str,
    source: SourceSpec,
    source_rank: int,
    source_conf: float,
    source_class: str,
    source_subtype: str,
    sample_meta: SampleMeta,
    candidate_xyxy: tuple[float, float, float, float],
    candidate_iou: float,
    verifier_label: int,
) -> dict[str, Any]:
    sample = sample_meta.sample
    x1, y1, x2, y2 = candidate_xyxy
    gt_x1, gt_y1, gt_x2, gt_y2 = sample_meta.gt_xyxy
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return {
        "split": split_name,
        "sample_id": sample.sample_id,
        "video_id": sample.video_id,
        "frame_index": sample.frame_index,
        "domain": sample_meta.domain,
        "image_path": repo_relative(sample.image_path, REPO_ROOT),
        "label_path": repo_relative(sample.label_path, REPO_ROOT),
        "image_width": sample_meta.image_width,
        "image_height": sample_meta.image_height,
        "gt_class": sample.class_id,
        "gt_x1": gt_x1,
        "gt_y1": gt_y1,
        "gt_x2": gt_x2,
        "gt_y2": gt_y2,
        "source": source.name,
        "source_priority": source.priority,
        "source_rank": source_rank,
        "source_conf": source_conf,
        "source_class": source_class,
        "source_subtype": source_subtype,
        "candidate_x1": x1,
        "candidate_y1": y1,
        "candidate_x2": x2,
        "candidate_y2": y2,
        "candidate_cx": (x1 + x2) / 2.0,
        "candidate_cy": (y1 + y2) / 2.0,
        "candidate_width": width,
        "candidate_height": height,
        "candidate_iou": candidate_iou,
        "matched_gt": candidate_iou >= 0.50,
        "verifier_label": verifier_label,
        "verifier_label_name": VERIFIER_LABEL_NAMES[verifier_label],
    }


def summarize_candidate_pool(
    *,
    samples: list[Task2Sample],
    rows: list[dict[str, Any]],
    sources: list[SourceSpec],
    top_k_values: list[int],
    baseline_source: str,
) -> dict[str, Any]:
    rows_by_sample = group_rows_by_sample(rows)
    metrics: dict[str, Any] = {
        "samples": len(samples),
        "class_counts": class_counts(samples),
        "domain_class_counts": domain_class_counts(samples),
        "rows": len(rows),
        "samples_with_candidate": sum(1 for sample in samples if rows_by_sample.get(sample.sample_id)),
        "candidate_label_counts": count_values(rows, "verifier_label_name"),
        "source_counts": count_values(rows, "source"),
        "source_good_sample_counts": source_good_sample_counts(samples, rows_by_sample, sources=sources),
        "all_candidates": summarize_samples(samples, rows_by_sample),
        "baseline_source": summarize_samples(
            samples,
            filter_rows_by_sample(rows_by_sample, lambda row: row["source"] == baseline_source),
        ),
        "topk_source_order": summarize_topk(
            samples,
            rows_by_sample,
            top_k_values=top_k_values,
            key_fn=candidate_sort_key_source_order,
        ),
        "topk_raw_conf": summarize_topk(
            samples,
            rows_by_sample,
            top_k_values=top_k_values,
            key_fn=candidate_sort_key_raw_conf,
        ),
        "best_source_counts": best_source_counts(samples, rows_by_sample),
        "by_gt_class": {},
        "by_domain": {},
        "by_domain_gt_class": {},
    }
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        metrics["by_gt_class"][str(class_id)] = summarize_group(
            class_samples,
            rows_by_sample,
            top_k_values=top_k_values,
            baseline_source=baseline_source,
        )
    for domain in sorted({domain_of(sample) for sample in samples}):
        domain_samples = [sample for sample in samples if domain_of(sample) == domain]
        metrics["by_domain"][domain] = summarize_group(
            domain_samples,
            rows_by_sample,
            top_k_values=top_k_values,
            baseline_source=baseline_source,
        )
    for domain in sorted({domain_of(sample) for sample in samples}):
        for class_id in sorted(class_counts(samples)):
            group_samples = [
                sample
                for sample in samples
                if domain_of(sample) == domain and sample.class_id == class_id
            ]
            if group_samples:
                metrics["by_domain_gt_class"][f"{domain}:{class_id}"] = summarize_group(
                    group_samples,
                    rows_by_sample,
                    top_k_values=top_k_values,
                    baseline_source=baseline_source,
                )
    return metrics


def summarize_group(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    top_k_values: list[int],
    baseline_source: str,
) -> dict[str, Any]:
    return {
        "all_candidates": summarize_samples(samples, rows_by_sample),
        "baseline_source": summarize_samples(
            samples,
            filter_rows_by_sample(rows_by_sample, lambda row: row["source"] == baseline_source),
        ),
        "topk_source_order": summarize_topk(
            samples,
            rows_by_sample,
            top_k_values=top_k_values,
            key_fn=candidate_sort_key_source_order,
        ),
        "topk_raw_conf": summarize_topk(
            samples,
            rows_by_sample,
            top_k_values=top_k_values,
            key_fn=candidate_sort_key_raw_conf,
        ),
    }


def summarize_samples(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    best_ious: list[float] = []
    candidate_counts: list[int] = []
    for sample in samples:
        candidates = rows_by_sample.get(sample.sample_id, [])
        candidate_counts.append(len(candidates))
        best_ious.append(max((float(row["candidate_iou"]) for row in candidates), default=0.0))
    summary: dict[str, float | int] = {
        "samples": len(samples),
        "samples_with_candidate": sum(count > 0 for count in candidate_counts),
        "mean_candidates_per_sample": float(np.mean(candidate_counts)) if candidate_counts else float("nan"),
        "median_candidates_per_sample": float(np.median(candidate_counts)) if candidate_counts else float("nan"),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "median_best_iou": float(np.median(best_ious)) if best_ious else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        summary[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return summary


def summarize_topk(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    top_k_values: list[int],
    key_fn: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for top_k in top_k_values:
        clipped: dict[str, list[dict[str, Any]]] = {}
        for sample in samples:
            ordered = sorted(rows_by_sample.get(sample.sample_id, []), key=key_fn)[:top_k]
            clipped[sample.sample_id] = ordered
        metrics[str(top_k)] = summarize_samples(samples, clipped)
    return metrics


def source_good_sample_counts(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
    *,
    sources: list[SourceSpec],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for source in sources:
        source_rows = filter_rows_by_sample(rows_by_sample, lambda row: row["source"] == source.name)
        source_result: dict[str, int] = {}
        for threshold in (0.25, 0.50, 0.75):
            count = 0
            for sample in samples:
                best_iou = max(
                    (float(row["candidate_iou"]) for row in source_rows.get(sample.sample_id, [])),
                    default=0.0,
                )
                if best_iou >= threshold:
                    count += 1
            source_result[f"samples_with_iou_{threshold:.2f}"] = count
        result[source.name] = source_result
    return result


def best_source_counts(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        candidates = rows_by_sample.get(sample.sample_id, [])
        if not candidates:
            counts["none"] = counts.get("none", 0) + 1
            continue
        best = max(candidates, key=lambda row: float(row["candidate_iou"]))
        source = str(best["source"])
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_go_gates(all_metrics: dict[str, Any], *, baseline_source: str) -> dict[str, Any]:
    combined = all_metrics["valid_combined"]
    phantom = all_metrics["valid_phantom"]
    animal = all_metrics["valid_animal"]

    combined_union_r50 = float(combined["all_candidates"]["recall_iou_0.50"])
    combined_base_r50 = float(combined["baseline_source"]["recall_iou_0.50"])
    phantom_union_r50 = float(phantom["all_candidates"]["recall_iou_0.50"])
    phantom_base_r50 = float(phantom["baseline_source"]["recall_iou_0.50"])
    animal_class0_r50 = float(
        animal["by_domain_gt_class"].get("animal:0", {}).get("all_candidates", {}).get("recall_iou_0.50", 0.0)
    )
    gates = {
        "baseline_source": baseline_source,
        "combined_r50_gain_vs_baseline": combined_union_r50 - combined_base_r50,
        "combined_r50_gain_pass": (combined_union_r50 - combined_base_r50) >= 0.02,
        "animal_class0_r50": animal_class0_r50,
        "animal_class0_r50_pass": animal_class0_r50 >= 0.50,
        "phantom_r50_union": phantom_union_r50,
        "phantom_r50_baseline": phantom_base_r50,
        "phantom_no_collapse_pass": phantom_union_r50 + 1e-12 >= phantom_base_r50,
    }
    gates["phase3_ranker_go"] = bool(
        gates["combined_r50_gain_pass"]
        and gates["animal_class0_r50_pass"]
        and gates["phantom_no_collapse_pass"]
    )
    return gates


def write_markdown_report(
    path: Path,
    metrics: dict[str, Any],
    args_record: dict[str, Any],
    *,
    split_names: list[str],
) -> None:
    display_top_k = "50" if 50 in args_record["top_k_values"] else str(max(args_record["top_k_values"]))
    lines: list[str] = [
        "# Stage2O Candidate Pool Audit",
        "",
        f"Run directory: `{args_record['run_dir']}`",
        "",
        "## Sources",
        "",
    ]
    for source in args_record["sources"]:
        lines.append(f"- `{source['name']}`: `{source['proposal_dir']}`")
    lines.extend(["", "## Split Summary", ""])
    lines.append(
        f"| Split | Samples | Rows | Union R@0.50 | Union R@0.75 | Baseline R@0.50 | Source-order Top{display_top_k} R@0.50 | Best sources |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for split_name in split_names:
        split = metrics[split_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    str(split["samples"]),
                    str(split["rows"]),
                    f"{float(split['all_candidates']['recall_iou_0.50']):.4f}",
                    f"{float(split['all_candidates']['recall_iou_0.75']):.4f}",
                    f"{float(split['baseline_source']['recall_iou_0.50']):.4f}",
                    f"{float(split['topk_source_order'][display_top_k]['recall_iou_0.50']):.4f}",
                    "`" + json.dumps(split["best_source_counts"], sort_keys=True) + "`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Domain/Class Recall", ""])
    lines.append(f"| Split | Group | Samples | Union R@0.50 | Union R@0.75 | Baseline R@0.50 | Top{display_top_k} R@0.50 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for split_name in split_names:
        split = metrics[split_name]
        for group_name, group in split["by_domain_gt_class"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        split_name,
                        group_name,
                        str(group["all_candidates"]["samples"]),
                        f"{float(group['all_candidates']['recall_iou_0.50']):.4f}",
                        f"{float(group['all_candidates']['recall_iou_0.75']):.4f}",
                        f"{float(group['baseline_source']['recall_iou_0.50']):.4f}",
                        f"{float(group['topk_source_order'][display_top_k]['recall_iou_0.50']):.4f}",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Go Gates", ""])
    for name, value in metrics["go_gates"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit only measures candidate-pool upper bounds. It does not prove that a learned ranker can recover the oracle candidate.",
            "Proceed to the CSV-driven ROI verifier only if the go gates are acceptable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_sample_meta(samples: list[Task2Sample]) -> dict[str, SampleMeta]:
    result: dict[str, SampleMeta] = {}
    for sample in samples:
        with Image.open(sample.image_path) as image:
            image_width, image_height = image.width, image.height
        result[sample.sample_id] = SampleMeta(
            sample=sample,
            domain=domain_of(sample),
            image_width=image_width,
            image_height=image_height,
            gt_xyxy=tuple(float(value) for value in sample.box.xyxy_pixels(image_width, image_height)),
        )
    return result


def parse_source_spec(value: str, *, priority: int) -> SourceSpec:
    if "=" not in value:
        raise ValueError(f"--source must be name=directory, got {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Source name is empty in {value!r}")
    return SourceSpec(name=name, proposal_dir=resolve_path(Path(path_text.strip())), priority=priority)


def parse_split_specs(args: argparse.Namespace) -> list[SplitSpec]:
    if args.split:
        specs: list[SplitSpec] = []
        for item in args.split:
            if "=" not in item:
                raise ValueError(f"--split must be name=label_list, got {item!r}")
            name, path_text = item.split("=", 1)
            name = name.strip()
            if not name:
                raise ValueError(f"Split name is empty in {item!r}")
            specs.append(SplitSpec(name, resolve_path(Path(path_text.strip()))))
        return specs
    return [
        SplitSpec("valid_combined", resolve_path(args.valid_combined_split)),
        SplitSpec("valid_phantom", resolve_path(args.valid_phantom_split)),
        SplitSpec("valid_animal", resolve_path(args.valid_animal_split)),
    ]


def parse_candidate_xyxy(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    keys = ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")
    if all(row.get(key, "") != "" for key in keys):
        return tuple(float(row[key]) for key in keys)  # type: ignore[return-value]
    keys = ("x1", "y1", "x2", "y2")
    if all(row.get(key, "") != "" for key in keys):
        return tuple(float(row[key]) for key in keys)  # type: ignore[return-value]
    return None


def proposal_confidence(row: dict[str, str]) -> float:
    for key in ("proposal_conf", "proposal_score", "candidate_score", "score", "confidence"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return 0.0


def verifier_label_from_iou(
    *,
    gt_class: int,
    candidate_iou: float,
    positive_iou: float,
    background_iou: float,
) -> int:
    if candidate_iou >= positive_iou:
        return gt_class + 1
    if candidate_iou <= background_iou:
        return 0
    return -1


def group_rows_by_sample(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        result.setdefault(str(row["sample_id"]), []).append(row)
    return result


def filter_rows_by_sample(
    rows_by_sample: dict[str, list[dict[str, Any]]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, list[dict[str, Any]]]:
    return {
        sample_id: [row for row in rows if predicate(row)]
        for sample_id, rows in rows_by_sample.items()
    }


def candidate_sort_key_source_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["source_priority"]),
        int(row["source_rank"]),
        -float(row["source_conf"]),
        str(row["source"]),
    )


def candidate_sort_key_raw_conf(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["source_conf"]),
        int(row["source_priority"]),
        int(row["source_rank"]),
        str(row["source"]),
    )


def domain_of(sample: Task2Sample) -> str:
    lowered = sample.video_id.lower()
    if "animal" in lowered:
        return "animal"
    if "human" in lowered:
        return "human"
    return "phantom"


def domain_class_counts(samples: Iterable[Task2Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        key = f"{domain_of(sample)}:{sample.class_id}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def count_values(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def parse_top_k_values(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"Expected positive comma-separated top-k values, got {value!r}")
    return result


def first_non_empty(row: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def parse_float(value: str | None, *, default: float | None) -> float | None:
    if value in (None, ""):
        return default
    return float(value)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def clip_xyxy(
    xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    clipped = (
        max(0.0, min(float(x1), float(image_width))),
        max(0.0, min(float(y1), float(image_height))),
        max(0.0, min(float(x2), float(image_width))),
        max(0.0, min(float(y2), float(image_height))),
    )
    if clipped[2] < clipped[0] or clipped[3] < clipped[1]:
        return (
            min(clipped[0], clipped[2]),
            min(clipped[1], clipped[3]),
            max(clipped[0], clipped[2]),
            max(clipped[1], clipped[3]),
        )
    return clipped


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "sample_id",
        "video_id",
        "frame_index",
        "domain",
        "image_path",
        "label_path",
        "image_width",
        "image_height",
        "gt_class",
        "gt_x1",
        "gt_y1",
        "gt_x2",
        "gt_y2",
        "source",
        "source_priority",
        "source_rank",
        "source_conf",
        "source_class",
        "source_subtype",
        "candidate_x1",
        "candidate_y1",
        "candidate_x2",
        "candidate_y2",
        "candidate_cx",
        "candidate_cy",
        "candidate_width",
        "candidate_height",
        "candidate_iou",
        "matched_gt",
        "verifier_label",
        "verifier_label_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_split_summary(split_name: str, metrics: dict[str, Any]) -> str:
    all_candidates = metrics["all_candidates"]
    baseline = metrics["baseline_source"]
    topk_key = "50" if "50" in metrics["topk_source_order"] else max(
        metrics["topk_source_order"],
        key=lambda value: int(value),
    )
    topk_item = metrics["topk_source_order"][topk_key]
    return (
        f"{split_name}: samples={metrics['samples']} rows={metrics['rows']} "
        f"union_r50={float(all_candidates['recall_iou_0.50']):.4f} "
        f"union_r75={float(all_candidates['recall_iou_0.75']):.4f} "
        f"baseline_r50={float(baseline['recall_iou_0.50']):.4f} "
        f"source_order_top{topk_key}_r50={float(topk_item['recall_iou_0.50']):.4f} "
        f"best_sources={metrics['best_source_counts']}"
    )


def format_gate_summary(gates: dict[str, Any]) -> str:
    if gates.get("phase3_ranker_go") is None:
        return f"Stage2O Phase3 gate: skipped ({gates.get('reason')})"
    status = "GO" if gates["phase3_ranker_go"] else "NO-GO"
    return (
        f"Stage2O Phase3 gate: {status} "
        f"combined_gain={float(gates['combined_r50_gain_vs_baseline']):.4f} "
        f"animal_class0_r50={float(gates['animal_class0_r50']):.4f} "
        f"phantom_union/base={float(gates['phantom_r50_union']):.4f}/"
        f"{float(gates['phantom_r50_baseline']):.4f}"
    )


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
