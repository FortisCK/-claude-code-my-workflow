#!/usr/bin/env python3
"""Evaluate oracle recall over a union of saved Task 2 proposal CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import Task2Sample, repo_relative  # noqa: E402
from cathaction.data.task2_roi import class_counts, load_task2_samples_from_split  # noqa: E402


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class SourceSpec:
    name: str
    proposal_dir: Path


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
        required=True,
        help="Proposal source as name=directory. Directory must contain {split}_proposals.csv files.",
    )
    parser.add_argument("--source-top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/proposal_union_oracle"))
    parser.add_argument("--name", default="proposal_csv_union")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    data_root = resolve_path(args.data_root)
    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = [parse_source_spec(item) for item in args.source]
    split_specs = [
        SplitSpec("valid_combined", resolve_path(args.valid_combined_split)),
        SplitSpec("valid_phantom", resolve_path(args.valid_phantom_split)),
        SplitSpec("valid_animal", resolve_path(args.valid_animal_split)),
    ]
    args_record = {
        **vars(args),
        "repo_root": REPO_ROOT.as_posix(),
        "data_root": repo_relative(data_root, REPO_ROOT),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "sources": [
            {"name": source.name, "proposal_dir": repo_relative(source.proposal_dir, REPO_ROOT)}
            for source in sources
        ],
    }
    (run_dir / "args.json").write_text(json.dumps(json_ready(args_record), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_ready(args_record), indent=2), flush=True)

    all_metrics: dict[str, Any] = {}
    for split_spec in split_specs:
        samples = load_task2_samples_from_split(data_root, split_spec.path)
        rows = load_union_rows(split_spec.name, sources=sources, source_top_k=args.source_top_k)
        metrics = summarize_union(samples, rows)
        all_metrics[split_spec.name] = metrics
        write_rows_csv(run_dir / f"{split_spec.name}_union_rows.csv", rows)
        (run_dir / f"{split_spec.name}_metrics.json").write_text(
            json.dumps(json_ready(metrics), indent=2) + "\n",
            encoding="utf-8",
        )
        print(format_summary(split_spec.name, metrics), flush=True)

    (run_dir / "summary_metrics.json").write_text(
        json.dumps(json_ready(all_metrics), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved proposal-union oracle evaluation to {run_dir}", flush=True)
    return 0


def load_union_rows(split_name: str, *, sources: list[SourceSpec], source_top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        csv_path = source.proposal_dir / f"{split_name}_proposals.csv"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not parse_bool(row["has_proposal"]):
                    continue
                if int(float(row["proposal_rank"])) > source_top_k:
                    continue
                rows.append(
                    {
                        "split": split_name,
                        "sample_id": row["sample_id"],
                        "gt_class": int(float(row["gt_class"])),
                        "source": source.name,
                        "source_rank": int(float(row["proposal_rank"])),
                        "source_conf": proposal_confidence(row),
                        "gt_iou": float(row["proposal_gt_iou"]),
                    }
                )
    return rows


def summarize_union(samples: list[Task2Sample], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    metrics: dict[str, Any] = {
        "samples": len(samples),
        "class_counts": class_counts(samples),
        "rows": len(rows),
        "samples_with_candidate": sum(1 for sample in samples if rows_by_sample.get(sample.sample_id)),
        "oracle": summarize_samples(samples, rows_by_sample),
        "oracle_by_gt_class": {},
        "best_source_counts": best_source_counts(samples, rows_by_sample),
    }
    for class_id in sorted(class_counts(samples)):
        class_samples = [sample for sample in samples if sample.class_id == class_id]
        metrics["oracle_by_gt_class"][str(class_id)] = summarize_samples(class_samples, rows_by_sample)
    return metrics


def summarize_samples(
    samples: list[Task2Sample],
    rows_by_sample: dict[str, list[dict[str, Any]]],
) -> dict[str, float | int]:
    best_ious: list[float] = []
    candidate_counts: list[int] = []
    for sample in samples:
        candidates = rows_by_sample.get(sample.sample_id, [])
        candidate_counts.append(len(candidates))
        best_ious.append(max([float(row["gt_iou"]) for row in candidates], default=0.0))
    summary: dict[str, float | int] = {
        "samples": len(samples),
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
        "median_best_iou": float(np.median(best_ious)) if best_ious else float("nan"),
        "mean_candidates_per_sample": float(np.mean(candidate_counts)) if candidate_counts else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        summary[f"recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return summary


def best_source_counts(samples: list[Task2Sample], rows_by_sample: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        candidates = rows_by_sample.get(sample.sample_id, [])
        if not candidates:
            counts["none"] = counts.get("none", 0) + 1
            continue
        best = max(candidates, key=lambda row: float(row["gt_iou"]))
        source = str(best["source"])
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def parse_source_spec(value: str) -> SourceSpec:
    if "=" not in value:
        raise ValueError(f"--source must be name=directory, got {value!r}")
    name, path_text = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Source name is empty in {value!r}")
    return SourceSpec(name=name, proposal_dir=resolve_path(Path(path_text.strip())))


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def proposal_confidence(row: dict[str, str]) -> float:
    if row.get("proposal_conf") not in (None, ""):
        return float(row["proposal_conf"])
    if row.get("proposal_score") not in (None, ""):
        return float(row["proposal_score"])
    return 0.0


def format_summary(split_name: str, metrics: dict[str, Any]) -> str:
    item = metrics["oracle"]
    parts = [
        split_name,
        f"r50={item['recall_iou_0.50']:.4f}",
        f"r75={item['recall_iou_0.75']:.4f}",
        f"mean_iou={item['mean_best_iou']:.4f}",
        f"best_sources={metrics['best_source_counts']}",
    ]
    return " ".join(parts)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
