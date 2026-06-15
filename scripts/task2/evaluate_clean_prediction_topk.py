#!/usr/bin/env python3
"""Evaluate top-k pruning policies on clean Task2 prediction CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import DetectionGroundTruth, DetectionPrediction, compute_detection_map
from scripts.task2.export_clean_predictions import apply_topk
from scripts.task2.export_stage2ab_domain_policy_predictions import DEFAULT_SPLITS


POLICIES = ("all", "global_topk", "animal_topk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--policy", action="append", choices=POLICIES, default=None)
    parser.add_argument("--k", action="append", type=int, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ground_truths_from_candidates(path: Path) -> list[DetectionGroundTruth]:
    candidates = read_csv(path)
    seen: set[str] = set()
    ground_truths: list[DetectionGroundTruth] = []
    for row in candidates:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ground_truths.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(row["gt_class"]),
                box_xyxy=(
                    float(row["gt_x1"]),
                    float(row["gt_y1"]),
                    float(row["gt_x2"]),
                    float(row["gt_y2"]),
                ),
            )
        )
    return ground_truths


def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        parsed: dict[str, Any] = dict(row)
        parsed["class_id"] = int(parsed["class_id"])
        parsed["score"] = float(parsed["score"])
        for key in ("x1", "y1", "x2", "y2"):
            parsed[key] = float(parsed[key])
        rows.append(parsed)
    return rows


def apply_animal_topk(rows: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    if topk <= 0:
        return rows
    kept: list[dict[str, Any]] = []
    animal_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("domain") != "animal":
            kept.append(row)
        else:
            animal_groups.setdefault((str(row["sample_id"]), int(row["class_id"])), []).append(row)
    for key in sorted(animal_groups):
        kept.extend(sorted(animal_groups[key], key=lambda row: float(row["score"]), reverse=True)[:topk])
    return kept


def select_rows(rows: list[dict[str, Any]], *, policy: str, k: int) -> list[dict[str, Any]]:
    if policy == "all":
        return rows
    if policy == "global_topk":
        return apply_topk(rows, k)
    if policy == "animal_topk":
        return apply_animal_topk(rows, k)
    raise ValueError(f"Unknown policy: {policy}")


def to_predictions(rows: list[dict[str, Any]]) -> list[DetectionPrediction]:
    return [
        DetectionPrediction(
            sample_id=str(row["sample_id"]),
            class_id=int(row["class_id"]),
            score=float(row["score"]),
            box_xyxy=(
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
            ),
        )
        for row in rows
    ]


def evaluate_split(
    *,
    run_dir: Path,
    prediction_dir: Path,
    split: str,
    policy: str,
    k: int,
) -> dict[str, Any]:
    ground_truths = ground_truths_from_candidates(run_dir / f"{split}_candidates_used.csv")
    rows = read_prediction_rows(prediction_dir / f"{split}_domain_policy_predictions.csv")
    selected = select_rows(rows, policy=policy, k=k)
    metrics = compute_detection_map(ground_truths, to_predictions(selected), class_ids=(0, 1))
    return {
        "split": split,
        "policy": policy,
        "k": None if policy == "all" else k,
        "rows": len(selected),
        "mAP50": metrics["mAP50"],
        "mAP50-95": metrics["mAP50-95"],
        "class0_ap50": metrics["classes"]["0"]["ap50"],
        "class0_ap50_95": metrics["classes"]["0"]["ap50_95"],
        "class1_ap50": metrics["classes"]["1"]["ap50"],
        "class1_ap50_95": metrics["classes"]["1"]["ap50_95"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    policies = args.policy or list(POLICIES)
    k_values = args.k or [1, 2, 3, 5, 10, 20, 50]
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        for policy in policies:
            policy_k_values = [0] if policy == "all" else k_values
            for k in policy_k_values:
                row = evaluate_split(
                    run_dir=args.run_dir,
                    prediction_dir=args.prediction_dir,
                    split=split,
                    policy=policy,
                    k=k,
                )
                rows.append(row)
                print(
                    f"{split} {policy}{'' if policy == 'all' else f'@{k}'} "
                    f"mAP50={float(row['mAP50']):.6f} "
                    f"mAP50-95={float(row['mAP50-95']):.6f} rows={row['rows']}",
                    flush=True,
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "clean_topk_summary.csv", rows)
    (args.output_dir / "clean_topk_summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

