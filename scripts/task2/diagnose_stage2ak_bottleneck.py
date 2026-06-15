#!/usr/bin/env python3
"""Attribute Task2 bottlenecks to candidate recall, box geometry, or ranking.

This diagnostic uses public-validation rows with ground truth fields. It is not
an inference/export script and must not be used on hidden test rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_class_ap,
    compute_detection_map,
    iou_xyxy,
)
from scripts.task2.export_stage2ab_domain_policy_predictions import (  # noqa: E402
    IDENTITY_BOX_TRANSFORM,
    apply_box_transform,
    load_box_transforms,
)


DEFAULT_THRESHOLDS = (0.50, 0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing {split}_domain_policy_predictions.csv.",
    )
    parser.add_argument("--splits", nargs="+", default=("valid_combined", "valid_phantom", "valid_animal"))
    parser.add_argument("--box-transform-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="IoU thresholds for recall attribution.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def fnum(value: str | float | int, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def box_from_row(row: dict[str, str], prefix: str = "") -> tuple[float, float, float, float]:
    if prefix == "" and "x1" not in row and "candidate_x1" in row:
        prefix = "candidate_"
    return (
        fnum(row[f"{prefix}x1"]),
        fnum(row[f"{prefix}y1"]),
        fnum(row[f"{prefix}x2"]),
        fnum(row[f"{prefix}y2"]),
    )


def load_ground_truths(candidates: list[dict[str, str]]) -> list[DetectionGroundTruth]:
    seen: set[str] = set()
    gts: list[DetectionGroundTruth] = []
    for row in candidates:
        sample_id = row["sample_id"]
        if sample_id in seen:
            continue
        seen.add(sample_id)
        gts.append(
            DetectionGroundTruth(
                sample_id=sample_id,
                class_id=int(row["gt_class"]),
                box_xyxy=box_from_row(row, "gt_"),
            )
        )
    return gts


def load_clean_predictions(path: Path) -> list[DetectionPrediction]:
    rows = read_csv(path)
    return [
        DetectionPrediction(
            sample_id=row["sample_id"],
            class_id=int(row["class_id"]),
            score=fnum(row["score"]),
            box_xyxy=box_from_row(row),
        )
        for row in rows
    ]


def policy_domain(domain: str) -> str:
    return domain if domain in {"phantom", "animal"} else "animal"


def transformed_candidate_box(
    row: dict[str, str],
    *,
    class_id: int,
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> tuple[float, float, float, float]:
    domain = policy_domain(row.get("domain", "phantom"))
    transform = box_transforms.get(domain, {}).get(str(class_id), IDENTITY_BOX_TRANSFORM)
    return apply_box_transform(row, transform)


def oracle_predictions(
    candidates: list[dict[str, str]],
    *,
    class_id: int,
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[DetectionPrediction]:
    predictions: list[DetectionPrediction] = []
    for row in candidates:
        candidate_box = transformed_candidate_box(row, class_id=class_id, box_transforms=box_transforms)
        if int(row["gt_class"]) == class_id:
            gt_box = box_from_row(row, "gt_")
            score = iou_xyxy(candidate_box, gt_box)
        else:
            score = 0.0
        predictions.append(
            DetectionPrediction(
                sample_id=row["sample_id"],
                class_id=class_id,
                score=score,
                box_xyxy=candidate_box,
            )
        )
    return predictions


def top1_by_source_rank_predictions(
    candidates: list[dict[str, str]],
    *,
    class_id: int,
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[DetectionPrediction]:
    best: dict[str, dict[str, str]] = {}
    for row in candidates:
        sample_id = row["sample_id"]
        old = best.get(sample_id)
        if old is None:
            best[sample_id] = row
            continue
        old_rank = fnum(old.get("source_rank", "999999"), 999999.0)
        new_rank = fnum(row.get("source_rank", "999999"), 999999.0)
        old_conf = fnum(old.get("source_conf", "0"))
        new_conf = fnum(row.get("source_conf", "0"))
        if (new_rank, -new_conf) < (old_rank, -old_conf):
            best[sample_id] = row
    return [
        DetectionPrediction(
            sample_id=row["sample_id"],
            class_id=class_id,
            score=fnum(row.get("source_conf", "0")),
            box_xyxy=transformed_candidate_box(row, class_id=class_id, box_transforms=box_transforms),
        )
        for row in best.values()
    ]


def group_key(row: dict[str, str] | DetectionGroundTruth) -> tuple[str, int]:
    if isinstance(row, DetectionGroundTruth):
        sample_id = row.sample_id
        # Domain is joined later from candidate metadata.
        raise TypeError(sample_id)
    return (row.get("domain", "unknown"), int(row["gt_class"]))


def candidate_recall_rows(
    candidates: list[dict[str, str]],
    *,
    thresholds: list[float],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    gt_rows_by_sample: dict[str, dict[str, str]] = {}
    candidates_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_sample[row["sample_id"]].append(row)
        gt_rows_by_sample.setdefault(row["sample_id"], row)

    rows: list[dict[str, Any]] = []
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample_id, gt_row in gt_rows_by_sample.items():
        class_id = int(gt_row["gt_class"])
        gt_box = box_from_row(gt_row, "gt_")
        candidate_ious = [
            iou_xyxy(
                transformed_candidate_box(candidate, class_id=class_id, box_transforms=box_transforms),
                gt_box,
            )
            for candidate in candidates_by_sample[sample_id]
        ]
        best_iou = max(candidate_ious) if candidate_ious else 0.0
        ranked = sorted(
            candidates_by_sample[sample_id],
            key=lambda item: (fnum(item.get("source_rank", "999999"), 999999.0), -fnum(item.get("source_conf", "0"))),
        )
        top1_iou = (
            iou_xyxy(
                transformed_candidate_box(ranked[0], class_id=class_id, box_transforms=box_transforms),
                gt_box,
            )
            if ranked
            else 0.0
        )
        buckets[(gt_row.get("domain", "unknown"), class_id)].append(
            {
                "sample_id": sample_id,
                "best_iou": best_iou,
                "top1_iou": top1_iou,
                "num_candidates": len(candidates_by_sample[sample_id]),
            }
        )

    for (domain, class_id), items in sorted(buckets.items()):
        best_ious = [float(item["best_iou"]) for item in items]
        top1_ious = [float(item["top1_iou"]) for item in items]
        out: dict[str, Any] = {
            "domain": domain,
            "class_id": class_id,
            "num_gt": len(items),
            "mean_candidates": mean(float(item["num_candidates"]) for item in items),
            "best_iou_mean": mean(best_ious),
            "best_iou_median": median(best_ious),
            "top1_iou_mean": mean(top1_ious),
            "top1_iou_median": median(top1_ious),
        }
        for threshold in thresholds:
            key = f"{threshold:.2f}".replace(".", "")
            out[f"recall_any_iou{key}"] = sum(iou >= threshold for iou in best_ious) / len(best_ious)
            out[f"recall_top1_iou{key}"] = sum(iou >= threshold for iou in top1_ious) / len(top1_ious)
        rows.append(out)
    return rows


def source_recall_rows(
    candidates: list[dict[str, str]],
    *,
    thresholds: list[float],
    box_transforms: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    gt_rows_by_sample: dict[str, dict[str, str]] = {}
    candidates_by_sample_source: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    sources: set[str] = set()
    for row in candidates:
        sample_id = row["sample_id"]
        source = row.get("source", "unknown") or "unknown"
        gt_rows_by_sample.setdefault(sample_id, row)
        candidates_by_sample_source[(sample_id, source)].append(row)
        sources.add(source)

    buckets: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for sample_id, gt_row in gt_rows_by_sample.items():
        class_id = int(gt_row["gt_class"])
        gt_box = box_from_row(gt_row, "gt_")
        for source in sources:
            source_candidates = candidates_by_sample_source.get((sample_id, source), [])
            best_iou = 0.0
            for candidate in source_candidates:
                overlap = iou_xyxy(
                    transformed_candidate_box(candidate, class_id=class_id, box_transforms=box_transforms),
                    gt_box,
                )
                best_iou = max(best_iou, overlap)
            buckets[(gt_row.get("domain", "unknown"), class_id, source)].append(best_iou)

    rows: list[dict[str, Any]] = []
    for (domain, class_id, source), ious in sorted(buckets.items()):
        out: dict[str, Any] = {
            "domain": domain,
            "class_id": class_id,
            "source": source,
            "num_gt": len(ious),
            "best_iou_mean": mean(ious),
            "best_iou_median": median(ious),
        }
        for threshold in thresholds:
            key = f"{threshold:.2f}".replace(".", "")
            out[f"recall_iou{key}"] = sum(iou >= threshold for iou in ious) / len(ious)
        rows.append(out)
    return rows


def ap_summary_rows(
    *,
    name: str,
    ground_truths: list[DetectionGroundTruth],
    predictions: list[DetectionPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id in (0, 1):
        ap50 = compute_class_ap(ground_truths, predictions, class_id=class_id, iou_threshold=0.50)
        ap75 = compute_class_ap(ground_truths, predictions, class_id=class_id, iou_threshold=0.75)
        ap90 = compute_class_ap(ground_truths, predictions, class_id=class_id, iou_threshold=0.90)
        rows.append(
            {
                "variant": name,
                "class_id": class_id,
                "ap50": ap50["ap"],
                "ap75": ap75["ap"],
                "ap90": ap90["ap"],
                "num_gt": ap50["num_gt"],
                "num_predictions": ap50["num_predictions"],
                "tp50": ap50["true_positives"],
                "fp50": ap50["false_positives"],
            }
        )
    return rows


def diagnose_split(
    *,
    split: str,
    run_dir: Path,
    prediction_dir: Path,
    box_transforms: dict[str, dict[str, dict[str, float]]],
    thresholds: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    candidates = read_csv(run_dir / f"{split}_candidates_used.csv")
    ground_truths = load_ground_truths(candidates)
    current_predictions = load_clean_predictions(prediction_dir / f"{split}_domain_policy_predictions.csv")
    oracle_all_predictions = [
        *oracle_predictions(candidates, class_id=0, box_transforms=box_transforms),
        *oracle_predictions(candidates, class_id=1, box_transforms=box_transforms),
    ]
    top1_predictions = [
        *top1_by_source_rank_predictions(candidates, class_id=0, box_transforms=box_transforms),
        *top1_by_source_rank_predictions(candidates, class_id=1, box_transforms=box_transforms),
    ]

    current_map = compute_detection_map(ground_truths, current_predictions, class_ids=(0, 1))
    oracle_map = compute_detection_map(ground_truths, oracle_all_predictions, class_ids=(0, 1))
    top1_map = compute_detection_map(ground_truths, top1_predictions, class_ids=(0, 1))
    recall_rows = candidate_recall_rows(
        candidates, thresholds=thresholds, box_transforms=box_transforms
    )
    source_rows = source_recall_rows(
        candidates, thresholds=thresholds, box_transforms=box_transforms
    )
    ap_rows = [
        *ap_summary_rows(name="current", ground_truths=ground_truths, predictions=current_predictions),
        *ap_summary_rows(name="oracle_iou_score", ground_truths=ground_truths, predictions=oracle_all_predictions),
        *ap_summary_rows(name="top1_source_rank", ground_truths=ground_truths, predictions=top1_predictions),
    ]

    write_csv(
        output_dir / f"{split}_candidate_recall.csv",
        recall_rows,
        fieldnames=list(recall_rows[0]) if recall_rows else [],
    )
    write_csv(
        output_dir / f"{split}_source_recall.csv",
        source_rows,
        fieldnames=list(source_rows[0]) if source_rows else [],
    )
    write_csv(
        output_dir / f"{split}_ap_attribution.csv",
        ap_rows,
        fieldnames=list(ap_rows[0]) if ap_rows else [],
    )
    summary = {
        "split": split,
        "num_candidates": len(candidates),
        "num_ground_truths": len(ground_truths),
        "current": {
            "mAP50": current_map["mAP50"],
            "mAP50-95": current_map["mAP50-95"],
        },
        "oracle_iou_score": {
            "mAP50": oracle_map["mAP50"],
            "mAP50-95": oracle_map["mAP50-95"],
        },
        "top1_source_rank": {
            "mAP50": top1_map["mAP50"],
            "mAP50-95": top1_map["mAP50-95"],
        },
        "candidate_recall": recall_rows,
        "source_recall": source_rows,
        "ap_attribution": ap_rows,
    }
    write_json(output_dir / f"{split}_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    box_transforms = load_box_transforms(args.box_transform_json)
    summaries = {
        split: diagnose_split(
            split=split,
            run_dir=args.run_dir,
            prediction_dir=args.prediction_dir,
            box_transforms=box_transforms,
            thresholds=[float(value) for value in args.thresholds],
            output_dir=args.output_dir,
        )
        for split in args.splits
    }
    manifest = {
        "artifact_type": "task2_stage2ak_bottleneck_attribution",
        "run_dir": args.run_dir.as_posix(),
        "prediction_dir": args.prediction_dir.as_posix(),
        "box_transform_json": args.box_transform_json.as_posix()
        if args.box_transform_json is not None
        else None,
        "thresholds": [float(value) for value in args.thresholds],
        "splits": {
            split: {
                "current": summary["current"],
                "oracle_iou_score": summary["oracle_iou_score"],
                "top1_source_rank": summary["top1_source_rank"],
                "num_candidates": summary["num_candidates"],
                "num_ground_truths": summary["num_ground_truths"],
            }
            for split, summary in summaries.items()
        },
    }
    write_json(args.output_dir / "stage2ak_bottleneck_attribution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
