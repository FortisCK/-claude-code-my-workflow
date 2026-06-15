#!/usr/bin/env python3
"""Append Stage2P refined boxes to Stage2O candidate CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402


DEFAULT_BASE_DIR = Path("outputs/task2/stage2o_candidate_pool/stage2o_valid_yolo_geometry_top50")
DEFAULT_REFINED_DIR = Path("outputs/task2/stage2p_box_refiner/convnext_tiny_yolo_geometry_iou20_full_eval_augmented")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", action="append", default=None, help="Base CSV as split_name=path.")
    parser.add_argument("--refined-csv", action="append", default=None, help="Refined CSV as split_name=path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2q_refined_candidate_pool"))
    parser.add_argument("--name", default="yolo_geometry_stage2p_refined")
    parser.add_argument("--source-name", default="stage2p_refined")
    parser.add_argument("--source-priority", type=int, default=2)
    parser.add_argument("--positive-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--exclude-base", action="store_true", help="Write only refined candidates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_specs = parse_specs(args.base_csv, default_base_specs())
    refined_specs = parse_specs(args.refined_csv, default_refined_specs())
    if set(base_specs) != set(refined_specs):
        raise ValueError(f"Base/refined split names differ: {sorted(base_specs)} vs {sorted(refined_specs)}")

    run_dir = resolve_path(args.output_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "source_name": args.source_name,
        "source_priority": args.source_priority,
        "include_base": not args.exclude_base,
        "splits": {},
    }
    for split_name in sorted(base_specs):
        base_path = resolve_path(base_specs[split_name])
        refined_path = resolve_path(refined_specs[split_name])
        base_rows = read_rows(base_path)
        refined_rows = read_rows(refined_path)
        expanded_rows, split_report = build_expanded_rows(
            base_rows=base_rows,
            refined_rows=refined_rows,
            split_name=split_name,
            source_name=str(args.source_name),
            source_priority=int(args.source_priority),
            positive_iou=float(args.positive_iou),
            background_iou=float(args.background_iou),
            include_base=not args.exclude_base,
        )
        output_path = run_dir / f"{split_name}_candidates.csv"
        write_rows(output_path, expanded_rows, fieldnames=list(base_rows[0].keys()))
        report["splits"][split_name] = {
            **split_report,
            "base_csv": repo_relative(base_path, REPO_ROOT),
            "refined_csv": repo_relative(refined_path, REPO_ROOT),
            "output_csv": repo_relative(output_path, REPO_ROOT),
        }
        print(
            f"{split_name}: base={split_report['base_rows']} refined={split_report['refined_rows']} "
            f"output={split_report['output_rows']} refined_r75={split_report['refined_rows_iou_ge_0.75']}",
            flush=True,
        )

    (run_dir / "summary.json").write_text(json.dumps(json_ready(report), indent=2) + "\n", encoding="utf-8")
    print(f"Saved expanded candidate pool to {run_dir}", flush=True)
    return 0


def build_expanded_rows(
    *,
    base_rows: list[dict[str, str]],
    refined_rows: list[dict[str, str]],
    split_name: str,
    source_name: str,
    source_priority: int,
    positive_iou: float,
    background_iou: float,
    include_base: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_lookup = {base_match_key(row): row for row in base_rows}
    expanded: list[dict[str, Any]] = [dict(row) for row in base_rows] if include_base else []
    missing: list[str] = []
    refined_iou_values: list[float] = []
    for refined in refined_rows:
        key = refined_match_key(refined)
        base = base_lookup.get(key)
        if base is None:
            missing.append(key)
            continue
        row = dict(base)
        candidate_xyxy = (
            float(refined["refined_x1"]),
            float(refined["refined_y1"]),
            float(refined["refined_x2"]),
            float(refined["refined_y2"]),
        )
        width = max(0.0, candidate_xyxy[2] - candidate_xyxy[0])
        height = max(0.0, candidate_xyxy[3] - candidate_xyxy[1])
        candidate_iou = float(refined["after_iou"])
        refined_iou_values.append(candidate_iou)
        verifier_label, verifier_label_name = verifier_label_for_iou(
            candidate_iou,
            gt_class=int(float(base["gt_class"])),
            positive_iou=positive_iou,
            background_iou=background_iou,
        )
        row.update(
            {
                "split": split_name,
                "source": source_name,
                "source_priority": source_priority,
                "source_rank": int(float(base.get("source_rank", 9999))),
                "source_conf": base.get("source_conf", "0"),
                "source_class": base.get("source_class", ""),
                "source_subtype": f"refined_from_{base.get('source', 'candidate')}",
                "candidate_x1": candidate_xyxy[0],
                "candidate_y1": candidate_xyxy[1],
                "candidate_x2": candidate_xyxy[2],
                "candidate_y2": candidate_xyxy[3],
                "candidate_cx": (candidate_xyxy[0] + candidate_xyxy[2]) / 2.0,
                "candidate_cy": (candidate_xyxy[1] + candidate_xyxy[3]) / 2.0,
                "candidate_width": width,
                "candidate_height": height,
                "candidate_iou": candidate_iou,
                "matched_gt": candidate_iou >= positive_iou,
                "verifier_label": verifier_label,
                "verifier_label_name": verifier_label_name,
            }
        )
        expanded.append(row)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} refined rows did not match base rows; first keys: {preview}")
    report = {
        "base_rows": len(base_rows),
        "refined_rows": len(refined_rows),
        "output_rows": len(expanded),
        "refined_rows_iou_ge_0.50": sum(value >= 0.50 for value in refined_iou_values),
        "refined_rows_iou_ge_0.75": sum(value >= 0.75 for value in refined_iou_values),
        "refined_mean_iou": float(sum(refined_iou_values) / len(refined_iou_values)) if refined_iou_values else math.nan,
    }
    return expanded, report


def verifier_label_for_iou(
    candidate_iou: float,
    *,
    gt_class: int,
    positive_iou: float,
    background_iou: float,
) -> tuple[int, str]:
    if candidate_iou >= positive_iou:
        return gt_class + 1, "collision" if gt_class == 1 else "normal"
    if candidate_iou <= background_iou:
        return 0, "background"
    return -1, "ignore"


def base_match_key(row: dict[str, str]) -> str:
    return ":".join(
        [
            row["sample_id"],
            row["source"],
            str(int(float(row["source_rank"]))),
            rounded_coord(row["candidate_x1"]),
            rounded_coord(row["candidate_y1"]),
            rounded_coord(row["candidate_x2"]),
            rounded_coord(row["candidate_y2"]),
        ]
    )


def refined_match_key(row: dict[str, str]) -> str:
    return ":".join(
        [
            row["sample_id"],
            row["source"],
            str(int(float(row["source_rank"]))),
            rounded_coord(row["x1"]),
            rounded_coord(row["y1"]),
            rounded_coord(row["x2"]),
            rounded_coord(row["y2"]),
        ]
    )


def rounded_coord(value: str) -> str:
    return f"{float(value):.4f}"


def default_base_specs() -> dict[str, Path]:
    return {
        "valid_combined": DEFAULT_BASE_DIR / "valid_combined_candidates.csv",
        "valid_phantom": DEFAULT_BASE_DIR / "valid_phantom_candidates.csv",
        "valid_animal": DEFAULT_BASE_DIR / "valid_animal_candidates.csv",
    }


def default_refined_specs() -> dict[str, Path]:
    return {
        "valid_combined": DEFAULT_REFINED_DIR / "valid_combined_eval_refined_candidates.csv",
        "valid_phantom": DEFAULT_REFINED_DIR / "valid_phantom_eval_refined_candidates.csv",
        "valid_animal": DEFAULT_REFINED_DIR / "valid_animal_eval_refined_candidates.csv",
    }


def parse_specs(values: list[str] | None, defaults: dict[str, Path]) -> dict[str, Path]:
    if values is None:
        return defaults
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected split_name=path, got {item!r}")
        name, path_text = item.split("=", 1)
        result[name.strip()] = Path(path_text.strip())
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
