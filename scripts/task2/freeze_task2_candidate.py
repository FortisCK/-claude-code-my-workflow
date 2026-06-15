#!/usr/bin/env python3
"""Parameterized freeze/verify for a Task 2 submission candidate (V / AQ / AI / ...).

One tool instead of a per-candidate verifier (avoids the copy-paste drift of a
3rd verify_stage2v clone). For a candidate's clean prediction directory it:
  - locates the valid_phantom / valid_animal / valid_combined prediction CSVs
    (supports both `{split}_predictions.csv` and `{split}_domain_policy_predictions.csv`);
  - validates the clean inference-only schema (asserts NO gt_* leakage, INV-3/INV-10);
  - counts rows per split / class / domain / distinct video;
  - writes a uniform `candidate_manifest.json` recording provenance + the candidate's
    existing metrics file (metrics are NOT recomputed here to avoid box-convention
    risk; re-evaluation on the official package happens at 2026-07-10).
With --strict it exits non-zero on a missing split CSV or a schema violation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_COLUMNS = [
    "sample_id", "video_id", "frame_index", "domain", "class_id", "score",
    "x1", "y1", "x2", "y2", "source", "source_rank", "score_mode", "policy_name",
]
SPLITS = ("valid_phantom", "valid_animal", "valid_combined")
CSV_PATTERNS = ("{split}_predictions.csv", "{split}_domain_policy_predictions.csv")


def display_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def find_split_csv(prediction_dir: Path, split: str) -> Path | None:
    for pattern in CSV_PATTERNS:
        candidate = prediction_dir / pattern.format(split=split)
        if candidate.is_file():
            return candidate
    return None


def summarize_csv(csv_path: Path) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    leaked = [c for c in header if c.lower().startswith("gt")]
    schema_ok = header == EXPECTED_COLUMNS and not leaked
    by_class = Counter(r["class_id"] for r in rows)
    by_domain = Counter(r["domain"] for r in rows)
    videos = {r["video_id"] for r in rows}
    return {
        "path": display_path(csv_path),
        "rows": len(rows),
        "schema_ok": schema_ok,
        "leaked_columns": leaked,
        "header": header,
        "by_class": dict(sorted(by_class.items())),
        "by_domain": dict(sorted(by_domain.items())),
        "num_videos": len(videos),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Candidate label, e.g. stage2v / stage2aq / stage2ai.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, default=None,
                        help="Existing metrics artifact for this candidate (recorded, not recomputed).")
    parser.add_argument("--role", default="candidate",
                        help="e.g. 'operational_default' (AQ) / 'coco_hedge' (AI) / 'conservative_fallback' (V).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Manifest path (default: <prediction-dir>/candidate_manifest.json).")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction_dir = args.prediction_dir if args.prediction_dir.is_absolute() else (REPO_ROOT / args.prediction_dir)
    if not prediction_dir.is_dir():
        print(f"ERROR: --prediction-dir not found: {prediction_dir}")
        return 2

    splits: dict[str, object] = {}
    missing: list[str] = []
    schema_violations: list[str] = []
    for split in SPLITS:
        csv_path = find_split_csv(prediction_dir, split)
        if csv_path is None:
            missing.append(split)
            continue
        summary = summarize_csv(csv_path)
        splits[split] = summary
        if not summary["schema_ok"]:
            schema_violations.append(split)

    manifest = {
        "artifact_type": "task2_candidate_freeze",
        "name": args.name,
        "role": args.role,
        "prediction_dir": display_path(prediction_dir),
        "metrics_json": display_path(args.metrics_json) if args.metrics_json else None,
        "frozen_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "expected_columns": EXPECTED_COLUMNS,
        "splits": splits,
        "missing_splits": missing,
        "schema_violations": schema_violations,
        "tie_break_record": "quality_reports/decisions/2026-06-15_task2_metric_interpretation.md",
        "note": "Metrics not recomputed here; re-evaluate all candidates on the official "
                "evaluator at 2026-07-10 per the tie-break record.",
    }
    output = args.output or (prediction_dir / "candidate_manifest.json")
    output = output if output.is_absolute() else (REPO_ROOT / output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.strict and (missing or schema_violations):
        print(f"ERROR (--strict): missing={missing} schema_violations={schema_violations}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
