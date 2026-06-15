#!/usr/bin/env python3
"""Export frozen Stage2AB domain-policy Task2 predictions without ground truth.

Stage2AB was selected on the current public validation split by sweeping
domain-aware score modes. This script is the inference-side equivalent: it
applies the frozen policy to candidate rows and verifier prediction rows, then
writes clean prediction CSVs. It intentionally does not compute metrics and
does not require GT fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from scripts.task2.export_clean_predictions import CLEAN_FIELDS, apply_topk, write_rows
from scripts.task2.verify_stage2ab_domain_policy import EXPECTED_POLICIES


CLASS_SUFFIX = {0: "normal", 1: "collision"}
DEFAULT_SPLITS = ("valid_combined", "valid_phantom", "valid_animal")
VALID_INPUT_DOMAINS = {"phantom", "animal", "human"}
POLICY_DOMAINS = {"phantom", "animal"}
ALIGNMENT_FIELDS = ("sample_id", "source", "source_rank")
FORBIDDEN_INPUT_ONLY_FIELDS = {
    "gt_class",
    "gt_iou",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
    "candidate_iou",
    "candidate_label",
    "verifier_label",
}
IDENTITY_BOX_TRANSFORM = {
    "dx_center": 0.0,
    "dy_center": 0.0,
    "log_w": 0.0,
    "log_h": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing {split}_candidates_used.csv and {split}_eval_prediction_rows.csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Split prefixes to export.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topk-per-sample-class",
        type=int,
        default=0,
        help="Keep only top-k predictions per sample_id/class_id. 0 keeps all rows.",
    )
    parser.add_argument(
        "--fallback-domain",
        choices=sorted(VALID_INPUT_DOMAINS),
        default=None,
        help=(
            "Domain to use only when candidate rows have no domain. By default "
            "missing/unknown domains fail fast."
        ),
    )
    parser.add_argument(
        "--human-policy-domain",
        choices=sorted(POLICY_DOMAINS),
        default="animal",
        help=(
            "Frozen policy domain to use for human rows. The exported domain "
            "stays 'human'; only the score-mode lookup is mapped."
        ),
    )
    parser.add_argument(
        "--infer-domain-from-id",
        action="store_true",
        help="Infer animal/phantom from sample_id or video_id when candidate domain is missing.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/stage2ab_gt_free_export_manifest.json.",
    )
    parser.add_argument(
        "--box-transform-json",
        type=Path,
        default=None,
        help=(
            "Optional Stage2AC transform JSON. Accepts either a JSON object with "
            "a top-level 'transforms' field or the transform mapping itself."
        ),
    )
    parser.add_argument(
        "--score-policy-json",
        type=Path,
        default=None,
        help=(
            "Optional frozen score-policy JSON. Accepts a Stage2 sweep JSON "
            "with best_class_policies or a direct class/domain policy mapping."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_box_transforms(path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    raw = loaded.get("transforms", loaded)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected transforms object")
    transforms: dict[str, dict[str, dict[str, float]]] = {}
    for domain, per_class_raw in raw.items():
        if not isinstance(per_class_raw, dict):
            raise ValueError(f"{path}: transform domain {domain!r} is not an object")
        transforms[str(domain)] = {}
        for class_id, transform_raw in per_class_raw.items():
            if not isinstance(transform_raw, dict):
                raise ValueError(
                    f"{path}: transform for {domain}/{class_id} is not an object"
                )
            transform = dict(IDENTITY_BOX_TRANSFORM)
            for key in IDENTITY_BOX_TRANSFORM:
                if key in transform_raw:
                    transform[key] = float(transform_raw[key])
            transforms[str(domain)][str(int(class_id))] = transform
    return transforms


def load_score_policies(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {class_id: dict(policy) for class_id, policy in EXPECTED_POLICIES.items()}
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected JSON object")
    raw = loaded.get("best_class_policies", loaded)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected best_class_policies object")
    policies: dict[str, dict[str, str]] = {}
    for class_id in ("0", "1"):
        class_raw = raw.get(class_id)
        if not isinstance(class_raw, dict):
            raise ValueError(f"{path}: missing policy for class {class_id}")
        policy_raw = class_raw.get("policy", class_raw)
        if not isinstance(policy_raw, dict):
            raise ValueError(f"{path}: class {class_id} policy is not an object")
        policy = {
            "phantom": str(policy_raw["phantom"]),
            "animal": str(policy_raw["animal"]),
            "default": str(policy_raw.get("default", policy_raw["phantom"])),
        }
        policies[class_id] = policy
    return policies


def _infer_domain_from_text(*values: str) -> str | None:
    text = " ".join(value.lower() for value in values if value)
    if "human" in text:
        return "human"
    if "animal" in text:
        return "animal"
    if "phantom" in text:
        return "phantom"
    return None


def resolve_domain(
    candidate: dict[str, str],
    *,
    fallback_domain: str | None,
    infer_domain_from_id: bool,
    split: str,
    row_index: int,
) -> str:
    domain = candidate.get("domain", "").strip().lower()
    if domain in VALID_INPUT_DOMAINS:
        return domain
    if infer_domain_from_id:
        inferred = _infer_domain_from_text(candidate.get("sample_id", ""), candidate.get("video_id", ""))
        if inferred is not None:
            return inferred
    if fallback_domain is not None:
        return fallback_domain
    raise ValueError(
        f"{split}: candidate row {row_index} has missing/unknown domain {domain!r}; "
        "provide candidate domain metadata or pass --fallback-domain explicitly"
    )


def policy_domain_for_input(domain: str, *, human_policy_domain: str) -> str:
    if domain == "human":
        return human_policy_domain
    if domain in POLICY_DOMAINS:
        return domain
    raise ValueError(f"Cannot select policy for domain {domain!r}")


def apply_box_transform(
    candidate: dict[str, str],
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(candidate[name]) for name in ("x1", "y1", "x2", "y2"))
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    center_x = 0.5 * (x1 + x2) + float(transform["dx_center"]) * width
    center_y = 0.5 * (y1 + y2) + float(transform["dy_center"]) * height
    new_width = width * math.exp(float(transform["log_w"]))
    new_height = height * math.exp(float(transform["log_h"]))
    new_x1 = center_x - 0.5 * new_width
    new_y1 = center_y - 0.5 * new_height
    new_x2 = center_x + 0.5 * new_width
    new_y2 = center_y + 0.5 * new_height
    image_width = float(candidate.get("image_width", 1e9) or 1e9)
    image_height = float(candidate.get("image_height", 1e9) or 1e9)
    new_x1 = max(0.0, min(image_width, new_x1))
    new_y1 = max(0.0, min(image_height, new_y1))
    new_x2 = max(0.0, min(image_width, new_x2))
    new_y2 = max(0.0, min(image_height, new_y2))
    if new_x2 <= new_x1:
        new_x2 = min(image_width, new_x1 + 1.0)
    if new_y2 <= new_y1:
        new_y2 = min(image_height, new_y1 + 1.0)
    return (new_x1, new_y1, new_x2, new_y2)


def validate_alignment(
    candidate: dict[str, str],
    prediction: dict[str, str],
    *,
    split: str,
    row_index: int,
) -> None:
    for field in ALIGNMENT_FIELDS:
        if str(candidate.get(field, "")) != str(prediction.get(field, "")):
            raise ValueError(
                f"{split}: candidate/prediction alignment mismatch at row {row_index} "
                f"for {field}: {candidate.get(field)!r} vs {prediction.get(field)!r}"
            )


def make_prediction_rows(
    candidates: list[dict[str, str]],
    predictions: list[dict[str, str]],
    *,
    split: str,
    fallback_domain: str | None = None,
    infer_domain_from_id: bool = False,
    human_policy_domain: str = "animal",
    box_transforms: dict[str, dict[str, dict[str, float]]] | None = None,
    score_policies: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if len(candidates) != len(predictions):
        raise ValueError(
            f"{split}: row count mismatch: {len(candidates)} candidates vs {len(predictions)} predictions"
        )

    output: list[dict[str, Any]] = []
    for class_id in (0, 1):
        suffix = CLASS_SUFFIX[class_id]
        for row_number, (candidate, prediction) in enumerate(zip(candidates, predictions), start=2):
            validate_alignment(candidate, prediction, split=split, row_index=row_number)
            domain = resolve_domain(
                candidate,
                fallback_domain=fallback_domain,
                infer_domain_from_id=infer_domain_from_id,
                split=split,
                row_index=row_number,
            )
            policy_domain = policy_domain_for_input(domain, human_policy_domain=human_policy_domain)
            class_policy = (score_policies or EXPECTED_POLICIES)[str(class_id)]
            score_mode = class_policy.get(policy_domain, class_policy.get("default", class_policy["phantom"]))
            score_key = f"{score_mode}_score_{suffix}"
            if score_key not in prediction:
                raise KeyError(f"{split}: missing score column {score_key}")
            transform = (
                (box_transforms or {})
                .get(policy_domain, {})
                .get(str(class_id), IDENTITY_BOX_TRANSFORM)
            )
            x1, y1, x2, y2 = apply_box_transform(candidate, transform)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"{split}: invalid box at row {row_number}: {(x1, y1, x2, y2)}")
            policy_name = (
                f"class{class_id}_{domain}"
                if domain == policy_domain
                else f"class{class_id}_{domain}_as_{policy_domain}"
            )
            output.append(
                {
                    "sample_id": candidate["sample_id"],
                    "video_id": candidate.get("video_id", ""),
                    "frame_index": candidate.get("frame_index", ""),
                    "domain": domain,
                    "class_id": class_id,
                    "score": float(prediction[score_key]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source": candidate.get("source", ""),
                    "source_rank": candidate.get("source_rank", ""),
                    "score_mode": score_mode,
                    "policy_name": policy_name,
                }
            )
    return output


def export_split(
    run_dir: Path,
    output_dir: Path,
    split: str,
    *,
    topk_per_sample_class: int = 0,
    fallback_domain: str | None = None,
    infer_domain_from_id: bool = False,
    human_policy_domain: str = "animal",
    box_transforms: dict[str, dict[str, dict[str, float]]] | None = None,
    score_policies: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    candidates_path = run_dir / f"{split}_candidates_used.csv"
    predictions_path = run_dir / f"{split}_eval_prediction_rows.csv"
    candidates = read_csv(candidates_path)
    predictions = read_csv(predictions_path)
    rows = make_prediction_rows(
        candidates,
        predictions,
        split=split,
        fallback_domain=fallback_domain,
        infer_domain_from_id=infer_domain_from_id,
        human_policy_domain=human_policy_domain,
        box_transforms=box_transforms,
        score_policies=score_policies,
    )
    rows = apply_topk(rows, topk_per_sample_class)
    output_path = output_dir / f"{split}_domain_policy_predictions.csv"
    write_rows(output_path, rows)

    domain_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    score_mode_counts: dict[str, int] = {}
    for row in rows:
        domain_counts[str(row["domain"])] = domain_counts.get(str(row["domain"]), 0) + 1
        class_counts[str(row["class_id"])] = class_counts.get(str(row["class_id"]), 0) + 1
        score_mode_counts[str(row["score_mode"])] = score_mode_counts.get(str(row["score_mode"]), 0) + 1
    return {
        "split": split,
        "candidate_csv": candidates_path.as_posix(),
        "prediction_csv": predictions_path.as_posix(),
        "output_csv": output_path.as_posix(),
        "rows": len(rows),
        "class_counts": dict(sorted(class_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "score_mode_counts": dict(sorted(score_mode_counts.items())),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    box_transforms = load_box_transforms(args.box_transform_json)
    score_policies = load_score_policies(args.score_policy_json)
    split_summaries = {
        split: export_split(
            args.run_dir,
            output_dir,
            split,
            topk_per_sample_class=int(args.topk_per_sample_class),
            fallback_domain=args.fallback_domain,
            infer_domain_from_id=bool(args.infer_domain_from_id),
            human_policy_domain=args.human_policy_domain,
            box_transforms=box_transforms,
            score_policies=score_policies,
        )
        for split in args.splits
    }
    manifest_path = args.manifest_json or output_dir / "stage2ab_gt_free_export_manifest.json"
    manifest = {
        "artifact_type": "task2_stage2ab_gt_free_domain_policy_export",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": args.run_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "topk_per_sample_class": int(args.topk_per_sample_class),
        "fallback_domain": args.fallback_domain,
        "infer_domain_from_id": bool(args.infer_domain_from_id),
        "human_policy_domain": args.human_policy_domain,
        "box_transform_json": args.box_transform_json.as_posix()
        if args.box_transform_json is not None
        else None,
        "box_transforms": box_transforms,
        "score_policy_json": args.score_policy_json.as_posix()
        if args.score_policy_json is not None
        else None,
        "clean_fields": list(CLEAN_FIELDS),
        "forbidden_output_fields": sorted(FORBIDDEN_INPUT_ONLY_FIELDS),
        "frozen_policy": score_policies,
        "metrics_available": False,
        "splits": split_summaries,
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": manifest_path.as_posix(), "splits": split_summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
