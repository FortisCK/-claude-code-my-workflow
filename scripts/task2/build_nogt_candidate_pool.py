#!/usr/bin/env python3
"""Build Stage2U-compatible candidate CSVs from proposal CSVs without GT.

This is the hidden-test counterpart to ``build_stage2o_candidate_pool.py``.
It combines multiple proposal-source CSVs into one candidate CSV per split, but
does not require labels, GT boxes, or oracle IoU. The output can be consumed by
``infer_stage2u_quality_ranker.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402
from scripts.task2.train_stage2o_candidate_ranker import first_non_empty, resolve_path  # noqa: E402


DEFAULT_SOURCES = (
    "yolo_stage2l=outputs/task2/yolo_proposal_eval/yolo11s_1024_stage2l_combined_bal_e20_stage2l_panels",
    "task1_geometry_rect=outputs/task2/geometry_proposals/task1_stage5_geometry_d3_rect_stage2l_panels",
    "sequence_tip=outputs/task2/sequence_tip_proposals/convnext_tip384_lr1e4_noamp_e3_top20_templates",
    "stage2w_dense=outputs/task2/sequence_tip_proposals/stage2w_tip384_dense_templates_top30",
    "stage2x_class1=outputs/task2/sequence_tip_proposals/stage2x_class1_centernet_dense_templates_top30",
)
DEFAULT_SPLITS = ("valid_combined", "valid_phantom", "valid_animal")
OUTPUT_FIELDS = (
    "split",
    "sample_id",
    "video_id",
    "frame_index",
    "domain",
    "image_path",
    "image_width",
    "image_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "source",
    "source_priority",
    "source_rank",
    "source_conf",
    "source_class",
    "source_subtype",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    proposal_dir: Path
    priority: int


@dataclass(frozen=True)
class SampleMeta:
    image_path: Path
    image_width: int
    image_height: int
    video_id: str
    frame_index: int
    domain: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=None, help="Proposal source as name=directory.")
    parser.add_argument("--split", action="append", default=None, help="Split prefix. Defaults to public-valid panels.")
    parser.add_argument("--source-top-k", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/nogt_candidate_pool"))
    parser.add_argument("--name", default="stage2av_nogt_five_source_top50")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("datasets/collision_detection/images"),
        help="Fallback image directory for proposal rows that lack image_path.",
    )
    parser.add_argument(
        "--fallback-domain",
        default=None,
        help="Domain to use when it cannot be read or inferred from ids.",
    )
    parser.add_argument(
        "--strict-metadata",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fail if image path/size metadata cannot be resolved.",
    )
    return parser.parse_args()


def parse_source(value: str, priority: int) -> SourceSpec:
    if "=" not in value:
        raise ValueError(f"source must use name=directory syntax: {value!r}")
    name, path_text = value.split("=", maxsplit=1)
    name = name.strip()
    path_text = path_text.strip()
    if not name or not path_text:
        raise ValueError(f"source must use name=directory syntax: {value!r}")
    return SourceSpec(name=name, proposal_dir=resolve_path(Path(path_text)), priority=priority)


def truthy(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "none", "nan"}


def parse_float(value: str, default: float | None = None) -> float | None:
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: str, default: int | None = None) -> int | None:
    parsed = parse_float(value, None)
    if parsed is None:
        return default
    return int(parsed)


def infer_video_frame(sample_id: str, row: dict[str, str]) -> tuple[str, int]:
    video_id = first_non_empty(row, "video_id", default="")
    frame_text = first_non_empty(row, "frame_index", default="")
    if video_id and frame_text:
        return video_id, int(float(frame_text))
    if "_" in sample_id:
        prefix, suffix = sample_id.rsplit("_", maxsplit=1)
        try:
            return video_id or prefix, int(float(suffix))
        except ValueError:
            return video_id or prefix, 0
    return video_id or sample_id, 0


def infer_domain(
    sample_id: str,
    video_id: str,
    row: dict[str, str],
    *,
    split: str,
    fallback_domain: str | None,
) -> str:
    domain = first_non_empty(row, "domain", default="").strip().lower()
    if domain:
        return domain
    text = f"{split} {sample_id} {video_id}".lower()
    for candidate in ("phantom", "animal", "human"):
        if candidate in text:
            return candidate
    if fallback_domain is not None:
        return fallback_domain
    return "unknown"


def candidate_image_path(row: dict[str, str], sample_id: str, image_dir: Path) -> Path | None:
    image_text = first_non_empty(row, "image_path", default="").strip()
    if image_text:
        return resolve_path(Path(image_text))
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = resolve_path(image_dir) / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def row_metadata(
    split: str,
    row: dict[str, str],
    *,
    image_dir: Path,
    fallback_domain: str | None,
) -> SampleMeta | None:
    sample_id = first_non_empty(row, "sample_id", default="")
    if not sample_id:
        return None
    image_path = candidate_image_path(row, sample_id, image_dir)
    if image_path is None:
        return None
    width = parse_int(first_non_empty(row, "image_width", default=""), None)
    height = parse_int(first_non_empty(row, "image_height", default=""), None)
    if width is None or height is None:
        width, height = image_size(image_path)
    video_id, frame_index = infer_video_frame(sample_id, row)
    domain = infer_domain(sample_id, video_id, row, split=split, fallback_domain=fallback_domain)
    return SampleMeta(
        image_path=image_path,
        image_width=int(width),
        image_height=int(height),
        video_id=video_id,
        frame_index=frame_index,
        domain=domain,
    )


def proposal_rank(row: dict[str, str]) -> int | None:
    return parse_int(first_non_empty(row, "proposal_rank", "source_rank", "rank", default=""), None)


def proposal_confidence(row: dict[str, str]) -> float:
    return float(
        parse_float(
            first_non_empty(row, "proposal_conf", "proposal_score", "candidate_score", "score", "confidence", default=""),
            0.0,
        )
        or 0.0
    )


def proposal_box(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    if first_non_empty(row, "proposal_x1", default="") != "":
        names = ("proposal_x1", "proposal_y1", "proposal_x2", "proposal_y2")
    elif first_non_empty(row, "candidate_x1", default="") != "":
        names = ("candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2")
    else:
        names = ("x1", "y1", "x2", "y2")
    values = [parse_float(first_non_empty(row, name, default=""), None) for name in names]
    if any(value is None for value in values):
        return None
    x1, y1, x2, y2 = (float(value) for value in values if value is not None)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def clip_box(
    xyxy: tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_source_rows(source: SourceSpec, split: str) -> list[dict[str, str]]:
    return read_csv(source.proposal_dir / f"{split}_proposals.csv")


def collect_metadata(
    source_rows: dict[str, dict[str, list[dict[str, str]]]],
    *,
    image_dir: Path,
    fallback_domain: str | None,
) -> dict[str, SampleMeta]:
    metadata: dict[str, SampleMeta] = {}
    for rows_by_split in source_rows.values():
        for split, rows in rows_by_split.items():
            for row in rows:
                sample_id = first_non_empty(row, "sample_id", default="")
                if not sample_id:
                    continue
                meta = row_metadata(split, row, image_dir=image_dir, fallback_domain=fallback_domain)
                if meta is not None and (sample_id not in metadata or metadata[sample_id].domain == "unknown"):
                    metadata[sample_id] = meta
    return metadata


def make_candidate_rows(
    split: str,
    source: SourceSpec,
    rows: list[dict[str, str]],
    *,
    metadata_by_sample: dict[str, SampleMeta],
    source_top_k: int,
    strict_metadata: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "skipped_no_proposal": 0,
        "skipped_rank": 0,
        "skipped_metadata": 0,
        "skipped_box": 0,
    }
    for row in rows:
        if first_non_empty(row, "has_proposal", default="true") and not truthy(first_non_empty(row, "has_proposal")):
            stats["skipped_no_proposal"] += 1
            continue
        rank = proposal_rank(row)
        if rank is None or rank > source_top_k:
            stats["skipped_rank"] += 1
            continue
        sample_id = first_non_empty(row, "sample_id", default="")
        meta = metadata_by_sample.get(sample_id)
        if meta is None:
            stats["skipped_metadata"] += 1
            if strict_metadata:
                raise ValueError(f"{split}/{source.name}: missing image metadata for sample {sample_id!r}")
            continue
        box = proposal_box(row)
        if box is None:
            stats["skipped_box"] += 1
            continue
        box = clip_box(box, image_width=meta.image_width, image_height=meta.image_height)
        if box is None:
            stats["skipped_box"] += 1
            continue
        x1, y1, x2, y2 = box
        output.append(
            {
                "split": split,
                "sample_id": sample_id,
                "video_id": meta.video_id,
                "frame_index": meta.frame_index,
                "domain": meta.domain,
                "image_path": repo_relative(meta.image_path, REPO_ROOT),
                "image_width": meta.image_width,
                "image_height": meta.image_height,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source": source.name,
                "source_priority": source.priority,
                "source_rank": rank,
                "source_conf": proposal_confidence(row),
                "source_class": first_non_empty(row, "proposal_class", "source_class", default=""),
                "source_subtype": first_non_empty(row, "proposal_source", "source_subtype", default=""),
            }
        )
        stats["kept_rows"] += 1
    return output, stats


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for row in rows:
        source_counts[str(row["source"])] = source_counts.get(str(row["source"]), 0) + 1
        domain_counts[str(row["domain"])] = domain_counts.get(str(row["domain"]), 0) + 1
    return {
        "rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
    }


def main() -> int:
    args = parse_args()
    split_names = args.split if args.split is not None else list(DEFAULT_SPLITS)
    source_values = args.source if args.source is not None else list(DEFAULT_SOURCES)
    sources = [parse_source(value, priority=index) for index, value in enumerate(source_values)]
    run_dir = resolve_path(args.output_dir) / args.name
    source_rows: dict[str, dict[str, list[dict[str, str]]]] = {
        source.name: {split: load_source_rows(source, split) for split in split_names}
        for source in sources
    }
    metadata_by_sample = collect_metadata(
        source_rows,
        image_dir=resolve_path(args.image_dir),
        fallback_domain=args.fallback_domain,
    )

    split_summaries: dict[str, Any] = {}
    for split in split_names:
        split_rows: list[dict[str, Any]] = []
        source_stats: dict[str, Any] = {}
        for source in sources:
            rows, stats = make_candidate_rows(
                split,
                source,
                source_rows[source.name][split],
                metadata_by_sample=metadata_by_sample,
                source_top_k=int(args.source_top_k),
                strict_metadata=bool(args.strict_metadata),
            )
            split_rows.extend(rows)
            source_stats[source.name] = stats
        split_rows.sort(
            key=lambda row: (
                str(row["video_id"]),
                int(row["frame_index"]),
                str(row["sample_id"]),
                int(row["source_priority"]),
                int(row["source_rank"]),
            )
        )
        write_rows(run_dir / f"{split}_candidates.csv", split_rows)
        split_summaries[split] = {
            **summarize_rows(split_rows),
            "path": repo_relative(run_dir / f"{split}_candidates.csv", REPO_ROOT),
            "source_stats": source_stats,
        }

    manifest = {
        "artifact_type": "task2_stage2av_nogt_candidate_pool",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": repo_relative(run_dir, REPO_ROOT),
        "source_top_k": int(args.source_top_k),
        "splits": split_summaries,
        "sources": [
            {
                "name": source.name,
                "proposal_dir": repo_relative(source.proposal_dir, REPO_ROOT),
                "priority": source.priority,
            }
            for source in sources
        ],
        "metadata_samples": len(metadata_by_sample),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
