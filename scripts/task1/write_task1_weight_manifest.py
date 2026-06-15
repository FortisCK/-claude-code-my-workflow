#!/usr/bin/env python3
"""Write a sha256 checksum manifest for the frozen Task 1 champion checkpoints.

Mirrors scripts/task2/write_task2_weight_manifest.py. The weight list is the
single source of truth defined in run_task1_submission_inference.CHAMPION_MODELS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.task1.run_task1_submission_inference import REQUIRED_WEIGHT_FILES  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "quality_reports/decisions/2026-06-15_task1_weight_manifest.json"


def display_path(value: str | Path) -> str:
    path = Path(value)
    abs_path = path if path.is_absolute() else (REPO_ROOT / path)
    try:
        return abs_path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return abs_path.resolve().as_posix()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_record(path_text: str) -> dict[str, object]:
    abs_path = (REPO_ROOT / path_text) if not Path(path_text).is_absolute() else Path(path_text)
    if not abs_path.is_file():
        raise FileNotFoundError(f"Task 1 weight file missing: {path_text}")
    stat = abs_path.stat()
    return {
        "path": display_path(path_text),
        "size_bytes": stat.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
        "sha256": sha256_file(abs_path),
    }


def build_manifest() -> dict[str, object]:
    return {
        "artifact_type": "task1_weight_checksum_manifest",
        "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "weights": [weight_record(path) for path in REQUIRED_WEIGHT_FILES],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    output = args.output if args.output.is_absolute() else (REPO_ROOT / args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
