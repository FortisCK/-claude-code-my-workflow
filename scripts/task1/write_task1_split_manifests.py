#!/usr/bin/env python3
"""Write CATHACTION Task 1 split manifests from the local extracted dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from cathaction.data.task1 import Task1Sample, build_task1_index


FIELDNAMES = [
    "split_role",
    "collection",
    "domain",
    "released_split",
    "sample_id",
    "case_id",
    "case_id_source",
    "image_path",
    "mask_path",
    "mask_encoding",
]


def sample_to_row(sample: Task1Sample, repo_root: Path) -> dict[str, str]:
    return {
        "split_role": sample.split_role,
        "collection": sample.collection,
        "domain": sample.domain,
        "released_split": sample.released_split,
        "sample_id": sample.sample_id,
        "case_id": sample.case_id or "",
        "case_id_source": sample.case_id_source,
        "image_path": _repo_relative(sample.image_path, repo_root),
        "mask_path": _repo_relative(sample.mask_path, repo_root),
        "mask_encoding": sample.mask_encoding,
    }


def write_manifests(data_root: Path, output_dir: Path, repo_root: Path) -> dict[str, object]:
    samples = build_task1_index(data_root, require_complete_pairs=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_by_role: dict[str, list[Task1Sample]] = {
        "released_train": [],
        "released_eval": [],
        "human_holdout": [],
    }
    for sample in samples:
        samples_by_role[sample.split_role].append(sample)

    for role, role_samples in samples_by_role.items():
        path = output_dir / f"{role}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            for sample in role_samples:
                writer.writerow(sample_to_row(sample, repo_root))

    summary = {
        "data_root": _repo_relative(data_root.resolve(), repo_root),
        "output_dir": _repo_relative(output_dir.resolve(), repo_root),
        "total_samples": len(samples),
        "roles": {
            role: {
                "count": len(role_samples),
                "collections": dict(sorted(Counter(s.collection for s in role_samples).items())),
                "domains": dict(sorted(Counter(s.domain for s in role_samples).items())),
                "case_id_sources": dict(
                    sorted(Counter(s.case_id_source for s in role_samples).items())
                ),
            }
            for role, role_samples in samples_by_role.items()
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _repo_relative(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/task1/splits"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_manifests(
        data_root=args.data_root,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
