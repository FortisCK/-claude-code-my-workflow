"""imagecas_inspect.py — sanity-check ImageCAS layout and one or many volumes.

Usage examples:

    # Inspect one specific case (deep print: shape, HU range, metadata, label):
    python scripts/python/imagecas_inspect.py --case-id 001

    # First 5 cases shallow-print (one line each):
    python scripts/python/imagecas_inspect.py --first 5

    # All cases, write SUMMARY.md aggregate:
    python scripts/python/imagecas_inspect.py --all --output data/imagecas/SUMMARY.md

This script lives in `scripts/python/`, not `code/`, because it is one-off
infrastructure for Week-1-2 Day 3-4 dataset bring-up.  It is NOT a model
entry-point and does NOT need a run-card under `experiments/runs/`.

If `code/data/imagecas_loader.py`'s naming-convention assumptions
(CASE_DIR_GLOB / VOLUME_FNAME / LABEL_FNAME) are wrong for the actual
Kaggle layout, the loader is the place to fix — this script just calls it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# Make `code` importable when running this as a script (not via `python -m`):
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from code.data import imagecas_loader as loader  # noqa: E402


def _format_meta_one_line(case_id: str, summary: dict[str, object]) -> str:
    shape = summary["shape"]
    spacing = summary["spacing"]
    return (
        f"{case_id:<8s} shape={shape}  "
        f"spacing=({spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}) mm  "
        f"HU=[{summary['hu_min']:.0f}, {summary['hu_max']:.0f}] mean={summary['hu_mean']:.0f}"
    )


def inspect_one(case_id: str) -> int:
    try:
        vol = loader.load_volume(case_id)
        lbl = loader.load_label(case_id)
    except FileNotFoundError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    summary = loader.summarize_volume(vol)
    print(f"=== Case {case_id} ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"label present? {lbl is not None}")
    return 0


def inspect_first(n: int) -> int:
    ids = loader.case_ids()
    if not ids:
        print("[FAIL] No cases found.", file=sys.stderr)
        return 1
    for cid in ids[:n]:
        try:
            vol = loader.load_volume(cid)
        except FileNotFoundError as exc:
            print(f"[FAIL] {cid}: {exc}", file=sys.stderr)
            continue
        s = loader.summarize_volume(vol)
        print(_format_meta_one_line(cid, s))
    return 0


def write_summary(output_path: Path) -> int:
    ids = loader.case_ids()
    if not ids:
        print("[FAIL] No cases found.", file=sys.stderr)
        return 1

    print(f"Inspecting {len(ids)} cases ... (this can take a few minutes)")
    rows: list[dict[str, object]] = []
    shapes: Counter[tuple[int, ...]] = Counter()
    labels_present = 0

    for i, cid in enumerate(ids, start=1):
        try:
            vol = loader.load_volume(cid)
        except FileNotFoundError as exc:
            print(f"  skip {cid}: {exc}", file=sys.stderr)
            continue
        s = loader.summarize_volume(vol)
        s["case_id"] = cid
        s["has_label"] = loader.load_label(cid) is not None
        if s["has_label"]:
            labels_present += 1
        rows.append(s)
        shapes[s["shape"]] += 1
        if i % 50 == 0:
            print(f"  ... {i}/{len(ids)}")

    n = len(rows)
    spacings_z = [float(r["spacing"][2]) for r in rows]
    spacings_xy = [float(r["spacing"][0]) for r in rows]
    hu_means = [float(r["hu_mean"]) for r in rows]
    n_voxels = [int(r["n_voxels"]) for r in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write(f"# ImageCAS dataset summary\n\n")
        f.write(f"**Cases inspected:** {n}\n")
        f.write(f"**Cases with segmentation label:** {labels_present} ({100 * labels_present / n:.1f}%)\n\n")
        f.write("## Shape distribution\n\n")
        f.write("| Shape (z, y, x) | Count |\n")
        f.write("| --- | --- |\n")
        for shape, count in sorted(shapes.items(), key=lambda x: -x[1]):
            f.write(f"| {shape} | {count} |\n")
        f.write("\n## Voxel spacing distribution (mm)\n\n")
        f.write(f"- z (slice thickness): min={min(spacings_z):.3f}, max={max(spacings_z):.3f}, "
                f"median={statistics.median(spacings_z):.3f}\n")
        f.write(f"- xy (in-plane): min={min(spacings_xy):.3f}, max={max(spacings_xy):.3f}, "
                f"median={statistics.median(spacings_xy):.3f}\n\n")
        f.write("## HU statistics\n\n")
        f.write(f"- Mean HU across cases: min={min(hu_means):.0f}, max={max(hu_means):.0f}, "
                f"median={statistics.median(hu_means):.0f}\n\n")
        f.write("## Volume size\n\n")
        f.write(f"- Voxel count: min={min(n_voxels):,}, max={max(n_voxels):,}, "
                f"median={int(statistics.median(n_voxels)):,}\n\n")
        f.write("## Per-case detail\n\n")
        f.write("| Case | Shape | Spacing (mm) | HU range | Has label |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in rows:
            sp = r["spacing"]
            f.write(
                f"| {r['case_id']} | {r['shape']} "
                f"| ({sp[0]:.2f},{sp[1]:.2f},{sp[2]:.2f}) "
                f"| [{r['hu_min']:.0f},{r['hu_max']:.0f}] mean={r['hu_mean']:.0f} "
                f"| {'yes' if r['has_label'] else 'no'} |\n"
            )

    print(f"Wrote summary: {output_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--case-id", type=str, help="Inspect one specific case in detail.")
    g.add_argument("--first", type=int, metavar="N", help="Inspect first N cases (one-line each).")
    g.add_argument("--all", action="store_true", help="Inspect all cases; with --output, write SUMMARY.md.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/imagecas/SUMMARY.md"),
        help="Path for SUMMARY.md (default: data/imagecas/SUMMARY.md). Used only with --all.",
    )
    args = p.parse_args()

    if args.case_id is not None:
        return inspect_one(args.case_id)
    if args.first is not None:
        return inspect_first(args.first)
    if args.all:
        return write_summary(args.output)
    return 1


if __name__ == "__main__":
    sys.exit(main())
