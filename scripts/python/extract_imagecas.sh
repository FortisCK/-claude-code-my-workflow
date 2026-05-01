#!/bin/bash
# extract_imagecas.sh — Extract ImageCAS multi-part split-zip downloads.
#
# Two-layer wrapping (discovered 2026-04-30 LTSI session):
#   Outer: Kaggle CLI 1.7.x auto-wraps each download in a .zip (whatever the
#          inner extension). E.g. `kaggle datasets download -f 1-200.z01`
#          produces a file `1-200.z01` whose CONTENTS are a single-file zip
#          with the actual `1-200.z01` inside.
#   Inner: 5-part split-zip per batch:
#          `1-200.change2zip` (last part with central directory, was originally
#                              `1-200.zip`, renamed by uploader to dodge Kaggle
#                              auto-decomp)
#          `1-200.z01-04`     (first 4 parts, 4.3 GB each)
#
# Per-batch extraction:
#   1. Unwrap all 5 Kaggle outer-zips into a temp dir.
#   2. Rename `1-200.change2zip` → `1-200.zip` (so unzip recognises last part).
#   3. Run `unzip 1-200.zip` — auto-detects multi-part across .z01-.z04.
#   4. Move extracted NIfTIs to `data/imagecas/raw/<batch>/`.
#   5. Delete temp + (optional) the staging zips.
#
# Usage:
#   bash scripts/python/extract_imagecas.sh [--keep-staging]
#
# Designed to run AFTER the download script (download_imagecas.sh equivalent)
# has finished. Idempotent — skips batches that are already extracted.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGING="$REPO_ROOT/data/imagecas/raw/zip-staging"
RAW="$REPO_ROOT/data/imagecas/raw"

KEEP_STAGING=0
for arg in "$@"; do
  case "$arg" in
    --keep-staging) KEEP_STAGING=1 ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if [ ! -d "$STAGING" ]; then
  echo "ERROR: staging dir not found: $STAGING"
  exit 1
fi

BATCHES=("1-200" "201-400" "401-600" "601-800" "801-1000")

echo "== ImageCAS extraction $(date) =="
echo "Staging: $STAGING"
echo "Output:  $RAW/<batch>/"
echo

for batch in "${BATCHES[@]}"; do
  outdir="$RAW/$batch"
  if [ -d "$outdir" ] && [ "$(ls -A "$outdir" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[skip] $batch — already extracted at $outdir"
    continue
  fi

  # Verify all 5 staging files present
  missing=0
  for ext in change2zip z01 z02 z03 z04; do
    if [ ! -f "$STAGING/$batch.$ext" ]; then
      echo "[skip] $batch — missing staging file: $batch.$ext"
      missing=1
      break
    fi
  done
  [ "$missing" -eq 1 ] && continue

  echo "[extract] $batch"
  tmp="$STAGING/_unwrap_$batch"
  rm -rf "$tmp"
  mkdir -p "$tmp"

  # Step 1: unwrap each Kaggle outer-zip (5 files)
  for ext in change2zip z01 z02 z03 z04; do
    echo "  unwrap $batch.$ext"
    unzip -q -o "$STAGING/$batch.$ext" -d "$tmp" || { echo "FAIL"; exit 1; }
  done

  # Step 2: rebuild the multi-part split-zip into a single complete archive.
  # PKZIP split format: z01..z04 are parts; the last part (originally `.zip`,
  # uploader renamed it to `.change2zip` to dodge Kaggle auto-decomp) holds the
  # central directory. Naive `unzip` fails at part boundaries with "bad zipfile
  # offset (lseek)". Naive `cat z01..z04 change2zip > all.zip` also fails — the
  # central directory still references multi-disk addressing ("End-of-centdir-64
  # signature not where expected"). The robust fix is `zip -FF`, which rebuilds
  # the central directory in-place to point at the unified archive.
  echo "  rebuilding $batch.zip via zip -FF"
  mv "$tmp/$batch.change2zip" "$tmp/$batch.zip"
  # NOTE: do NOT pass -q to zip -FF — silent mode causes it to emit a
  # 22-byte empty zip in batch contexts (verified 2026-04-30).
  ( cd "$tmp" && zip -FF "$batch.zip" --output "${batch}_fixed.zip" ) \
    || { echo "FAIL: zip -FF $batch"; exit 1; }

  # Free originals immediately to save disk during the unzip step
  rm "$tmp/$batch.zip" "$tmp/$batch.z01" "$tmp/$batch.z02" \
     "$tmp/$batch.z03" "$tmp/$batch.z04"

  # Step 3: extract the rebuilt archive
  echo "  unzip ${batch}_fixed.zip"
  mkdir -p "$outdir"
  ( cd "$tmp" && unzip -q -o "${batch}_fixed.zip" -d "$outdir" ) \
    || { echo "FAIL: unzip $batch"; exit 1; }
  rm "$tmp/${batch}_fixed.zip"

  # The archive contains a top-level dir named "<batch>" (e.g. "1-200/").
  # Flatten: move contents up so $outdir contains *.nii.gz directly.
  inner="$outdir/$batch"
  if [ -d "$inner" ]; then
    mv "$inner"/* "$outdir/"
    rmdir "$inner"
  fi

  # Step 4: cleanup temp
  rm -rf "$tmp"

  n=$(ls "$outdir"/*.img.nii.gz 2>/dev/null | wc -l)
  echo "  → $n image volumes in $outdir"
done

# Optional: drop the staging zips
if [ "$KEEP_STAGING" -eq 0 ]; then
  echo
  echo "== Cleaning up zip-staging (use --keep-staging to retain) =="
  rm -rf "$STAGING"
  echo "Removed $STAGING"
fi

echo
echo "== Done $(date) =="
echo "== Final disk =="
df -h "$REPO_ROOT" | tail -1
echo "== Per-batch counts =="
for batch in "${BATCHES[@]}"; do
  d="$RAW/$batch"
  if [ -d "$d" ]; then
    n_img=$(ls "$d"/*.img.nii.gz 2>/dev/null | wc -l)
    n_label=$(ls "$d"/*.label.nii.gz 2>/dev/null | wc -l)
    sz=$(du -sh "$d" | awk '{print $1}')
    echo "  $batch: $n_img images / $n_label labels / $sz"
  fi
done
