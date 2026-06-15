from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.task2.filter_candidate_prediction_sources import parse_args, run_filter


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["sample_id", "source", "source_rank", "value"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_filter_keeps_aligned_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = tmp_path / "candidates.csv"
    predictions = tmp_path / "predictions.csv"
    out_candidates = tmp_path / "out_candidates.csv"
    out_predictions = tmp_path / "out_predictions.csv"
    rows = [
        {"sample_id": "s1", "source": "a", "source_rank": "1", "value": "0"},
        {"sample_id": "s1", "source": "b", "source_rank": "2", "value": "1"},
        {"sample_id": "s2", "source": "a", "source_rank": "1", "value": "2"},
    ]
    _write_csv(candidates, rows)
    _write_csv(predictions, rows)
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter",
            "--input-candidates",
            str(candidates),
            "--input-predictions",
            str(predictions),
            "--output-candidates",
            str(out_candidates),
            "--output-predictions",
            str(out_predictions),
            "--keep-source",
            "a",
        ],
    )

    result = run_filter(parse_args())

    assert result["candidate_rows_after"] == 2
    assert [row["source"] for row in _read_csv(out_candidates)] == ["a", "a"]
    assert _read_csv(out_candidates) == _read_csv(out_predictions)


def test_filter_rejects_unaligned_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidates = tmp_path / "candidates.csv"
    predictions = tmp_path / "predictions.csv"
    _write_csv(candidates, [{"sample_id": "s1", "source": "a", "source_rank": "1", "value": "0"}])
    _write_csv(predictions, [{"sample_id": "s1", "source": "b", "source_rank": "1", "value": "0"}])
    monkeypatch.setattr(
        "sys.argv",
        [
            "filter",
            "--input-candidates",
            str(candidates),
            "--input-predictions",
            str(predictions),
            "--output-candidates",
            str(tmp_path / "out_candidates.csv"),
            "--output-predictions",
            str(tmp_path / "out_predictions.csv"),
            "--keep-source",
            "a",
        ],
    )

    with pytest.raises(ValueError, match="source mismatch"):
        run_filter(parse_args())

