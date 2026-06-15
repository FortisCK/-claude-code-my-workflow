from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.task2.audit_stage2aa_source_match import build_report, parse_args


def _write_rows(path: Path, sources: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "source"])
        writer.writeheader()
        for index, source in enumerate(sources):
            writer.writerow({"sample_id": f"s{index // 2}", "source": source})


def _args(*values: str):
    return parse_args().parse_args(values)  # type: ignore[attr-defined]


def test_source_match_report_identifies_matching_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    _write_rows(train, ["a", "b", "a"])
    _write_rows(valid, ["b", "a"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--train-candidates",
            str(train),
            "--valid-candidates",
            str(valid),
        ],
    )

    report = build_report(parse_args())

    assert report["source_set_match"] is True
    assert report["common_sources"] == ["a", "b"]
    assert report["train_only_sources"] == []
    assert report["valid_only_sources"] == []


def test_source_match_report_identifies_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = tmp_path / "train.csv"
    valid = tmp_path / "valid.csv"
    _write_rows(train, ["a", "b"])
    _write_rows(valid, ["a", "c"])
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--train-candidates",
            str(train),
            "--valid-candidates",
            str(valid),
        ],
    )

    report = build_report(parse_args())

    assert report["source_set_match"] is False
    assert report["train_only_sources"] == ["b"]
    assert report["valid_only_sources"] == ["c"]

