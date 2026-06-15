"""Regression tests for Task 2 video-level (case-level) split disjointness (INV-2).

Guards against the frame-level dedup bug in prepare_yolo_splits.py that left 5
phantom video_0 frames in train while video_0 is the entire valid_phantom panel.
"""

from pathlib import Path

import pytest

from scripts.task2.audit_split_disjointness import (
    audit,
    discover_groups,
    video_ids_from_list,
)

DATA_ROOT = Path("datasets/collision_detection")


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def test_video_id_parsing_phantom_and_animal(tmp_path: Path) -> None:
    list_file = tmp_path / "v.txt"
    _write(
        list_file,
        [
            "labels/video_0_00000189.txt",
            "labels/video_12_00004692.txt",
            "labels/video_2_animal_00000022.txt",  # animal keeps the _animal segment
        ],
    )
    assert video_ids_from_list(list_file) == {"video_0", "video_12", "video_2_animal"}


def test_audit_passes_when_video_disjoint(tmp_path: Path) -> None:
    train = tmp_path / "train_labels.txt"
    valid = tmp_path / "valid_labels.txt"
    _write(train, ["labels/video_1_00000001.txt", "labels/video_2_00000002.txt"])
    _write(valid, ["labels/video_0_00000003.txt"])
    assert audit({"train": train}, {"valid": valid}) == 0


def test_audit_fails_on_same_video_leak(tmp_path: Path) -> None:
    # video_0 in both train and valid (different frames) is the exact bug we fixed.
    train = tmp_path / "train_labels.txt"
    valid = tmp_path / "valid_labels.txt"
    _write(train, ["labels/video_0_00000189.txt", "labels/video_1_00000001.txt"])
    _write(valid, ["labels/video_0_00004090.txt"])
    assert audit({"train": train}, {"valid": valid}) == 1


def test_discover_groups_classifies_train_vs_heldout(tmp_path: Path) -> None:
    _write(tmp_path / "train_clean_labels.txt", ["labels/video_1_00000001.txt"])
    _write(tmp_path / "valid_phantom_labels.txt", ["labels/video_0_00000001.txt"])
    _write(tmp_path / "valid_animal_labels.txt", ["labels/video_2_animal_00000001.txt"])
    train, held_out = discover_groups(tmp_path)
    assert set(train) == {"train_clean_labels.txt"}
    assert set(held_out) == {"valid_phantom_labels.txt", "valid_animal_labels.txt"}


@pytest.mark.skipif(
    not (DATA_ROOT / "train_phantom.txt").exists(),
    reason="collision_detection dataset not present",
)
def test_prepare_yolo_splits_is_video_disjoint(tmp_path: Path) -> None:
    from scripts.task2.prepare_yolo_splits import prepare_splits

    summary = prepare_splits(
        data_root=DATA_ROOT.resolve(),
        output_dir=tmp_path / "splits",
        config_dir=tmp_path / "cfg",
        repo_root=Path.cwd(),
        class_names=("normal", "collision"),
    )
    disjoint = summary["video_level_disjointness"]
    # The shipped split left exactly 5 phantom video_0 frames in train.
    assert disjoint["residual_same_video_leak_frames_removed"] == 5
    assert "video_0" in disjoint["train_raw_video_overlap"]
    # After the fix, train_clean shares no video with any validation split.
    assert disjoint["train_clean_video_overlap"] == []

    # And the standalone auditor agrees on the freshly written lists.
    assert audit(
        {"train_clean": tmp_path / "splits" / "train_clean_labels.txt"},
        {
            "valid_phantom": tmp_path / "splits" / "valid_phantom_labels.txt",
            "valid_animal": tmp_path / "splits" / "valid_animal_labels.txt",
        },
    ) == 0
