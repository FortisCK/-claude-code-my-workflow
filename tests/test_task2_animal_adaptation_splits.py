from __future__ import annotations

from pathlib import Path

from scripts.task2.prepare_animal_adaptation_splits import prepare_animal_adaptation_splits


def test_animal_adaptation_splits_are_video_disjoint(tmp_path: Path) -> None:
    repo_root = Path.cwd()

    summary = prepare_animal_adaptation_splits(
        data_root=repo_root / "datasets/collision_detection",
        base_split_dir=repo_root / "configs/task2/splits",
        output_dir=tmp_path / "splits_adapt_animal",
        config_dir=tmp_path / "configs",
        repo_root=repo_root,
        class_names=("normal", "collision"),
    )

    folds = summary["folds"]
    assert set(folds) == {"train_v1_val_v2", "train_v2_val_v1", "train_v1_v2_val_v0"}
    for fold in folds.values():
        train_videos = set(fold["animal_train_videos"])
        val_videos = set(fold["animal_val_videos"])
        assert train_videos.isdisjoint(val_videos)

    v1_v2 = folds["train_v1_v2_val_v0"]
    assert v1_v2["train"]["class_counts"]["1"] > 0
    assert v1_v2["val"]["class_counts"]["0"] > 0
