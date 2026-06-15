from pathlib import Path

from PIL import Image

from cathaction.data.task2_roi import (
    RoiCropConfig,
    compute_roi_bounds,
    crop_task2_roi,
    load_task2_samples_from_split,
)


def test_roi_crop_uses_square_bounds_and_padding(tmp_path: Path) -> None:
    _write_task2_pair(tmp_path, "video_0_00000001", "1 0.05 0.05 0.1 0.1")
    split = tmp_path / "split.txt"
    split.write_text("labels/video_0_00000001.txt\n", encoding="utf-8")
    sample = load_task2_samples_from_split(tmp_path, split)[0]
    image = Image.open(sample.image_path)

    config = RoiCropConfig(crop_scale=8.0, min_crop_size=64, max_crop_size=64, fill=17)
    bounds = compute_roi_bounds(sample, image.width, image.height, config)
    roi = crop_task2_roi(image, sample, config)

    assert bounds == (-27, -27, 37, 37)
    assert roi.size == (64, 64)
    assert roi.getpixel((0, 0)) == (17, 17, 17)


def test_load_task2_samples_from_split_accepts_image_entries(tmp_path: Path) -> None:
    _write_task2_pair(tmp_path, "video_1_animal_00000022", "0 0.5 0.5 0.2 0.2")
    split = tmp_path / "split.txt"
    split.write_text("images/video_1_animal_00000022.jpg\n", encoding="utf-8")

    samples = load_task2_samples_from_split(tmp_path, split)

    assert len(samples) == 1
    assert samples[0].sample_id == "video_1_animal_00000022"
    assert samples[0].class_id == 0


def _write_task2_pair(root: Path, stem: str, label: str) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(root / "images" / f"{stem}.jpg")
    (root / "labels" / f"{stem}.txt").write_text(f"{label}\n", encoding="utf-8")

