"""Task 2 collision-detection dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


TASK2_IMAGE_SUFFIX = ".jpg"
TASK2_LABEL_SUFFIX = ".txt"


@dataclass(frozen=True)
class Task2Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def xyxy_pixels(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        x1 = (self.x_center - self.width / 2.0) * image_width
        y1 = (self.y_center - self.height / 2.0) * image_height
        x2 = (self.x_center + self.width / 2.0) * image_width
        y2 = (self.y_center + self.height / 2.0) * image_height
        return x1, y1, x2, y2


@dataclass(frozen=True)
class Task2Sample:
    sample_id: str
    video_id: str
    frame_index: int
    image_path: Path
    label_path: Path
    box: Task2Box

    @property
    def class_id(self) -> int:
        return self.box.class_id


def build_task2_index(data_root: Path | str = "datasets/collision_detection") -> list[Task2Sample]:
    root = Path(data_root)
    label_dir = root / "labels"
    image_dir = root / "images"
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Task 2 label directory not found: {label_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Task 2 image directory not found: {image_dir}")

    samples: list[Task2Sample] = []
    for label_path in sorted(label_dir.glob(f"*{TASK2_LABEL_SUFFIX}")):
        image_path = image_dir / f"{label_path.stem}{TASK2_IMAGE_SUFFIX}"
        if not image_path.is_file():
            raise ValueError(f"Missing Task 2 image for label {label_path}: {image_path}")
        video_id, frame_index = parse_task2_sample_stem(label_path.stem)
        samples.append(
            Task2Sample(
                sample_id=label_path.stem,
                video_id=video_id,
                frame_index=frame_index,
                image_path=image_path,
                label_path=label_path,
                box=read_single_task2_box(label_path),
            )
        )
    return samples


def read_single_task2_box(label_path: Path | str) -> Task2Box:
    path = Path(label_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one Task 2 box in {path}, found {len(lines)}")
    return parse_task2_label_line(lines[0], source=path.as_posix())


def parse_task2_label_line(line: str, *, source: str = "<line>") -> Task2Box:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO fields in {source}, found {len(parts)}: {line!r}")
    class_value = float(parts[0])
    class_id = int(class_value)
    if class_value != class_id:
        raise ValueError(f"Task 2 class id must be an integer in {source}: {parts[0]!r}")
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    for name, value in {
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Task 2 YOLO field {name}={value} is outside [0, 1] in {source}")
    return Task2Box(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def parse_task2_sample_stem(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_([0-9]+)", stem)
    if match is None:
        raise ValueError(f"Cannot infer Task 2 video/frame from sample stem: {stem!r}")
    return match.group(1), int(match.group(2))


def read_task2_split_label_names(
    data_root: Path | str, split_file: Path | str
) -> list[str]:
    root = Path(data_root)
    path = Path(split_file)
    if not path.is_absolute():
        candidate = root / path
        path = candidate if candidate.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"Task 2 split file not found: {split_file}")

    label_names: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entry = Path(line)
        if entry.suffix.lower() == TASK2_IMAGE_SUFFIX:
            label_name = f"{entry.stem}{TASK2_LABEL_SUFFIX}"
        elif entry.suffix.lower() == TASK2_LABEL_SUFFIX:
            label_name = entry.name
        else:
            raise ValueError(f"Unsupported Task 2 split entry {line!r} in {path}")
        label_names.append(label_name)
    return label_names


def label_name_to_sample(samples_by_label_name: dict[str, Task2Sample], label_name: str) -> Task2Sample:
    try:
        return samples_by_label_name[label_name]
    except KeyError as exc:
        raise ValueError(f"Task 2 split references unknown label file: {label_name}") from exc


def samples_by_label_name(samples: Iterable[Task2Sample]) -> dict[str, Task2Sample]:
    result: dict[str, Task2Sample] = {}
    for sample in samples:
        key = sample.label_path.name
        if key in result:
            raise ValueError(f"Duplicate Task 2 label name: {key}")
        result[key] = sample
    return result


def repo_relative(path: Path | str, repo_root: Path | str) -> str:
    target = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return target.as_posix()
