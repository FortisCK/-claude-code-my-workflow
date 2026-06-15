"""Heatmap target helpers for CATHACTION Task 2 localization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from cathaction.data.task2 import Task2Box


@dataclass(frozen=True)
class HeatmapTarget:
    heatmap: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor
    center_index: tuple[int, int]


@dataclass(frozen=True)
class DecodedBox:
    class_id: int
    score: float
    x_center: float
    y_center: float
    width: float
    height: float

    def to_task2_box(self) -> Task2Box:
        return Task2Box(
            class_id=self.class_id,
            x_center=self.x_center,
            y_center=self.y_center,
            width=self.width,
            height=self.height,
        )


def make_heatmap_target(
    box: Task2Box,
    *,
    output_size: tuple[int, int],
    num_classes: int = 2,
    sigma: float = 2.0,
) -> HeatmapTarget:
    """Create class heatmap, size map, and center offset target.

    Coordinates are normalized to the resized network input and then projected
    to the model output grid.
    """

    out_h, out_w = output_size
    if not 0 <= box.class_id < num_classes:
        raise ValueError(f"class_id {box.class_id} outside num_classes={num_classes}")
    cx = clamp(box.x_center * out_w, 0.0, float(out_w - 1))
    cy = clamp(box.y_center * out_h, 0.0, float(out_h - 1))
    ix = int(math.floor(cx))
    iy = int(math.floor(cy))

    heatmap = torch.zeros((num_classes, out_h, out_w), dtype=torch.float32)
    draw_gaussian(heatmap[box.class_id], cx, cy, sigma=sigma)

    size = torch.zeros((2, out_h, out_w), dtype=torch.float32)
    offset = torch.zeros((2, out_h, out_w), dtype=torch.float32)
    size[0, iy, ix] = float(box.width)
    size[1, iy, ix] = float(box.height)
    offset[0, iy, ix] = float(cx - ix)
    offset[1, iy, ix] = float(cy - iy)
    return HeatmapTarget(heatmap=heatmap, size=size, offset=offset, center_index=(iy, ix))


def draw_gaussian(heatmap: torch.Tensor, cx: float, cy: float, *, sigma: float) -> None:
    height, width = heatmap.shape
    radius = max(1, int(math.ceil(3 * sigma)))
    ix = int(math.floor(cx))
    iy = int(math.floor(cy))
    left = max(0, ix - radius)
    right = min(width - 1, ix + radius)
    top = max(0, iy - radius)
    bottom = min(height - 1, iy + radius)
    if right < left or bottom < top:
        return

    y = torch.arange(top, bottom + 1, dtype=torch.float32).view(-1, 1)
    x = torch.arange(left, right + 1, dtype=torch.float32).view(1, -1)
    patch = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    current = heatmap[top : bottom + 1, left : right + 1]
    torch.maximum(current, patch, out=current)


def decode_heatmap_prediction(
    heatmap_logits: torch.Tensor,
    size_map: torch.Tensor,
    offset_map: torch.Tensor | None = None,
) -> DecodedBox:
    """Decode one sample prediction into normalized bbox coordinates."""

    if heatmap_logits.ndim != 3:
        raise ValueError(f"heatmap_logits must be [C,H,W], got shape {tuple(heatmap_logits.shape)}")
    if size_map.shape[0] != 2:
        raise ValueError(f"size_map must have 2 channels, got shape {tuple(size_map.shape)}")
    class_heatmap = torch.sigmoid(heatmap_logits)
    flat_index = int(torch.argmax(class_heatmap).item())
    _, out_h, out_w = class_heatmap.shape
    class_id = flat_index // (out_h * out_w)
    spatial_index = flat_index % (out_h * out_w)
    y_index = spatial_index // out_w
    x_index = spatial_index % out_w
    score = float(class_heatmap[class_id, y_index, x_index].item())

    if offset_map is not None:
        dx = float(offset_map[0, y_index, x_index].item())
        dy = float(offset_map[1, y_index, x_index].item())
    else:
        dx = 0.5
        dy = 0.5
    x_center = clamp((x_index + dx) / out_w, 0.0, 1.0)
    y_center = clamp((y_index + dy) / out_h, 0.0, 1.0)
    width = clamp(float(size_map[0, y_index, x_index].item()), 1e-4, 1.0)
    height = clamp(float(size_map[1, y_index, x_index].item()), 1e-4, 1.0)
    return DecodedBox(
        class_id=int(class_id),
        score=score,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
