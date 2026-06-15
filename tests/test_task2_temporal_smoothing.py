from __future__ import annotations

from scripts.task2.evaluate_temporal_smoothing import (
    Candidate,
    DecodedFrame,
    decode_video,
)


def make_candidate(
    sample_id: str,
    frame_index: int,
    *,
    center_x: float,
    confidence: float,
    rank: int,
) -> Candidate:
    half = 5.0
    return Candidate(
        sample_id=sample_id,
        split="valid",
        video_id="video_0",
        frame_index=frame_index,
        gt_class=0,
        image_width=100,
        image_height=100,
        gt_xyxy=(center_x - half, 45.0, center_x + half, 55.0),
        rank=rank,
        proposal_class=0,
        confidence=confidence,
        xyxy=(center_x - half, 45.0, center_x + half, 55.0),
        source_iou=1.0,
    )


def make_frame(sample_id: str, frame_index: int) -> DecodedFrame:
    return DecodedFrame(
        sample_id=sample_id,
        split="valid",
        video_id="video_0",
        frame_index=frame_index,
        gt_class=0,
        image_width=100,
        image_height=100,
        gt_xyxy=(45.0, 45.0, 55.0, 55.0),
        selected=None,
    )


def test_temporal_decoder_can_prefer_smooth_lower_confidence_path() -> None:
    frames = [make_frame(f"video_0_{index:08d}", index) for index in range(3)]
    candidates = {
        frames[0].sample_id: [
            make_candidate(frames[0].sample_id, 0, center_x=50.0, confidence=0.45, rank=1),
            make_candidate(frames[0].sample_id, 0, center_x=10.0, confidence=0.90, rank=2),
        ],
        frames[1].sample_id: [
            make_candidate(frames[1].sample_id, 1, center_x=51.0, confidence=0.45, rank=1),
            make_candidate(frames[1].sample_id, 1, center_x=90.0, confidence=0.90, rank=2),
        ],
        frames[2].sample_id: [
            make_candidate(frames[2].sample_id, 2, center_x=52.0, confidence=0.45, rank=1),
            make_candidate(frames[2].sample_id, 2, center_x=10.0, confidence=0.90, rank=2),
        ],
    }

    decoded = decode_video(
        video_frames=frames,
        candidates_by_sample=candidates,
        distance_weight=2.0,
        size_weight=0.0,
        class_switch_weight=0.0,
        distance_scale=10.0,
        size_scale=10.0,
        gap_power=1.0,
        max_transition_cost=25.0,
        eps=1e-6,
    )

    assert [round(frame.selected.center[0]) for frame in decoded if frame.selected] == [50, 51, 52]


def test_temporal_decoder_reduces_to_confidence_when_no_transition_penalty() -> None:
    frames = [make_frame(f"video_0_{index:08d}", index) for index in range(2)]
    candidates = {
        frames[0].sample_id: [
            make_candidate(frames[0].sample_id, 0, center_x=50.0, confidence=0.10, rank=1),
            make_candidate(frames[0].sample_id, 0, center_x=10.0, confidence=0.90, rank=2),
        ],
        frames[1].sample_id: [
            make_candidate(frames[1].sample_id, 1, center_x=51.0, confidence=0.10, rank=1),
            make_candidate(frames[1].sample_id, 1, center_x=90.0, confidence=0.90, rank=2),
        ],
    }

    decoded = decode_video(
        video_frames=frames,
        candidates_by_sample=candidates,
        distance_weight=0.0,
        size_weight=0.0,
        class_switch_weight=0.0,
        distance_scale=10.0,
        size_scale=10.0,
        gap_power=1.0,
        max_transition_cost=25.0,
        eps=1e-6,
    )

    assert [round(frame.selected.center[0]) for frame in decoded if frame.selected] == [10, 90]
