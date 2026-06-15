#!/usr/bin/env python3
"""Train a tabular per-frame candidate selector for Task 2 Stage2T."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())
if SRC_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SRC_ROOT.as_posix())

from cathaction.data.task2 import repo_relative  # noqa: E402
from cathaction.metrics.detection import (  # noqa: E402
    DetectionGroundTruth,
    DetectionPrediction,
    compute_detection_map,
)


SCORE_MODES = (
    "roi",
    "bg_suppressed_roi",
    "source_roi",
    "sqrt_source_roi",
    "rank_decay_roi",
    "source_rank_decay_roi",
)
QUALITY_MODES = ("pred_iou", "prob_iou50", "prob_iou75", "blend")
RANK_KEYS = ("source_conf", "p_max_class", *[f"{mode}_max" for mode in SCORE_MODES])
TARGET_COLUMNS = {
    "gt_iou",
    "candidate_iou",
    "gt_class",
    "verifier_label",
    "gt_x1",
    "gt_y1",
    "gt_x2",
    "gt_y2",
}


@dataclass
class CandidatePredictionRecord:
    split: str
    sample_id: str
    video_id: str
    frame_index: int
    domain: str
    source: str
    source_rank: int
    source_priority: int
    source_conf: float
    row_index: int
    image_width: int
    image_height: int
    gt_class: int
    gt_xyxy: tuple[float, float, float, float]
    xyxy: tuple[float, float, float, float]
    gt_iou: float
    prob_background: float
    prob_normal: float
    prob_collision: float
    scores: dict[str, float]
    rank_features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBuilder:
    numeric_names: tuple[str, ...]
    source_categories: tuple[str, ...]
    domain_categories: tuple[str, ...]
    include_domain: bool

    @classmethod
    def fit(
        cls,
        records: list[CandidatePredictionRecord],
        *,
        include_domain: bool,
    ) -> "FeatureBuilder":
        numeric_names = tuple(build_numeric_feature_names())
        if TARGET_COLUMNS & set(numeric_names):
            raise ValueError(f"Target columns leaked into features: {TARGET_COLUMNS & set(numeric_names)}")
        source_categories = tuple(sorted({record.source for record in records}))
        domain_categories = tuple(sorted({record.domain for record in records})) if include_domain else tuple()
        return cls(
            numeric_names=numeric_names,
            source_categories=source_categories,
            domain_categories=domain_categories,
            include_domain=include_domain,
        )

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_names)
        names.extend(f"source={value}" for value in self.source_categories)
        if self.include_domain:
            names.extend(f"domain={value}" for value in self.domain_categories)
        return names

    def transform(self, records: list[CandidatePredictionRecord]) -> np.ndarray:
        matrix = np.zeros((len(records), len(self.feature_names)), dtype=np.float32)
        source_offset = len(self.numeric_names)
        domain_offset = source_offset + len(self.source_categories)
        source_index = {value: idx for idx, value in enumerate(self.source_categories)}
        domain_index = {value: idx for idx, value in enumerate(self.domain_categories)}
        for row_idx, record in enumerate(records):
            for col_idx, name in enumerate(self.numeric_names):
                matrix[row_idx, col_idx] = float(numeric_feature_value(record, name))
            source_col = source_index.get(record.source)
            if source_col is not None:
                matrix[row_idx, source_offset + source_col] = 1.0
            if self.include_domain:
                domain_col = domain_index.get(record.domain)
                if domain_col is not None:
                    matrix[row_idx, domain_offset + domain_col] = 1.0
        return np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)


class ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((x.shape[0],), self.value, dtype=np.float32)


class ConstantClassifier:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p1 = np.full((x.shape[0],), self.probability, dtype=np.float32)
        return np.stack([1.0 - p1, p1], axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-candidate-csv",
        type=Path,
        default=Path(
            "outputs/task2/stage2o_ranker/"
            "convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_candidates_used.csv"
        ),
    )
    parser.add_argument(
        "--train-prediction-csv",
        type=Path,
        default=Path(
            "outputs/task2/stage2o_ranker/"
            "convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval/train_eval_prediction_rows.csv"
        ),
    )
    parser.add_argument(
        "--valid-run-dir",
        type=Path,
        default=Path("outputs/task2/stage2o_ranker/convnext_tiny_stage2o_yolo_geometry_train_subset_full_eval"),
    )
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task2/stage2t_tabular_selector"))
    parser.add_argument("--name", default="histgb_stage2o_trainpred_fullvalid")
    parser.add_argument("--include-domain-feature", default=False, action="store_true")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--l2-regularization", type=float, default=0.02)
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--score-mode", action="append", default=None, choices=SCORE_MODES)
    parser.add_argument("--quality-mode", action="append", default=None, choices=QUALITY_MODES)
    parser.add_argument("--k", action="append", type=int, default=None, help="0 means keep all candidates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    train_candidate_csv = resolve_path(args.train_candidate_csv)
    train_prediction_csv = resolve_path(args.train_prediction_csv)
    valid_run_dir = resolve_path(args.valid_run_dir)
    splits = args.split or ["valid_combined", "valid_phantom", "valid_animal"]
    score_modes = tuple(args.score_mode or SCORE_MODES)
    quality_modes = tuple(args.quality_mode or QUALITY_MODES)
    k_values = tuple(args.k or [0, 1, 2, 3, 5, 10, 20, 50])

    train_records = load_aligned_records(
        candidate_path=train_candidate_csv,
        prediction_path=train_prediction_csv,
        split_name="train",
    )
    add_rank_features(train_records)
    feature_builder = FeatureBuilder.fit(train_records, include_domain=bool(args.include_domain_feature))
    x_train = feature_builder.transform(train_records)
    y_iou = np.asarray([record.gt_iou for record in train_records], dtype=np.float32)
    models = train_quality_models(x_train, y_iou, args=args)

    valid_records_by_split: dict[str, list[CandidatePredictionRecord]] = {}
    summary_rows: list[dict[str, Any]] = []
    best_prediction_rows: dict[str, list[dict[str, Any]]] = {}
    best_policy: dict[str, Any] | None = None

    for split in splits:
        records = load_aligned_records(
            candidate_path=valid_run_dir / f"{split}_candidates_used.csv",
            prediction_path=valid_run_dir / f"{split}_eval_prediction_rows.csv",
            split_name=split,
        )
        add_rank_features(records)
        valid_records_by_split[split] = records
        x_valid = feature_builder.transform(records)
        quality_scores = predict_quality_scores(models, x_valid)
        split_rows, selected_by_policy = evaluate_split(
            records,
            quality_scores=quality_scores,
            split_name=split,
            score_modes=score_modes,
            quality_modes=quality_modes,
            k_values=k_values,
        )
        summary_rows.extend(split_rows)
        if split == "valid_combined":
            best_row = max(split_rows, key=lambda row: float(row["mAP50_95"]))
            best_policy = {
                "quality_mode": best_row["quality_mode"],
                "score_mode": best_row["score_mode"],
                "k": int(best_row["k"]),
                "mAP50": float(best_row["mAP50"]),
                "mAP50_95": float(best_row["mAP50_95"]),
            }
            best_key = policy_key(best_row["quality_mode"], best_row["score_mode"], int(best_row["k"]))
            best_prediction_rows[split] = selected_by_policy[best_key]

    if best_policy is not None:
        for split, records in valid_records_by_split.items():
            predictions = best_prediction_rows.get(split)
            if predictions is None:
                x_valid = feature_builder.transform(records)
                quality_scores = predict_quality_scores(models, x_valid)
                predictions = build_prediction_rows(
                    records,
                    quality_scores[str(best_policy["quality_mode"])],
                    score_mode=str(best_policy["score_mode"]),
                    k=int(best_policy["k"]),
                )
                best_prediction_rows[split] = predictions
            write_csv(output_dir / f"{split}_best_policy_predictions.csv", predictions)

    write_csv(output_dir / "stage2t_summary.csv", summary_rows)
    write_json(
        output_dir / "stage2t_summary.json",
        {
            "rows": summary_rows,
            "best_valid_combined_policy": best_policy,
            "train_rows": len(train_records),
            "feature_count": len(feature_builder.feature_names),
            "feature_names": feature_builder.feature_names,
        },
    )
    write_markdown(output_dir / "stage2t_report.md", summary_rows, best_policy=best_policy)
    write_json(
        output_dir / "args.json",
        {
            **vars(args),
            "train_candidate_csv": repo_relative(train_candidate_csv, REPO_ROOT),
            "train_prediction_csv": repo_relative(train_prediction_csv, REPO_ROOT),
            "valid_run_dir": repo_relative(valid_run_dir, REPO_ROOT),
            "output_dir": repo_relative(output_dir, REPO_ROOT),
        },
    )
    with (output_dir / "models.pkl").open("wb") as handle:
        pickle.dump({"models": models, "feature_builder": feature_builder}, handle)

    print(f"Train rows: {len(train_records)}")
    print(f"Feature count: {len(feature_builder.feature_names)}")
    if best_policy is not None:
        print(
            "Best valid_combined: "
            f"quality={best_policy['quality_mode']} "
            f"score={best_policy['score_mode']} "
            f"k={best_policy['k']} "
            f"mAP50={best_policy['mAP50']:.4f} "
            f"mAP50-95={best_policy['mAP50_95']:.4f}"
        )
    print(f"Saved Stage2T results to {output_dir}")
    return 0


def load_aligned_records(
    *,
    candidate_path: Path,
    prediction_path: Path,
    split_name: str,
) -> list[CandidatePredictionRecord]:
    candidate_rows = read_csv(candidate_path)
    prediction_rows = read_csv(prediction_path)
    if len(candidate_rows) != len(prediction_rows):
        raise ValueError(
            f"{split_name}: candidate/prediction row count mismatch: "
            f"{len(candidate_rows)} vs {len(prediction_rows)}"
        )
    records: list[CandidatePredictionRecord] = []
    for row_index, (candidate, prediction) in enumerate(zip(candidate_rows, prediction_rows)):
        assert_aligned(split_name, row_index, candidate, prediction)
        sample_id = first_non_empty(candidate, "sample_id")
        video_id = first_non_empty(candidate, "video_id", default=sample_id.rsplit("_", 1)[0])
        frame_index = int(float(first_non_empty(candidate, "frame_index", default=sample_id.rsplit("_", 1)[-1])))
        scores: dict[str, float] = {}
        for mode in SCORE_MODES:
            scores[f"{mode}_score_normal"] = float(first_non_empty(prediction, f"{mode}_score_normal"))
            scores[f"{mode}_score_collision"] = float(first_non_empty(prediction, f"{mode}_score_collision"))
        records.append(
            CandidatePredictionRecord(
                split=first_non_empty(candidate, "split", default=split_name),
                sample_id=sample_id,
                video_id=video_id,
                frame_index=frame_index,
                domain=first_non_empty(candidate, "domain", default=domain_from_video_id(video_id)),
                source=first_non_empty(candidate, "source", default="proposal"),
                source_rank=int(float(first_non_empty(candidate, "source_rank", "rank", default="9999"))),
                source_priority=int(float(first_non_empty(candidate, "source_priority", default="0"))),
                source_conf=float(first_non_empty(candidate, "source_conf", "det_conf", default="0")),
                row_index=row_index,
                image_width=int(float(first_non_empty(candidate, "image_width"))),
                image_height=int(float(first_non_empty(candidate, "image_height"))),
                gt_class=int(float(first_non_empty(candidate, "gt_class"))),
                gt_xyxy=parse_xyxy(candidate, "gt_x1", "gt_y1", "gt_x2", "gt_y2"),
                xyxy=parse_candidate_xyxy(candidate),
                gt_iou=float(first_non_empty(candidate, "gt_iou", "candidate_iou", default="0")),
                prob_background=float(first_non_empty(prediction, "prob_background")),
                prob_normal=float(first_non_empty(prediction, "prob_normal")),
                prob_collision=float(first_non_empty(prediction, "prob_collision")),
                scores=scores,
            )
        )
    return records


def assert_aligned(
    split_name: str,
    row_index: int,
    candidate: dict[str, str],
    prediction: dict[str, str],
) -> None:
    checks = [
        ("sample_id", first_non_empty(candidate, "sample_id"), first_non_empty(prediction, "sample_id")),
        ("source", first_non_empty(candidate, "source"), first_non_empty(prediction, "source")),
        (
            "source_rank",
            str(int(float(first_non_empty(candidate, "source_rank", default="9999")))),
            str(int(float(first_non_empty(prediction, "source_rank", default="9999")))),
        ),
    ]
    for name, left, right in checks:
        if left != right:
            raise ValueError(f"{split_name} row {row_index}: {name} mismatch: {left!r} vs {right!r}")
    candidate_iou = float(first_non_empty(candidate, "gt_iou", "candidate_iou", default="0"))
    prediction_iou = float(first_non_empty(prediction, "gt_iou", default=str(candidate_iou)))
    if abs(candidate_iou - prediction_iou) > 1e-6:
        raise ValueError(f"{split_name} row {row_index}: gt_iou mismatch: {candidate_iou} vs {prediction_iou}")


def add_rank_features(records: list[CandidatePredictionRecord]) -> None:
    grouped: dict[str, list[tuple[int, CandidatePredictionRecord]]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record.sample_id, []).append((index, record))
    for group in grouped.values():
        group_size = len(group)
        for key in RANK_KEYS:
            ordered = sorted(group, key=lambda item: (-rank_source_value(item[1], key), item[0]))
            for rank, (_index, record) in enumerate(ordered, start=1):
                record.rank_features[f"{key}_rank"] = float(rank)
                record.rank_features[f"{key}_inv_rank"] = 1.0 / math.sqrt(float(rank))
                record.rank_features[f"{key}_rank_frac"] = float(rank) / max(float(group_size), 1.0)


def rank_source_value(record: CandidatePredictionRecord, key: str) -> float:
    if key == "source_conf":
        return record.source_conf
    if key == "p_max_class":
        return max(record.prob_normal, record.prob_collision)
    if key.endswith("_max"):
        mode = key.removesuffix("_max")
        return max(record.scores[f"{mode}_score_normal"], record.scores[f"{mode}_score_collision"])
    raise ValueError(f"Unknown rank key: {key}")


def build_numeric_feature_names() -> list[str]:
    names = [
        "source_conf",
        "source_rank_log",
        "source_priority",
        "image_width_log",
        "image_height_log",
        "box_cx_rel",
        "box_cy_rel",
        "box_w_rel",
        "box_h_rel",
        "box_area_rel",
        "box_log_aspect",
        "prob_background",
        "prob_normal",
        "prob_collision",
        "prob_foreground",
        "p_max_class",
        "p_class_margin",
        "p_normal_minus_collision",
    ]
    for mode in SCORE_MODES:
        names.extend(
            [
                f"{mode}_score_normal",
                f"{mode}_score_collision",
                f"{mode}_score_max",
                f"{mode}_score_margin",
            ]
        )
    for key in RANK_KEYS:
        names.extend([f"{key}_rank", f"{key}_inv_rank", f"{key}_rank_frac"])
    return names


def numeric_feature_value(record: CandidatePredictionRecord, name: str) -> float:
    x1, y1, x2, y2 = record.xyxy
    width = max(float(record.image_width), 1.0)
    height = max(float(record.image_height), 1.0)
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if name == "source_conf":
        return record.source_conf
    if name == "source_rank_log":
        return math.log1p(max(float(record.source_rank), 0.0))
    if name == "source_priority":
        return float(record.source_priority)
    if name == "image_width_log":
        return math.log1p(width)
    if name == "image_height_log":
        return math.log1p(height)
    if name == "box_cx_rel":
        return ((x1 + x2) * 0.5) / width
    if name == "box_cy_rel":
        return ((y1 + y2) * 0.5) / height
    if name == "box_w_rel":
        return box_w / width
    if name == "box_h_rel":
        return box_h / height
    if name == "box_area_rel":
        return (box_w * box_h) / max(width * height, 1.0)
    if name == "box_log_aspect":
        return math.log((box_w + 1.0) / (box_h + 1.0))
    if name == "prob_background":
        return record.prob_background
    if name == "prob_normal":
        return record.prob_normal
    if name == "prob_collision":
        return record.prob_collision
    if name == "prob_foreground":
        return max(0.0, 1.0 - record.prob_background)
    if name == "p_max_class":
        return max(record.prob_normal, record.prob_collision)
    if name == "p_class_margin":
        return abs(record.prob_normal - record.prob_collision)
    if name == "p_normal_minus_collision":
        return record.prob_normal - record.prob_collision
    if name.endswith("_score_max"):
        mode = name.removesuffix("_score_max")
        return max(record.scores[f"{mode}_score_normal"], record.scores[f"{mode}_score_collision"])
    if name.endswith("_score_margin"):
        mode = name.removesuffix("_score_margin")
        return abs(record.scores[f"{mode}_score_normal"] - record.scores[f"{mode}_score_collision"])
    if name in record.scores:
        return record.scores[name]
    if name in record.rank_features:
        return record.rank_features[name]
    raise KeyError(f"Unknown feature name: {name}")


def train_quality_models(x_train: np.ndarray, y_iou: np.ndarray, *, args: argparse.Namespace) -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("Stage2T requires scikit-learn for the tabular selector") from exc

    y_iou = np.clip(y_iou, 0.0, 1.0)
    regressor = HistGradientBoostingRegressor(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        l2_regularization=float(args.l2_regularization),
        random_state=int(args.random_state),
        loss="squared_error",
    )
    regressor.fit(x_train, y_iou)
    models: dict[str, Any] = {"reg_iou": regressor}
    for threshold, name in [(0.50, "clf_iou50"), (0.75, "clf_iou75")]:
        labels = (y_iou >= threshold).astype(np.int64)
        if len(np.unique(labels)) < 2:
            models[name] = ConstantClassifier(float(np.mean(labels)))
        else:
            classifier = HistGradientBoostingClassifier(
                max_iter=int(args.max_iter),
                learning_rate=float(args.learning_rate),
                l2_regularization=float(args.l2_regularization),
                random_state=int(args.random_state),
                loss="log_loss",
            )
            classifier.fit(x_train, labels)
            models[name] = classifier
    return models


def predict_quality_scores(models: dict[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    pred_iou = np.clip(np.asarray(models["reg_iou"].predict(x), dtype=np.float32), 0.0, 1.0)
    prob_iou50 = positive_probability(models["clf_iou50"], x)
    prob_iou75 = positive_probability(models["clf_iou75"], x)
    blend = np.clip(0.50 * pred_iou + 0.30 * prob_iou50 + 0.20 * prob_iou75, 0.0, 1.0)
    return {
        "pred_iou": pred_iou,
        "prob_iou50": prob_iou50,
        "prob_iou75": prob_iou75,
        "blend": blend,
    }


def positive_probability(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if probabilities.shape[1] == 1:
        return np.zeros((x.shape[0],), dtype=np.float32)
    return np.clip(np.asarray(probabilities[:, 1], dtype=np.float32), 0.0, 1.0)


def evaluate_split(
    records: list[CandidatePredictionRecord],
    *,
    quality_scores: dict[str, np.ndarray],
    split_name: str,
    score_modes: Iterable[str],
    quality_modes: Iterable[str],
    k_values: Iterable[int],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    ground_truths = build_ground_truths(records)
    rows: list[dict[str, Any]] = []
    selected_by_policy: dict[str, list[dict[str, Any]]] = {}
    for quality_mode in quality_modes:
        quality = quality_scores[quality_mode]
        for score_mode in score_modes:
            for k in k_values:
                prediction_rows = build_prediction_rows(records, quality, score_mode=score_mode, k=int(k))
                metrics = compute_detection_map(
                    ground_truths,
                    [
                        DetectionPrediction(
                            row["sample_id"],
                            int(row["class_id"]),
                            float(row["score"]),
                            (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                        )
                        for row in prediction_rows
                    ],
                    class_ids=(0, 1),
                )
                loc = summarize_selected_localization(records, prediction_rows)
                row = {
                    "split": split_name,
                    "quality_mode": quality_mode,
                    "score_mode": score_mode,
                    "k": int(k),
                    "mAP50": metrics["mAP50"],
                    "mAP50_95": metrics["mAP50-95"],
                    "class0_ap50": metrics["classes"]["0"]["ap50"],
                    "class0_ap50_95": metrics["classes"]["0"]["ap50_95"],
                    "class1_ap50": metrics["classes"]["1"]["ap50"],
                    "class1_ap50_95": metrics["classes"]["1"]["ap50_95"],
                    "num_predictions": len(prediction_rows),
                    **loc,
                }
                rows.append(row)
                selected_by_policy[policy_key(quality_mode, score_mode, int(k))] = prediction_rows
                print(
                    f"{split_name} {quality_mode} {score_mode} "
                    f"{'all' if int(k) == 0 else f'top{k}'} "
                    f"mAP50={float(row['mAP50']):.4f} "
                    f"mAP50-95={float(row['mAP50_95']):.4f} "
                    f"locR75={float(row['loc_recall_iou_0.75']):.4f}",
                    flush=True,
                )
    return rows, selected_by_policy


def build_prediction_rows(
    records: list[CandidatePredictionRecord],
    quality: np.ndarray,
    *,
    score_mode: str,
    k: int,
) -> list[dict[str, Any]]:
    selected = select_candidate_indices(records, quality, k=k)
    rows: list[dict[str, Any]] = []
    for index in selected:
        record = records[index]
        q = float(np.clip(quality[index], 0.0, 1.0))
        for class_id, class_name in [(0, "normal"), (1, "collision")]:
            base_score = float(record.scores[f"{score_mode}_score_{class_name}"])
            score = q * base_score
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "class_id": class_id,
                    "score": score,
                    "x1": record.xyxy[0],
                    "y1": record.xyxy[1],
                    "x2": record.xyxy[2],
                    "y2": record.xyxy[3],
                    "candidate_row_index": record.row_index,
                    "selector_quality": q,
                    "score_mode": score_mode,
                    "source": record.source,
                    "source_rank": record.source_rank,
                    "source_conf": record.source_conf,
                    "gt_iou": record.gt_iou,
                }
            )
    rows.sort(key=lambda row: (row["sample_id"], int(row["class_id"]), -float(row["score"])))
    return rows


def select_candidate_indices(records: list[CandidatePredictionRecord], quality: np.ndarray, *, k: int) -> list[int]:
    if k == 0:
        return list(range(len(records)))
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    grouped: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        grouped.setdefault(record.sample_id, []).append(index)
    selected: list[int] = []
    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        selected.extend(
            sorted(
                group,
                key=lambda index: (
                    -float(quality[index]),
                    records[index].source_priority,
                    records[index].source_rank,
                    records[index].source,
                    records[index].row_index,
                ),
            )[:k]
        )
    return selected


def build_ground_truths(records: list[CandidatePredictionRecord]) -> list[DetectionGroundTruth]:
    by_sample: dict[str, CandidatePredictionRecord] = {}
    for record in records:
        previous = by_sample.get(record.sample_id)
        if previous is None:
            by_sample[record.sample_id] = record
        elif previous.gt_class != record.gt_class or previous.gt_xyxy != record.gt_xyxy:
            raise ValueError(f"Inconsistent GT metadata for sample {record.sample_id}")
    return [
        DetectionGroundTruth(record.sample_id, record.gt_class, record.gt_xyxy)
        for record in sorted(by_sample.values(), key=lambda item: item.sample_id)
    ]


def summarize_selected_localization(
    records: list[CandidatePredictionRecord],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, float | int]:
    sample_ids = sorted({record.sample_id for record in records})
    record_by_index = {record.row_index: record for record in records}
    selected_indices = {
        int(row["candidate_row_index"])
        for row in prediction_rows
    }
    retained_by_sample: dict[str, list[CandidatePredictionRecord]] = {sample_id: [] for sample_id in sample_ids}
    for row_index in selected_indices:
        record = record_by_index[row_index]
        retained_by_sample[record.sample_id].append(record)
    best_ious = [
        max((record.gt_iou for record in retained_by_sample[sample_id]), default=0.0)
        for sample_id in sample_ids
    ]
    result: dict[str, float | int] = {
        "retained_candidates": len(selected_indices),
        "mean_retained_candidates_per_sample": len(selected_indices) / max(len(sample_ids), 1),
        "loc_mean_best_iou": float(np.mean(best_ious)) if best_ious else float("nan"),
    }
    for threshold in (0.25, 0.50, 0.75):
        result[f"loc_recall_iou_{threshold:.2f}"] = (
            float(np.mean([iou >= threshold for iou in best_ious])) if best_ious else float("nan")
        )
    return result


def write_markdown(path: Path, rows: list[dict[str, Any]], *, best_policy: dict[str, Any] | None) -> None:
    combined_rows = [row for row in rows if row["split"] == "valid_combined"]
    best_rows = sorted(combined_rows, key=lambda row: float(row["mAP50_95"]), reverse=True)[:20]
    lines = [
        "# Task2 Stage2T Tabular Selector",
        "",
        "## Best valid_combined Policies",
        "",
        "| Rank | Quality | Base score | k | mAP50 | mAP50-95 | loc R@0.75 | predictions |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(best_rows, start=1):
        lines.append(
            f"| {rank} | {row['quality_mode']} | {row['score_mode']} | {row['k']} | "
            f"{float(row['mAP50']):.4f} | {float(row['mAP50_95']):.4f} | "
            f"{float(row['loc_recall_iou_0.75']):.4f} | {int(row['num_predictions'])} |"
        )
    if best_policy is not None:
        lines.extend(
            [
                "",
                "## Selected Policy",
                "",
                f"- quality mode: `{best_policy['quality_mode']}`",
                f"- score mode: `{best_policy['score_mode']}`",
                f"- k: `{best_policy['k']}`",
                f"- valid_combined mAP50: `{float(best_policy['mAP50']):.4f}`",
                f"- valid_combined mAP50-95: `{float(best_policy['mAP50_95']):.4f}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def policy_key(quality_mode: str, score_mode: str, k: int) -> str:
    return f"{quality_mode}::{score_mode}::{k}"


def parse_candidate_xyxy(raw: dict[str, str]) -> tuple[float, float, float, float]:
    if first_non_empty(raw, "x1", default="") != "":
        return parse_xyxy(raw, "x1", "y1", "x2", "y2")
    return parse_xyxy(raw, "candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2")


def parse_xyxy(raw: dict[str, str], x1: str, y1: str, x2: str, y2: str) -> tuple[float, float, float, float]:
    return (
        float(first_non_empty(raw, x1)),
        float(first_non_empty(raw, y1)),
        float(first_non_empty(raw, x2)),
        float(first_non_empty(raw, y2)),
    )


def domain_from_video_id(video_id: str) -> str:
    return "animal" if "animal" in video_id.lower() else "phantom"


def first_non_empty(raw: dict[str, str], *keys: str, default: str | None = None) -> str:
    for key in keys:
        value = raw.get(key, "")
        if value != "":
            return value
    if default is not None:
        return default
    raise KeyError(f"None of the requested keys are present/non-empty: {keys}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(json_ready(row) for row in rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return repo_relative(value, REPO_ROOT)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
