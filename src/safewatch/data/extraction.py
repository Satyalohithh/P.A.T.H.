"""Headless per-clip behavioral feature extraction.

Reuses the exact Phase 1-4 runtime chain from ``scripts/demo_features.py``
(VideoFileSource -> YOLO detection -> ByteTrack -> YOLO pose -> FeatureEngine)
but writes :class:`FeatureRecord` streams to disk instead of rendering frames.
Records are cached per clip as Long-format Parquet, so experiments can be
restarted without re-running inference.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from safewatch.core.schemas.features import FeatureRecord

if TYPE_CHECKING:
    from safewatch.core.config import Settings
    from safewatch.features.behavioral_features import FeatureEngine

RECORD_COLUMNS = (
    "video_id",
    "track_id",
    "frame_index",
    "timestamp",
    "pair_track_id",
    "name",
    "value",
)
"""Long-format Parquet schema used for feature-record caching."""


def records_to_frame(records: list[FeatureRecord]) -> pd.DataFrame:
    """Expand FeatureRecords into a Long-format DataFrame (one row per value)."""

    rows: list[dict[str, object]] = []
    for record in records:
        for name, value in zip(record.names, record.values, strict=True):
            rows.append(
                {
                    "video_id": record.video_id,
                    "track_id": record.track_id,
                    "frame_index": record.frame_index,
                    "timestamp": record.timestamp,
                    "pair_track_id": record.pair_track_id,
                    "name": name,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def frame_to_records(frame: pd.DataFrame) -> list[FeatureRecord]:
    """Regroup a Long-format frame back into :class:`FeatureRecord` objects."""

    grouped = frame.sort_values(["track_id", "frame_index"]).groupby(
        ["track_id", "frame_index"], sort=False
    )
    records: list[FeatureRecord] = []
    for (track_id, frame_index), variation in grouped:
        names = tuple(str(name) for name in variation["name"])
        values = tuple(float(value) for value in variation["value"])
        head = variation.iloc[0]
        pair_track_id = _optional_int(head["pair_track_id"])
        video_id = str(head["video_id"]) if not pd.isna(head["video_id"]) else None
        records.append(
            FeatureRecord(
                track_id=int(track_id),
                frame_index=int(frame_index),
                timestamp=float(head["timestamp"]),
                names=names,
                values=values,
                pair_track_id=pair_track_id,
                video_id=video_id,
            )
        )
    return records


def save_records(records: list[FeatureRecord], path: Path) -> None:
    """Persist extracted records as Long-format Parquet (resumable cache)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_frame(records).to_parquet(path, index=False)


def load_records(path: Path) -> list[FeatureRecord]:
    """Load records cached by :meth:`save_records`."""

    return frame_to_records(pd.read_parquet(path))


def extract_clip_features(
    video_path: Path,
    settings: Settings,
    *,
    video_id: str,
    frames_cap: int | None = None,
    engine_factory: Callable[[], FeatureEngine] | None = None,
) -> list[FeatureRecord]:
    """Run real detection/tracking/pose/features over one clip, headless.

    ``engine_factory`` defaults to the standard FeatureEngine (same
    construction as the demo), while tests and smoke runs inject a stub.
    """

    from safewatch.detection import YOLOPersonDetector
    from safewatch.features.behavioral_features import FeatureEngine
    from safewatch.ingestion import VideoFileSource
    from safewatch.pose import YOLOPoseEstimator
    from safewatch.tracking import ByteTrackTracker

    if engine_factory is None:

        def engine_factory() -> FeatureEngine:
            return FeatureEngine(
                keypoint_confidence_threshold=settings.keypoint_confidence_threshold,
                window_frames=settings.interaction_window_frames,
                smoothing_sigma=settings.smoothing_sigma_frames,
                smoothing_kernel_width=settings.smoothing_kernel_width,
            )

    detector = YOLOPersonDetector(settings.detection)
    tracker = ByteTrackTracker(
        settings.tracking,
        person_class_id=settings.detection.person_class,
    )
    pose_estimator = YOLOPoseEstimator(settings.pose)
    feature_engine = engine_factory()

    features: list[FeatureRecord] = []
    source = VideoFileSource(path=str(video_path))
    try:
        source.open()
        detector.warmup()
        tracker.initialize()
        pose_estimator.warmup()
        frames_processed = 0
        while True:
            if frames_cap is not None and frames_processed >= frames_cap:
                break
            result = source.read()
            if result is None:
                break
            metadata, frame = result
            detections = detector.detect(frame, timestamp=metadata.timestamp)
            tracks = tracker.update(detections)
            pose_records = pose_estimator.batch_estimate(
                frame, tracks, timestamp=metadata.timestamp
            )
            for feature_record in feature_engine.process(
                pose_records, frame_index=metadata.frame_index
            ):
                features.append(
                    FeatureRecord(
                        track_id=feature_record.track_id,
                        frame_index=feature_record.frame_index,
                        timestamp=feature_record.timestamp,
                        names=feature_record.names,
                        values=feature_record.values,
                        pair_track_id=feature_record.pair_track_id,
                        video_id=video_id,
                    )
                )
            frames_processed += 1
    finally:
        source.close()
    return features


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(float(value))


__all__ = [
    "RECORD_COLUMNS",
    "extract_clip_features",
    "frame_to_records",
    "load_records",
    "records_to_frame",
    "save_records",
]
