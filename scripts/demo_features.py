#!/usr/bin/env python3
"""Real-time behavioral feature extraction over tracked persons.

Feeds a video file or webcam through YOLO person detection, ByteTrack, YOLO
pose estimation, and the Phase 4 FeatureEngine, showing the extracted
per-frame records (A.3 torso lean, B.1 limb angular velocities, C.1-C.3
interaction, D.6 reciprocity) and pair overlays in an OpenCV window until 'q'
is pressed or the stream ends.

Usage:
    uv run python scripts/demo_features.py --source sample.mp4
    uv run python scripts/demo_features.py --source 0 --device cpu
    uv run python scripts/demo_features.py --source sample.mp4 --benchmark --frames 300
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from safewatch.core.config import Settings, load_settings
from safewatch.core.schemas.features import FeatureRecord
from safewatch.core.schemas.pose import PoseRecord
from safewatch.detection import YOLOPersonDetector
from safewatch.features.behavioral_features import FeatureEngine
from safewatch.ingestion import VideoFileSource, WebcamSource
from safewatch.ingestion.interfaces import FrameSource
from safewatch.pose import YOLOPoseEstimator
from safewatch.tracking import ByteTrackTracker
from safewatch.utils.annotation import (
    FrameAnnotator,
    InteractionAnnotator,
    PoseAnnotator,
)
from safewatch.utils.logging import configure_logging, get_logger


def _build_source(source: str) -> FrameSource:
    if source.isdigit():
        return WebcamSource(device_index=int(source))
    return VideoFileSource(path=source)


def _load_runtime_config(
    config_path: str,
    conf: float | None,
    device: str | None,
    pose_conf: float | None,
) -> Settings:
    overrides: dict[str, object] = {}
    detection_overrides: dict[str, object] = {}
    if conf is not None:
        detection_overrides["conf_threshold"] = conf
    if device is not None:
        detection_overrides["device"] = device
    if detection_overrides:
        overrides["detection"] = detection_overrides
    pose_overrides: dict[str, object] = {}
    if pose_conf is not None:
        pose_overrides["conf_threshold"] = pose_conf
    if pose_overrides:
        overrides["pose"] = pose_overrides
    return load_settings(config_path=config_path, overrides=overrides)


def _feature_engine(settings: Settings) -> FeatureEngine:
    return FeatureEngine(
        keypoint_confidence_threshold=settings.keypoint_confidence_threshold,
        window_frames=settings.interaction_window_frames,
        smoothing_sigma=settings.smoothing_sigma_frames,
        smoothing_kernel_width=settings.smoothing_kernel_width,
    )


def run(
    settings: Settings,
    source: str,
    display: bool,
    num_frames: int | None,
    benchmark: bool,
) -> None:
    detector = YOLOPersonDetector(settings.detection)
    tracker = ByteTrackTracker(
        settings.tracking,
        person_class_id=settings.detection.person_class,
    )
    pose_estimator = YOLOPoseEstimator(settings.pose)
    feature_engine = _feature_engine(settings)
    track_annotator = FrameAnnotator()
    pose_annotator = PoseAnnotator()
    pair_annotator = InteractionAnnotator()
    benchmark_logger = get_logger("safewatch.benchmark")
    source_obj = _build_source(source)

    detection_times: list[float] = []
    tracking_times: list[float] = []
    pose_times: list[float] = []
    feature_times: list[float] = []
    feature_records = 0
    frames_processed = 0
    started = time.monotonic()

    try:
        source_obj.open()
        detector.warmup()
        tracker.initialize()
        pose_estimator.warmup()
        while True:
            result = source_obj.read()
            if result is None:
                break
            metadata, frame = result

            detect_started = time.monotonic()
            detections = detector.detect(frame, timestamp=metadata.timestamp)
            detect_elapsed = time.monotonic() - detect_started

            track_started = time.monotonic()
            tracks = tracker.update(detections)
            track_elapsed = time.monotonic() - track_started

            pose_started = time.monotonic()
            records = pose_estimator.batch_estimate(
                frame, tracks, timestamp=metadata.timestamp
            )
            pose_elapsed = time.monotonic() - pose_started

            feature_started = time.monotonic()
            feature_records_list = feature_engine.process(
                records, frame_index=metadata.frame_index
            )
            feature_elapsed = time.monotonic() - feature_started

            detection_times.append(detect_elapsed * 1000.0)
            tracking_times.append(track_elapsed * 1000.0)
            pose_times.append(pose_elapsed * 1000.0)
            feature_times.append(feature_elapsed * 1000.0)
            feature_records += len(feature_records_list)
            frames_processed += 1

            if display:
                annotated = track_annotator.annotate_tracks(frame, tracks)
                annotated = pose_annotator.annotate(annotated, records)
                annotated = _annotate_pairs(
                    annotated,
                    records,
                    feature_records_list,
                    pair_annotator,
                )
                cv2.imshow("SafeWatch: behavioral features", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if num_frames is not None and frames_processed >= num_frames:
                break
    finally:
        source_obj.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - started
    _report_benchmark(
        benchmark_logger,
        benchmark=benchmark,
        frames_processed=frames_processed,
        feature_records=feature_records,
        detection_times=detection_times,
        tracking_times=tracking_times,
        pose_times=pose_times,
        feature_times=feature_times,
        elapsed=elapsed,
    )


def _annotate_pairs(
    annotated: np.ndarray,
    records: list[PoseRecord],
    feature_records: list[FeatureRecord],
    pair_annotator: InteractionAnnotator,
) -> np.ndarray:
    """Draw the pair overlays (distance + reciprocity) for each active pair."""

    records_by_track = {record.track_id: record for record in records}
    drawn: set[tuple[int, int]] = set()
    for feature_record in feature_records:
        if feature_record.pair_track_id is None:
            continue
        pair = (feature_record.track_id, feature_record.pair_track_id)
        if pair in drawn or (pair[1], pair[0]) in drawn:
            continue
        first = records_by_track.get(feature_record.track_id)
        second = records_by_track.get(feature_record.pair_track_id)
        if first is None or second is None:
            continue
        drawn.add(pair)
        values = dict(zip(feature_record.names, feature_record.values))
        annotated = pair_annotator.draw_distance(
            annotated,
            first,
            second,
            values.get("inter.distance", float("nan")),
        )
        annotated = pair_annotator.draw_reciprocity(
            annotated,
            first,
            second,
            values.get("temporal.reciprocity", float("nan")),
        )
    return annotated


def _report_benchmark(
    logger: logging.Logger,
    *,
    benchmark: bool,
    frames_processed: int,
    feature_records: int,
    detection_times: list[float],
    tracking_times: list[float],
    pose_times: list[float],
    feature_times: list[float],
    elapsed: float,
) -> None:
    if not benchmark:
        return
    if frames_processed == 0:
        return  # figures are meaningless without frames; emit nothing
    mean_detection = sum(detection_times) / len(detection_times)
    mean_tracking = sum(tracking_times) / len(tracking_times)
    mean_pose = sum(pose_times) / len(pose_times)
    mean_feature = sum(feature_times) / len(feature_times)
    fps = frames_processed / elapsed if elapsed > 0 else 0.0
    _log_benchmark(
        logger,
        {
            "frames": frames_processed,
            "feature_records": feature_records,
            "detection_ms": round(mean_detection, 3),
            "tracking_ms": round(mean_tracking, 3),
            "pose_ms": round(mean_pose, 3),
            "feature_ms": round(mean_feature, 3),
            "fps": round(fps, 2),
            "elapsed_s": round(elapsed, 3),
        },
    )


def _log_benchmark(logger: logging.Logger, extra: dict[str, object]) -> None:
    """Emit a JSON benchmark record with measured figures as ``extra`` fields."""

    logger.info(
        "benchmark complete",
        extra={"safewatch_extra": extra},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="video file path or webcam device index (e.g. 0)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="path to the runtime YAML config",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="override the detection confidence threshold",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="override the inference device",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=None,
        help="override the pose confidence threshold",
    )
    parser.add_argument(
        "--frames", type=int, default=None, help="stop after this many frames"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="run without a display window"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="log per-stage latencies and FPS"
    )
    args = parser.parse_args()

    settings = _load_runtime_config(
        config_path=args.config,
        conf=args.conf,
        device=args.device,
        pose_conf=args.pose_conf,
    )
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    run(
        settings=settings,
        source=args.source,
        display=not args.no_display,
        num_frames=args.frames,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
