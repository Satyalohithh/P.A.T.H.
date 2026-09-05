#!/usr/bin/env python3
"""Real-time person tracking preview with stable IDs.

Feeds a video file or webcam through YOLO person detection and ByteTrack,
showing the annotated result in an OpenCV window until 'q' is pressed or the
stream ends. Track labels (e.g. ``Person #7``) are stable across frames.

Usage:
    uv run python scripts/demo_tracking.py --source sample.mp4
    uv run python scripts/demo_tracking.py --source 0 --device cpu
    uv run python scripts/demo_tracking.py --source sample.mp4 --benchmark --frames 300
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2

from safewatch.core.config import Settings, load_settings
from safewatch.detection import YOLOPersonDetector
from safewatch.ingestion import VideoFileSource, WebcamSource
from safewatch.ingestion.interfaces import FrameSource
from safewatch.tracking import ByteTrackTracker
from safewatch.utils.annotation import FrameAnnotator
from safewatch.utils.logging import configure_logging, get_logger


def _build_source(source: str) -> FrameSource:
    if source.isdigit():
        return WebcamSource(device_index=int(source))
    return VideoFileSource(path=source)


def _load_runtime_config(
    config_path: str,
    conf: float | None,
    device: str | None,
    track_thresh: float | None,
    match_thresh: float | None,
    track_buffer: int | None,
    min_hits: int | None,
) -> Settings:
    overrides: dict[str, object] = {}
    detection_overrides: dict[str, object] = {}
    if conf is not None:
        detection_overrides["conf_threshold"] = conf
    if device is not None:
        detection_overrides["device"] = device
    if detection_overrides:
        overrides["detection"] = detection_overrides

    tracking_overrides: dict[str, object] = {}
    if track_thresh is not None:
        tracking_overrides["track_thresh"] = track_thresh
    if match_thresh is not None:
        tracking_overrides["match_thresh"] = match_thresh
    if track_buffer is not None:
        tracking_overrides["track_buffer"] = track_buffer
    if min_hits is not None:
        tracking_overrides["min_hits"] = min_hits
    if tracking_overrides:
        overrides["tracking"] = tracking_overrides

    return load_settings(config_path=config_path, overrides=overrides)


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
    annotator = FrameAnnotator()
    benchmark_logger = get_logger("safewatch.benchmark")
    source_obj = _build_source(source)

    detection_times: list[float] = []
    tracking_times: list[float] = []
    frames_processed = 0
    started = time.monotonic()

    try:
        source_obj.open()
        detector.warmup()
        tracker.initialize()
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

            detection_times.append(detect_elapsed * 1000.0)
            tracking_times.append(track_elapsed * 1000.0)
            frames_processed += 1

            if display:
                annotated = annotator.annotate_tracks(frame, tracks)
                cv2.imshow("SafeWatch: person tracking", annotated)
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
        detection_times=detection_times,
        tracking_times=tracking_times,
        elapsed=elapsed,
    )


def _report_benchmark(
    logger: logging.Logger,
    *,
    benchmark: bool,
    frames_processed: int,
    detection_times: list[float],
    tracking_times: list[float],
    elapsed: float,
) -> None:
    if not benchmark:
        return
    if frames_processed == 0:
        return  # figures are meaningless without frames; emit nothing
    mean_detection = sum(detection_times) / len(detection_times)
    mean_tracking = sum(tracking_times) / len(tracking_times)
    fps = frames_processed / elapsed if elapsed > 0 else 0.0
    _log_benchmark(
        logger,
        {
            "frames": frames_processed,
            "detection_ms": round(mean_detection, 3),
            "tracking_ms": round(mean_tracking, 3),
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
        "--track-thresh",
        type=float,
        default=None,
        help="override the tracking threshold",
    )
    parser.add_argument(
        "--match-thresh",
        type=float,
        default=None,
        help="override the tracking match threshold",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=None,
        help="override the lost-track buffer (frames)",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help="override the minimum hits to report a track",
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
        track_thresh=args.track_thresh,
        match_thresh=args.match_thresh,
        track_buffer=args.track_buffer,
        min_hits=args.min_hits,
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
