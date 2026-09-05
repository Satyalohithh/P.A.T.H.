#!/usr/bin/env python3
"""Real-time person-detection preview.

Feeds a video file or webcam through YOLO person detection and shows the
annotated result in an OpenCV window until 'q' is pressed or the stream ends.

Usage:
    uv run python scripts/demo_detection.py --source sample.mp4
    uv run python scripts/demo_detection.py --source 0 --device cpu
    uv run python scripts/demo_detection.py --no-display --source sample.mp4
"""

from __future__ import annotations

import argparse

import cv2

from safewatch.core.config import load_settings
from safewatch.detection import YOLOPersonDetector
from safewatch.ingestion import VideoFileSource, WebcamSource
from safewatch.ingestion.interfaces import FrameSource
from safewatch.utils.annotation import FrameAnnotator


def _build_source(source: str) -> FrameSource:
    if source.isdigit():
        return WebcamSource(device_index=int(source))
    return VideoFileSource(path=source)


def run(
    source: str, config_path: str, conf: float | None, device: str | None, display: bool
) -> None:
    settings = load_settings(config_path=config_path)
    detection_defaults = settings.detection

    overrides: dict[str, object] = {}
    if conf is not None:
        overrides["conf_threshold"] = conf
    if device is not None:
        overrides["device"] = device
    detection_config = detection_defaults.model_copy(update=overrides)

    detector = YOLOPersonDetector(detection_config)
    annotator = FrameAnnotator()
    source_obj = _build_source(source)

    try:
        source_obj.open()
        detector.warmup()
        while True:
            result = source_obj.read()
            if result is None:
                break
            metadata, frame = result
            detections = detector.detect(frame, timestamp=metadata.timestamp)
            if display:
                annotated = annotator.annotate(frame, detections)
                cv2.imshow("SafeWatch: person detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        source_obj.close()
        if display:
            cv2.destroyAllWindows()


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
        "--no-display", action="store_true", help="run without a display window"
    )
    args = parser.parse_args()

    run(
        source=args.source,
        config_path=args.config,
        conf=args.conf,
        device=args.device,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
