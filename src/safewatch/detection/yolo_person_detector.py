"""Ultralytics YOLO person-detection implementation."""

from __future__ import annotations

import logging
import statistics
from collections import deque
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from safewatch.core.config import DetectionConfig
from safewatch.core.exceptions import DetectionError
from safewatch.core.schemas.detection import BBox, Detection, DetectionList
from safewatch.core.types import Frame, TimePoint
from safewatch.detection.person_detector import PersonDetector
from safewatch.utils.logging import get_logger

ModelFactory: TypeAlias = Callable[[str], Any]


class YOLOPersonDetector(PersonDetector):
    """Person detector backed by an Ultralytics YOLO detection model.

    Only the configured ``person_class`` is kept; every other COCO class is
    filtered out at inference time. The model is loaded lazily on first use so
    tests can inject a mock via ``model_factory`` without downloading weights.
    """

    def __init__(
        self,
        config: DetectionConfig,
        *,
        logger: logging.Logger | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or get_logger("safewatch.detection.yolo")
        self._model_factory = model_factory or self._default_factory
        self._model: Any = None
        self._latencies: deque[float] = deque(maxlen=64)

    # -- PersonDetector ------------------------------------------------------

    def detect(self, frame: Frame, *, timestamp: TimePoint = 0.0) -> DetectionList:
        model = self._ensure_model()
        results = self._run(model, frame)
        return self._to_detections(results, timestamp=timestamp)

    def warmup(self) -> None:
        model = self._ensure_model()
        warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self._run(model, warmup_frame)
        except Exception as exc:
            raise DetectionError(f"detector warmup failed: {exc}") from exc

    @property
    def latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return statistics.median(self._latencies)

    # -- internals -----------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(self._config.weights)
        return self._model

    def _run(self, model: Any, frame: Frame) -> Any:
        device = self._resolve_device()
        try:
            results = model(
                frame,
                conf=self._config.conf_threshold,
                classes=[self._config.person_class],
                device=device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectionError(f"detection inference failed: {exc}") from exc
        return results[0]

    def _resolve_device(self) -> str:
        return self._config.device if self._config.device != "auto" else "cpu"

    def _default_factory(self, weights: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - guarded by deps
            raise DetectionError(
                "ultralytics is not installed; run `uv sync --group dev`"
            ) from exc
        return YOLO(weights)

    def _to_detections(self, result: Any, *, timestamp: TimePoint) -> DetectionList:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        detections: DetectionList = []
        boxes_data = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        for det_id, (xyxy, conf, cls) in enumerate(zip(boxes_data, confs, classes)):
            xmin, ymin, xmax, ymax = (float(v) for v in xyxy)
            detections.append(
                Detection(
                    id=det_id,
                    bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    confidence=float(conf),
                    class_id=int(cls),
                    class_name="person",
                    timestamp=timestamp,
                )
            )
        return detections


__all__ = ["YOLOPersonDetector"]
