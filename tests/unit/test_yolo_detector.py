"""Tests for YOLOPersonDetector (mocked ultralytics model)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from safewatch.core.config import DetectionConfig
from safewatch.core.exceptions import DetectionError
from safewatch.detection.yolo_person_detector import YOLOPersonDetector


class _T:
    """Renderable array that answers .cpu().numpy() like ultralytics tensors."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _T:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy: list, conf: list, cls: list) -> None:
        self.xyxy = _T(np.asarray(xyxy, dtype=float))
        self.conf = _T(np.asarray(conf, dtype=float))
        self.cls = _T(np.asarray(cls, dtype=int))


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes | None) -> None:
        self.boxes = boxes

    def __len__(self) -> int:
        return 0 if self.boxes is None else 1


class _FakeModel:
    def __init__(self, boxes: _FakeBoxes | None, fail: bool = False) -> None:
        self.boxes = boxes
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def __call__(self, frame: Any, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("inference boom")
        return [_FakeResult(self.boxes)]


def _make_detector(
    boxes: _FakeBoxes | None,
    config: DetectionConfig | None = None,
    fail: bool = False,
) -> tuple[YOLOPersonDetector, _FakeModel]:
    model = _FakeModel(boxes, fail=fail)
    detector = YOLOPersonDetector(
        config or DetectionConfig(),
        model_factory=lambda weights: model,
    )
    return detector, model


def test_detect_maps_boxes_to_detections() -> None:
    boxes = _FakeBoxes(
        xyxy=[[1, 2, 10, 20], [50, 60, 80, 120]],
        conf=[0.9, 0.7],
        cls=[0, 0],
    )
    detector, _ = _make_detector(boxes)
    detections = detector.detect(np.zeros((100, 100, 3), np.uint8), timestamp=1.5)
    assert len(detections) == 2
    first, second = detections
    assert first.id == 0
    assert first.bbox.xmin == 1.0
    assert first.bbox.ymax == 20.0
    assert first.confidence == 0.9
    assert first.class_name == "person"
    assert first.timestamp == 1.5
    assert second.id == 1
    assert second.bbox.xmax == 80.0


def test_detect_empty_returns_empty() -> None:
    detector, model = _make_detector(_FakeBoxes(xyxy=[], conf=[], cls=[]))
    assert detector.detect(np.zeros((10, 10, 3), np.uint8)) == []
    assert len(model.calls) == 1


def test_no_boxes_returns_empty() -> None:
    detector, _ = _make_detector(None)
    assert detector.detect(np.zeros((10, 10, 3), np.uint8)) == []


def test_model_loaded_lazily_once() -> None:
    boxes = _FakeBoxes(xyxy=[[0, 0, 5, 5]], conf=[0.5], cls=[0])
    model = _FakeModel(boxes)
    detector = YOLOPersonDetector(
        DetectionConfig(), model_factory=lambda weights: model
    )
    assert model.calls == []
    detector.detect(np.zeros((10, 10, 3), np.uint8))
    detector.detect(np.zeros((10, 10, 3), np.uint8))
    assert len(model.calls) == 2


def test_detector_filters_classes_and_confidence() -> None:
    boxes = _FakeBoxes(xyxy=[[0, 0, 5, 5]], conf=[0.8], cls=[0])
    detector, model = _make_detector(boxes, config=DetectionConfig(conf_threshold=0.6))
    detector.detect(np.zeros((10, 10, 3), np.uint8))
    call_kwargs = model.calls[0]
    assert call_kwargs["conf"] == 0.6
    assert call_kwargs["classes"] == [0]
    assert call_kwargs["device"] == "cpu"


def test_latency_ms_rolling_median() -> None:
    detector, _ = _make_detector(_FakeBoxes(xyxy=[], conf=[], cls=[]))
    assert detector.latency_ms == 0.0
    for _ in range(3):
        detector.detect(np.zeros((10, 10, 3), np.uint8))
    assert detector.latency_ms >= 0.0


def test_warmup_runs_inference() -> None:
    detector, model = _make_detector(_FakeBoxes(xyxy=[], conf=[], cls=[]))
    detector.warmup()
    assert len(model.calls) == 1


def test_inference_error_raises_detection_error() -> None:
    detector, _ = _make_detector(_FakeBoxes(xyxy=[], conf=[], cls=[]), fail=True)
    with pytest.raises(DetectionError):
        detector.detect(np.zeros((10, 10, 3), np.uint8))
