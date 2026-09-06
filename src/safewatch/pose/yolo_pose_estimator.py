"""YOLO-based pose estimator (YOLOv8n-pose)."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from statistics import median
from typing import Any

import numpy as np

from safewatch.core.config import PoseConfig
from safewatch.core.exceptions import PoseEstimationError
from safewatch.core.schemas.detection import BBox
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.schemas.tracking import Track
from safewatch.core.types import Frame, TimePoint
from safewatch.pose.associator import PoseAssociator, PoseCandidate
from safewatch.pose.estimator import PoseEstimator
from safewatch.pose.keypoint_layout import validate_keypoints

KEYPOINTS_PER_INSTANCE = 17

ModelFactory = Callable[[str], Any]


class YOLOPoseEstimator(PoseEstimator):
    """Pose estimation via a YOLO keypoint model.

    Weights load lazily through ``model_factory`` (defaulting to ultralytics'
    ``YOLO``), which keeps the constructor testable without a model download.
    Each frame runs one full-frame inference; detected people become
    :class:`PoseCandidate` instances and are matched to tracks by IoU.
    """

    def __init__(
        self,
        config: PoseConfig,
        model_factory: ModelFactory | None = None,
        associator: PoseAssociator | None = None,
    ) -> None:
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self._config = config
        self._model_factory = model_factory
        self._model: Any = None
        self._associator = associator or PoseAssociator()
        self._latency_ms: deque[float] = deque(maxlen=64)

    def _get_model(self) -> Any:
        """Load the ultralytics model once, on first use."""

        if self._model is None:
            try:
                self._model = self._model_factory(self._config.weights)
            except Exception as error:  # pragma: no cover - backend boundary
                raise PoseEstimationError(
                    f"failed to load pose weights {self._config.weights!r}: {error}"
                ) from error
        return self._model

    def warmup(self) -> None:
        """Load weights and run a zero-image warmup inference.

        The warmup pass is purely about allocating weights and device caches
        before steady-state use.
        """

        model = self._get_model()
        try:
            warmup_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            self._run(model, warmup_frame)
        except Exception as error:  # pragma: no cover - backend boundary
            raise PoseEstimationError(f"pose warmup failed: {error}") from error

    def estimate(
        self,
        frame: Frame,
        track: Track,
        *,
        timestamp: TimePoint = 0.0,
    ) -> PoseRecord | None:
        records = self.batch_estimate(frame, [track], timestamp=timestamp)
        return records[0] if records else None

    def batch_estimate(
        self,
        frame: Frame,
        tracks: list[Track],
        *,
        timestamp: TimePoint = 0.0,
    ) -> list[PoseRecord]:
        if not tracks:
            return []
        candidates = self._detect_candidates(frame)
        return self._associator.associate(tracks, candidates, timestamp=timestamp)

    def _detect_candidates(self, frame: Frame) -> list[PoseCandidate]:
        """Run one full-frame inference and map instances to candidates."""

        model = self._get_model()
        started = time.perf_counter()
        predictions = self._run(model, frame)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms > 0.0:
            self._latency_ms.append(elapsed_ms)

        candidates: list[PoseCandidate] = []
        for result in predictions:
            boxes = getattr(result, "boxes", None)
            keypoints = getattr(result, "keypoints", None)
            if boxes is None or keypoints is None:
                continue
            data = keypoints.data
            if data is None:
                continue
            keypoints_np = (
                data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
            )
            xyxys = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for row in range(keypoints_np.shape[0]):
                points = keypoints_np[row]
                if points.shape[0] != KEYPOINTS_PER_INSTANCE:
                    raise PoseEstimationError(
                        f"pose model returned {points.shape[0]} keypoints, "
                        f"expected {KEYPOINTS_PER_INSTANCE}"
                    )
                keypoints = tuple(
                    tuple(float(value) for value in point) for point in points
                )
                validate_keypoints(keypoints)
                x1, y1, x2, y2 = (float(v) for v in xyxys[row])
                instance_conf = float(confs[row]) if confs[row] else 0.0
                candidates.append(
                    PoseCandidate(
                        bbox=BBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2),
                        keypoints=keypoints,
                        confidence=instance_conf,
                    )
                )
        return candidates

    def _run(self, model: Any, frame: Frame) -> Any:
        """Run one inference and return the list of result objects."""

        try:
            return model(
                frame,
                conf=self._config.conf_threshold,
                verbose=False,
                device=self._resolve_device(),
            )
        except Exception as error:  # pragma: no cover - backend boundary
            raise PoseEstimationError(f"pose inference failed: {error}") from error

    def _resolve_device(self) -> str:
        return self._config.device if self._config.device != "auto" else "cpu"

    @property
    def latency_ms(self) -> float:
        if not self._latency_ms:
            return 0.0
        return median(self._latency_ms)


__all__ = ["YOLOPoseEstimator"]
