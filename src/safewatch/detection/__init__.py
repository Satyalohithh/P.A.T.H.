"""Person detection stage."""

from __future__ import annotations

from safewatch.detection.model_loader import ModelArtifact, ModelLoader
from safewatch.detection.person_detector import PersonDetector
from safewatch.detection.yolo_person_detector import YOLOPersonDetector

__all__ = [
    "ModelArtifact",
    "ModelLoader",
    "PersonDetector",
    "YOLOPersonDetector",
]
