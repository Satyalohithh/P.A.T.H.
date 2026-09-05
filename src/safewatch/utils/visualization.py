"""Visualization helpers for debugging and evaluation."""

from __future__ import annotations

from safewatch.core.schemas.detection import DetectionList
from safewatch.core.schemas.pose import PoseResult
from safewatch.ingestion.decoder import Frame


class Visualization:
    """Draws boxes, skeletons, and risk overlays onto frames."""

    @staticmethod
    def draw_detections(frame: Frame, detections: DetectionList) -> Frame:
        raise NotImplementedError("TODO(implementation): Visualization.draw_detections")

    @staticmethod
    def draw_pose(frame: Frame, pose: PoseResult) -> Frame:
        raise NotImplementedError("TODO(implementation): Visualization.draw_pose")

    @staticmethod
    def draw_risk_overlay(frame: Frame, score: float, level: str) -> Frame:
        raise NotImplementedError(
            "TODO(implementation): Visualization.draw_risk_overlay"
        )


__all__ = ["DetectionList", "PoseResult", "Visualization"]
