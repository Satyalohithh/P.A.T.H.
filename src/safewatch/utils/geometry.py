"""Geometry helpers (reimplementation of scipy/numpy vector math)."""

from __future__ import annotations

from safewatch.core.types import Vec2D


class Geometry:
    """Pure-python geometry functions used before numpy is installed."""

    @staticmethod
    def distance(a: Vec2D, b: Vec2D) -> float:
        raise NotImplementedError("TODO(implementation): Geometry.distance")

    @staticmethod
    def angle_between(v1: Vec2D, v2: Vec2D) -> float:
        raise NotImplementedError("TODO(implementation): Geometry.angle_between")

    @staticmethod
    def project(v: Vec2D, onto: Vec2D) -> Vec2D:
        raise NotImplementedError("TODO(implementation): Geometry.project")


__all__ = ["Geometry", "Vec2D"]
