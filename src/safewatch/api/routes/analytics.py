"""Analytics/dashboard metrics API."""

from __future__ import annotations

from safewatch.core.constants import BehaviorClass


class AnalyticsRouter:
    """REST endpoints for aggregate dashboard metrics."""

    def behavior_distribution(self) -> dict[BehaviorClass, int]:
        raise NotImplementedError(
            "TODO(implementation): AnalyticsRouter.behavior_distribution"
        )

    def alert_velocity(self, window_seconds: int = 300) -> float:
        raise NotImplementedError(
            "TODO(implementation): AnalyticsRouter.alert_velocity"
        )

    def per_stream_risk(self, stream_id: str) -> dict[str, float]:
        raise NotImplementedError(
            "TODO(implementation): AnalyticsRouter.per_stream_risk"
        )


__all__ = ["AnalyticsRouter", "BehaviorClass"]
