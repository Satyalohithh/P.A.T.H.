"""Repository pattern over SQLAlchemy sessions."""

from __future__ import annotations

from collections.abc import Sequence

from safewatch.core.schemas.alert import Alert
from safewatch.core.schemas.behavior import ClassificationResult


class AlertRepository:
    """Persistence/query operations for alerts."""

    def insert(self, alert: Alert) -> None:
        raise NotImplementedError("TODO(implementation): AlertRepository.insert")

    def update_status(self, alert_id: str, status: str) -> None:
        raise NotImplementedError("TODO(implementation): AlertRepository.update_status")

    def list(self, limit: int = 100) -> list[Alert]:
        raise NotImplementedError("TODO(implementation): AlertRepository.list")


class BehaviorRecordRepository:
    """Persistence/query operations for classified windows."""

    def insert_many(self, records: Sequence[ClassificationResult]) -> None:
        raise NotImplementedError(
            "TODO(implementation): BehaviorRecordRepository.insert_many"
        )

    def count_by_label(self, label: int) -> int:
        raise NotImplementedError(
            "TODO(implementation): BehaviorRecordRepository.count_by_label"
        )


__all__ = [
    "Alert",
    "AlertRepository",
    "BehaviorRecordRepository",
    "ClassificationResult",
]
