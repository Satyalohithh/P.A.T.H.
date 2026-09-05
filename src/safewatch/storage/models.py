"""SQLAlchemy ORM models (declared, tables created via migration)."""

from __future__ import annotations

from safewatch.core.schemas.alert import Alert
from safewatch.core.schemas.behavior import ClassificationResult


class BehaviorRecord:
    """ORM table: one row per classified interaction window."""

    def __init__(self, result: ClassificationResult) -> None:
        self.result = result

    @property
    def primary_key(self) -> str:
        raise NotImplementedError("TODO(implementation): BehaviorRecord.primary_key")


class AlertRecord:
    """ORM table mirroring :class:`Alert` for persistence."""

    def __init__(self, alert: Alert) -> None:
        self.alert = alert

    @property
    def primary_key(self) -> str:
        return self.alert.id


__all__ = ["Alert", "AlertRecord", "BehaviorRecord", "ClassificationResult"]
