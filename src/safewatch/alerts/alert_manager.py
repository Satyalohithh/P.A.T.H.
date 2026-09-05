"""Alert lifecycle: creation, acknowledgement, resolution."""

from __future__ import annotations

from safewatch.core.constants import AlertStatus
from safewatch.core.schemas.alert import Alert
from safewatch.core.schemas.risk import RiskAssessment
from safewatch.core.types import StreamId


class AlertManager:
    """Owns alert records and their status transitions."""

    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}

    def create(self, stream_id: StreamId, risk: RiskAssessment, message: str) -> Alert:
        raise NotImplementedError("TODO(implementation): AlertManager.create")

    def acknowledge(self, alert_id: str) -> Alert:
        raise NotImplementedError("TODO(implementation): AlertManager.acknowledge")

    def resolve(self, alert_id: str) -> Alert:
        raise NotImplementedError("TODO(implementation): AlertManager.resolve")

    def get(self, alert_id: str) -> Alert | None:
        raise NotImplementedError("TODO(implementation): AlertManager.get")


__all__ = ["Alert", "AlertManager", "AlertStatus", "RiskAssessment", "StreamId"]
