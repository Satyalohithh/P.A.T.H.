"""Alert query API."""

from __future__ import annotations

from safewatch.core.schemas.alert import Alert


class AlertsRouter:
    """REST endpoints for alert history and acknowledgement."""

    def list_alerts(self, stream_id: str | None, limit: int = 100) -> list[Alert]:
        raise NotImplementedError("TODO(implementation): AlertsRouter.list_alerts")

    def acknowledge_alert(self, alert_id: str) -> Alert:
        raise NotImplementedError(
            "TODO(implementation): AlertsRouter.acknowledge_alert"
        )


__all__ = ["AlertsRouter"]
