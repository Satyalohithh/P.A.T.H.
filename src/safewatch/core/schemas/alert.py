"""Alert data contracts."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from safewatch.core.constants import AlertStatus
from safewatch.core.schemas.risk import RiskAssessment
from safewatch.core.types import StreamId


@dataclass(frozen=True, slots=True)
class Alert:
    """An alert emitted when a risk rule fires."""

    id: str
    stream_id: StreamId
    risk: RiskAssessment
    message: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, stream_id: StreamId, risk: RiskAssessment, message: str) -> Alert:
        return cls(
            id=str(uuid.uuid4()),
            stream_id=stream_id,
            risk=risk,
            message=message,
        )


class AlertEvent(dict):
    """JSON-serializable alert payload for webhook/websocket dispatch."""

    def __init__(self, alert: Alert) -> None:
        super().__init__(
            id=alert.id,
            stream_id=alert.stream_id,
            status=alert.status.value,
            message=alert.message,
            risk={
                "score": alert.risk.score,
                "level": alert.risk.level.value,
                "frame_index": alert.risk.frame_index,
                "track_id": alert.risk.track_id,
            },
            created_at=alert.created_at,
        )


@dataclass(frozen=True, slots=True)
class AlertReply:
    """Operator acknowledgement of an alert."""

    alert_id: str
    acknowledged_at: float = 0.0
    status: AlertStatus = AlertStatus.ACKNOWLEDGED


__all__ = ["Alert", "AlertEvent", "AlertReply"]
