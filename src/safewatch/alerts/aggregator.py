"""Alert aggregation across sources and time windows."""

from __future__ import annotations

from collections.abc import Iterable

from safewatch.core.schemas.alert import Alert
from safewatch.core.types import StreamId


class AlertAggregator:
    """Group/deduplicate alerts over a sliding window."""

    def __init__(self, window_seconds: float = 300.0) -> None:
        self.window_seconds = window_seconds

    def group(self, alerts: Iterable[Alert]) -> list[list[Alert]]:
        raise NotImplementedError("TODO(implementation): AlertAggregator.group")

    def digest_for(self, stream_id: StreamId, alerts: Iterable[Alert]) -> list[Alert]:
        raise NotImplementedError("TODO(implementation): AlertAggregator.digest_for")


__all__ = ["Alert", "AlertAggregator", "StreamId"]
