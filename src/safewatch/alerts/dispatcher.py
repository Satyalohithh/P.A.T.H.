"""Alert dispatch channels (websocket, database, webhook)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from safewatch.core.schemas.alert import Alert


class AlertDispatcher:
    """Fan-out of alerts to registered channel callables."""

    def __init__(self, channels: Iterable[str] = ("websocket", "database")) -> None:
        self.channels = tuple(channels)

    def dispatch(self, alert: Alert) -> None:
        raise NotImplementedError("TODO(implementation): AlertDispatcher.dispatch")

    def register_channel(self, name: str, handler: Callable[[Alert], None]) -> None:
        raise NotImplementedError(
            "TODO(implementation): AlertDispatcher.register_channel"
        )


__all__ = ["Alert", "AlertDispatcher"]
