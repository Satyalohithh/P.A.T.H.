"""WebSocket endpoints for live alert push."""

from __future__ import annotations

from safewatch.core.schemas.alert import AlertEvent


class WebsocketRouter:
    """Broadcasts alerts to connected dashboard clients."""

    def broadcast(self, event: AlertEvent) -> None:
        raise NotImplementedError("TODO(implementation): WebsocketRouter.broadcast")

    def clients(self) -> int:
        raise NotImplementedError("TODO(implementation): WebsocketRouter.clients")


__all__ = ["AlertEvent", "WebsocketRouter"]
