"""API route blueprints."""

from __future__ import annotations

from safewatch.api.routes.alerts import AlertsRouter
from safewatch.api.routes.analytics import AnalyticsRouter
from safewatch.api.routes.streams import StreamsRouter
from safewatch.api.routes.websocket import WebsocketRouter

__all__ = ["AlertsRouter", "AnalyticsRouter", "StreamsRouter", "WebsocketRouter"]
