"""Per-rule cooldown tracking to suppress duplicates."""

from __future__ import annotations

from collections.abc import Mapping

from safewatch.core.types import StreamId


class CooldownTracker:
    """Marks a (stream, rule) key as cooling down for N seconds."""

    def __init__(self, cooldown_seconds: Mapping[str, float]) -> None:
        self.cooldown_seconds = dict(cooldown_seconds)

    def in_cooldown(self, rule_id: str, stream_id: StreamId) -> bool:
        raise NotImplementedError("TODO(implementation): CooldownTracker.in_cooldown")

    def hit(self, rule_id: str, stream_id: StreamId) -> None:
        raise NotImplementedError("TODO(implementation): CooldownTracker.hit")

    def reset(self) -> None:
        raise NotImplementedError("TODO(implementation): CooldownTracker.reset")


__all__ = ["CooldownTracker", "StreamId"]
