"""Timing utilities: Timer context manager and timed decorator.

Functional stdlib-only helpers used for latency budgets and profiling.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import ContextDecorator
from typing import Any, Self, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def elapsed_ms(seconds: float) -> float:
    """Convert a seconds value to milliseconds."""

    return seconds * 1000.0


class Timer(ContextDecorator):
    """Records elapsed wall-clock time over context or as a decorator."""

    def __init__(self, name: str = "timer", *, accumulate: bool = True) -> None:
        self.name = name
        self.accumulate = accumulate
        self._start: float | None = None
        self.intervals: list[float] = []

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._start is not None, "Timer started"
        elapsed = time.perf_counter() - self._start
        self.intervals.append(elapsed)
        self._start = None

    @property
    def last(self) -> float:
        """Duration (seconds) of the most recent interval."""

        return self.intervals[-1] if self.intervals else 0.0

    @property
    def total(self) -> float:
        return sum(self.intervals)

    @property
    def count(self) -> int:
        return len(self.intervals)


def timed(func: F) -> F:
    """Decorator wrapping ``func`` inside a :class:`Timer` per call."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timer = Timer(func.__name__)
        with timer:
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


__all__ = ["Timer", "elapsed_ms", "timed"]
