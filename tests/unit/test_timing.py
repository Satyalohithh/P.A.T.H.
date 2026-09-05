"""Tests for timing utilities."""

from __future__ import annotations

import time

from safewatch.utils.timing import Timer, elapsed_ms, timed


class TestTimer:
    def test_records_elapsed(self) -> None:
        timer = Timer("x")
        with timer:
            time.sleep(0.01)
        assert timer.count == 1
        assert timer.last >= 0.01

    def test_accumulates_intervals(self) -> None:
        timer = Timer("x")
        with timer:
            pass
        with timer:
            pass
        assert timer.count == 2
        assert timer.total >= 0.0

    def test_empty_timer_defaults(self) -> None:
        timer = Timer("x")
        assert timer.last == 0.0
        assert timer.total == 0.0
        assert timer.count == 0


class TestElapsedMs:
    def test_conversion(self) -> None:
        assert elapsed_ms(0.5) == 500.0


class TestTimed:
    def test_decorator_runs_and_returns(self) -> None:
        calls: list[float] = []

        @timed
        def work(v: int) -> int:
            calls.append(time.perf_counter())
            return v + 1

        assert work(1) == 2
        assert len(calls) == 1
