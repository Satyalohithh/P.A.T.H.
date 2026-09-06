"""Tests for the sliding-window temporal aggregator."""

from __future__ import annotations

import math

from safewatch.core.schemas.features import FeatureVector
from safewatch.features.temporal_aggregator import TemporalAggregator


class TestWindow:
    def test_window_returns_column_means(self) -> None:
        agg = TemporalAggregator(window_frames=30)
        for frame in range(3):
            agg.push(
                FeatureVector(
                    names=("a", "b"),
                    values=(frame, 2 * frame),
                    frame_index=frame,
                    track_id=1,
                )
            )
        pooled = agg.window(1)
        assert pooled.names == ("a", "b")
        assert pooled.values == (1.0, 2.0)
        assert pooled.frame_index == 2

    def test_window_skips_nan(self) -> None:
        agg = TemporalAggregator()
        for value in (float("nan"), 4.0, 2.0):
            agg.push(
                FeatureVector(
                    names=("a",),
                    values=(value,),
                    frame_index=0,
                    track_id=1,
                )
            )
        assert agg.window(1).values == (3.0,)

    def test_window_trims_to_fixed_depth(self) -> None:
        agg = TemporalAggregator(window_frames=2)
        for frame in range(5):
            agg.push(
                FeatureVector(
                    names=("a",),
                    values=(float(frame),),
                    frame_index=frame,
                    track_id=1,
                )
            )
        pooled = agg.window(1)
        assert pooled.frame_index == 4
        assert pooled.values == (3.5,)

    def test_unknown_track_empty(self) -> None:
        agg = TemporalAggregator()
        assert agg.window(99).names == ()
        assert agg.window(99).values == ()

    def test_scoped_per_track(self) -> None:
        agg = TemporalAggregator()
        agg.push(FeatureVector(names=("a",), values=(1.0,), frame_index=0, track_id=1))
        agg.push(FeatureVector(names=("a",), values=(9.0,), frame_index=0, track_id=2))
        assert agg.window(1).values == (1.0,)
        assert agg.window(2).values == (9.0,)


class TestSummary:
    def test_summary_statistics(self) -> None:
        agg = TemporalAggregator(window_frames=30)
        for frame, value in enumerate((1.0, 2.0, 3.0, 4.0)):
            agg.push(
                FeatureVector(
                    names=("a",),
                    values=(value,),
                    frame_index=10 + frame,
                    track_id=1,
                )
            )
        summary = agg.summary(1)
        assert summary["mean"] == 2.5
        assert summary["min"] == 1.0
        assert summary["max"] == 4.0
        assert abs(summary["std"] - 1.1180339887) < 1e-6
        assert summary["window_start"] == 10
        assert summary["window_end"] == 13

    def test_summary_uses_finite_entries_only(self) -> None:
        agg = TemporalAggregator()
        agg.push(
            FeatureVector(
                names=("a",), values=(float("nan"),), frame_index=0, track_id=1
            )
        )
        agg.push(FeatureVector(names=("a",), values=(6.0,), frame_index=1, track_id=1))
        assert agg.summary(1)["mean"] == 6.0

    def test_summary_empty_track_is_nan(self) -> None:
        agg = TemporalAggregator()
        summary = agg.summary(1)
        assert math.isnan(summary["mean"])
        assert math.isnan(summary["std"])
        assert math.isnan(summary["min"])
        assert math.isnan(summary["max"])
        assert summary["window_start"] == 0
        assert summary["window_end"] == 0


class TestLifecycle:
    def test_reset_clears_history(self) -> None:
        agg = TemporalAggregator()
        agg.push(FeatureVector(names=("a",), values=(1.0,), frame_index=0, track_id=1))
        agg.reset()
        assert agg.window(1).values == ()
        assert math.isnan(agg.summary(1)["mean"])
