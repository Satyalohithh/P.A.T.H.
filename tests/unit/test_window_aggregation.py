"""Tests for FeatureRecord -> pair-window dataset aggregation."""

from __future__ import annotations

import math

import pytest

from safewatch.behavior.windowing import (
    FEATURE_NAMES_EGO,
    FEATURE_NAMES_PAIR,
    build_pair_windows,
)
from safewatch.core.schemas.features import FeatureRecord

EGO = FEATURE_NAMES_EGO
PAIR = FEATURE_NAMES_PAIR


def _stream(
    frames: int,
    *,
    ego_a: tuple[float, ...] | None = None,
    ego_b: tuple[float, ...] | None = None,
    pair: tuple[float, ...] | None = None,
) -> list[FeatureRecord]:
    ego_a = ego_a or (0.1, 1.0, 2.0, 3.0, 4.0)
    ego_b = ego_b or (0.2, 5.0, 6.0, 7.0, 8.0)
    pair = pair or (10.0, 0.5, 2.0, 0.9)
    records: list[FeatureRecord] = []
    for frame_index in range(frames):
        records.append(
            FeatureRecord(
                track_id=1,
                frame_index=frame_index,
                timestamp=frame_index / 30.0,
                names=EGO,
                values=ego_a,
            )
        )
        records.append(
            FeatureRecord(
                track_id=2,
                frame_index=frame_index,
                timestamp=frame_index / 30.0,
                names=EGO,
                values=ego_b,
            )
        )
        records.append(
            FeatureRecord(
                track_id=1,
                frame_index=frame_index,
                timestamp=frame_index / 30.0,
                names=PAIR,
                values=pair,
                pair_track_id=2,
            )
        )
    return records


def test_row_shape_and_names() -> None:
    rows = build_pair_windows(_stream(70), video_id="cam-01", window_frames=30)
    assert len(rows) == 3
    row = rows[0]
    assert row.dimension == 14
    assert row.names[:5] == tuple(f"{name}.A" for name in EGO)
    assert row.names[5:10] == tuple(f"{name}.B" for name in EGO)
    assert row.names[10:] == PAIR


def test_window_means() -> None:
    rows = build_pair_windows(_stream(70), video_id="v", window_frames=30)
    first = rows[0].values
    assert first[0] == pytest.approx(0.1)
    assert first[5] == pytest.approx(0.2)
    assert first[1] == pytest.approx(1.0)
    assert first[10] == pytest.approx(10.0)
    assert first[13] == pytest.approx(0.9)
    assert rows[0].metadata["window_start"] == 0
    assert rows[0].metadata["window_end"] == 29
    assert rows[1].metadata["window_start"] == 30
    assert rows[2].metadata["window_start"] == 60


def test_mean_skips_nan_values() -> None:
    stream: list[FeatureRecord] = []
    for frame_index in range(2):
        stream.append(
            FeatureRecord(
                track_id=1,
                frame_index=frame_index,
                timestamp=float(frame_index),
                names=("pose.torso_lean", "motion.arm_angular_vel.L"),
                values=(2.0, float("nan")),
            )
        )
        stream.append(
            FeatureRecord(
                track_id=2,
                frame_index=frame_index,
                timestamp=float(frame_index),
                names=("pose.torso_lean", "motion.arm_angular_vel.L"),
                values=(4.0, float("nan")),
            )
        )
        stream.append(
            FeatureRecord(
                track_id=1,
                frame_index=frame_index,
                timestamp=float(frame_index),
                names=PAIR,
                values=(10.0, 0.5, 2.0, 0.9),
                pair_track_id=2,
            )
        )
    rows = build_pair_windows(stream, video_id="v", window_frames=30)
    assert len(rows) == 1
    row = rows[0]
    assert row.values[0] == pytest.approx(2.0)
    assert math.isnan(row.values[1])
    assert row.values[5] == pytest.approx(4.0)


def test_row_skipped_when_track_missing_in_window() -> None:
    stream = _stream(60)
    # window 1 (frames 30..59) drops every presence of track 2.
    for record in list(stream):
        if record.frame_index >= 30 and record.track_id == 2:
            stream.remove(record)
    rows = build_pair_windows(stream, video_id="v", window_frames=30)
    assert [r.metadata["window_start"] for r in rows] == [0]


def test_no_pair_presence_yields_no_rows() -> None:
    records = [
        FeatureRecord(
            track_id=1, frame_index=5, timestamp=5.0, names=EGO[:1], values=(1.0,)
        )
    ]
    assert build_pair_windows(records, video_id="v") == []


def test_invalid_window_frames_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_pair_windows([], video_id="v", window_frames=0)


def test_metadata_carries_video_and_window() -> None:
    rows = build_pair_windows(_stream(30), video_id="stream-9", window_frames=30)
    assert rows[0].metadata["video_id"] == "stream-9"
    assert rows[0].metadata["tracks"] == ("1", "2")
    assert rows[0].label is None


def test_unsigned_records_pool_into_each_scoped_video() -> None:
    rows_a = build_pair_windows(_stream(30), video_id="v-a", window_frames=30)
    rows_b = build_pair_windows(_stream(30), video_id="v-b", window_frames=30)
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0].metadata["video_id"] == "v-a"
    assert rows_b[0].metadata["video_id"] == "v-b"


def test_video_id_scopes_mixed_stream() -> None:
    stream_a = _stream(30)
    labeled_a = [
        FeatureRecord(
            track_id=r.track_id,
            frame_index=r.frame_index,
            timestamp=r.timestamp,
            names=r.names,
            values=r.values,
            pair_track_id=r.pair_track_id,
            video_id="v-a",
        )
        for r in stream_a
    ]
    stream_b = _stream(30, ego_a=(9.9, 1.0, 2.0, 3.0, 4.0))
    labeled_b = [
        FeatureRecord(
            track_id=r.track_id,
            frame_index=r.frame_index,
            timestamp=r.timestamp,
            names=r.names,
            values=r.values,
            pair_track_id=r.pair_track_id,
            video_id="v-b",
        )
        for r in stream_b
    ]
    mixed = [*labeled_a, *labeled_b]
    rows_a = build_pair_windows(mixed, video_id="v-a", window_frames=30)
    rows_b = build_pair_windows(mixed, video_id="v-b", window_frames=30)
    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0].values[0] == pytest.approx(0.1)
    assert rows_b[0].values[0] == pytest.approx(9.9)
