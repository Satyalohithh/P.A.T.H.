"""Tests for FeatureRecord <-> DataFrame / Parquet round-tripping."""

from __future__ import annotations

import math
from pathlib import Path

from safewatch.core.schemas.features import FeatureRecord
from safewatch.data.extraction import (
    frame_to_records,
    load_records,
    records_to_frame,
    save_records,
)


def _records() -> list[FeatureRecord]:
    return [
        FeatureRecord(
            track_id=1,
            frame_index=0,
            timestamp=0.0,
            names=("pose.torso_lean", "inter.distance"),
            values=(0.3, 2.0),
            pair_track_id=2,
            video_id="clip-a",
        ),
        FeatureRecord(
            track_id=1,
            frame_index=1,
            timestamp=1 / 30.0,
            names=("pose.torso_lean",),
            values=(float("nan"),),
            video_id="clip-a",
        ),
    ]


def test_records_to_frame_long_format() -> None:
    frame = records_to_frame(_records())

    assert list(frame.columns) == [
        "video_id",
        "track_id",
        "frame_index",
        "timestamp",
        "pair_track_id",
        "name",
        "value",
    ]
    assert len(frame) == 3
    pair_rows = frame[frame["name"] == "inter.distance"]
    assert int(pair_rows["pair_track_id"].iloc[0]) == 2


def test_frame_to_records_roundtrip() -> None:
    original = _records()
    rebuilt = frame_to_records(records_to_frame(original))

    assert len(rebuilt) == len(original)
    first = rebuilt[0]
    assert first.track_id == 1
    assert first.frame_index == 0
    assert first.video_id == "clip-a"
    assert first.pair_track_id == 2
    assert first.names == ("pose.torso_lean", "inter.distance")
    assert first.values == (0.3, 2.0)


def test_nan_and_missing_pair_survive_roundtrip() -> None:
    rebuilt = frame_to_records(records_to_frame(_records()))
    second = rebuilt[1]
    assert second.pair_track_id is None
    assert math.isnan(second.values[0])


def test_save_and_load_records_parquet(tmp_path: Path) -> None:
    output = tmp_path / "features" / "clip.parquet"
    save_records(_records(), output)
    assert output.is_file()

    loaded = load_records(output)
    assert len(loaded) == 2
    assert loaded[0].values[0] == 0.3
    assert loaded[0].video_id == "clip-a"
