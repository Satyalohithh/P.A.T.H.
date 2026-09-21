"""Tests for dataset statistics computation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from safewatch.data.statistics import compute_statistics, write_statistics


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_path": [f"raw/clip_{i:03d}.mp4" for i in range(8)],
            "label": ["aggressive", "normal"] * 4,
            "source_dataset": ["rwf2000", "hockey_fight", "rwf2000", "hockey_fight"]
            * 2,
            "duration_s": [2.0, 8.0, 0.5, 4.0] * 2,
        }
    )


def test_class_counts_and_availability() -> None:
    stats = compute_statistics(_manifest())

    assert stats["total_clips"] == 8
    assert stats["class_counts"] == {"normal": 4, "playful": 0, "aggressive": 4}
    assert stats["available_classes"] == ["normal", "aggressive"]
    assert stats["missing_classes"] == ["playful"]


def test_source_distribution_and_crosstab() -> None:
    stats = compute_statistics(_manifest())

    assert stats["source_distribution"] == {"rwf2000": 4, "hockey_fight": 4}
    assert stats["class_source_crosstab"]["aggressive"] == {
        "rwf2000": 4,
        "hockey_fight": 0,
    }
    assert stats["class_source_crosstab"]["normal"] == {
        "rwf2000": 0,
        "hockey_fight": 4,
    }


def test_duration_summary() -> None:
    stats = compute_statistics(_manifest())

    duration = stats["duration"]
    assert duration["count"] == 8
    assert duration["missing"] == 0
    assert duration["mean_s"] == 3.625
    assert duration["min_s"] == 0.5
    assert duration["max_s"] == 8.0
    assert duration["histogram_s"]["[0,1)"] == 2
    assert duration["histogram_s"]["[1,3)"] == 2
    assert duration["histogram_s"]["[3,5)"] == 2
    assert duration["histogram_s"]["[5,10)"] == 2


def test_duration_is_none_without_probe() -> None:
    manifest = _manifest().drop(columns=["duration_s"])
    stats = compute_statistics(manifest)
    assert stats["duration"] is None


def test_invalid_label_rejected() -> None:
    manifest = _manifest()
    manifest.loc[0, "label"] = "fight"
    try:
        compute_statistics(manifest)
    except ValueError as exc:
        assert "invalid label" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid label")


def test_write_statistics_json(tmp_path: Path) -> None:
    output = tmp_path / "dataset_statistics.json"
    report = write_statistics(_manifest(), output)

    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == report
    assert payload["available_classes"] == ["normal", "aggressive"]
