"""Tests for manifest -> train/val/test splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from safewatch.data.split import (
    SPLIT_FILES,
    split_manifest,
)


def _manifest(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_path": [f"raw/clip_{i:03d}.mp4" for i in range(n)],
            "label": ["aggressive", "normal"] * (n // 2),
            "source_dataset": ["rwf2000", "hockey_fight"] * (n // 2),
        }
    )


def test_split_writes_three_json_files(tmp_path: Path) -> None:
    output = tmp_path / "splits"
    result = split_manifest(
        _manifest(12), output, val_size=0.25, test_size=0.25, seed=7
    )

    assert result.counts == {"train": 6, "val": 3, "test": 3}
    for filename in SPLIT_FILES:
        assert (output / filename).is_file()


def test_split_format_matches_placeholder_schema(tmp_path: Path) -> None:
    output = tmp_path / "splits"
    split_manifest(_manifest(12), output, val_size=0.25, test_size=0.25, seed=7)

    payload = json.loads((output / "train.json").read_text(encoding="utf-8"))
    assert payload["version"] == "v0"
    assert "clips" in payload
    assert payload["clips"][0]["video_path"]
    assert payload["clips"][0]["label"] in {"aggressive", "normal"}
    assert payload["clips"][0]["source_dataset"]


def test_split_stratifies_by_label(tmp_path: Path) -> None:
    output = tmp_path / "splits"
    split_manifest(_manifest(20), output, val_size=0.25, test_size=0.25, seed=1)
    for split in ("train", "val", "test"):
        clips = json.loads((output / f"{split}.json").read_text(encoding="utf-8"))[
            "clips"
        ]
        labels = [clip["label"] for clip in clips]
        assert labels.count("aggressive") >= 1
        assert labels.count("normal") >= 1


def test_split_is_seed_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    split_manifest(_manifest(16), first, val_size=0.2, test_size=0.2, seed=42)
    split_manifest(_manifest(16), second, val_size=0.2, test_size=0.2, seed=42)

    for split in SPLIT_FILES:
        a = json.loads((first / split).read_text(encoding="utf-8"))["clips"]
        b = json.loads((second / split).read_text(encoding="utf-8"))["clips"]
        assert a == b


def test_split_rejects_too_small_classes(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        {
            "video_path": [f"raw/clip_{i:03d}.mp4" for i in range(4)],
            "label": ["aggressive", "aggressive", "normal", "playful"],
            "source_dataset": ["rwf2000"] * 4,
        }
    )
    with pytest.raises(ValueError, match="stratified"):
        split_manifest(manifest, tmp_path / "splits", val_size=0.25, test_size=0.25)


def test_split_rejects_invalid_sizes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="val_size and test_size"):
        split_manifest(_manifest(12), tmp_path / "splits", val_size=0.6, test_size=0.6)


def test_split_rejects_duplicate_paths(tmp_path: Path) -> None:
    manifest = _manifest(4)
    manifest.loc[1, "video_path"] = manifest.loc[0, "video_path"]
    with pytest.raises(ValueError, match="duplicate video_path"):
        split_manifest(manifest, tmp_path / "splits")
