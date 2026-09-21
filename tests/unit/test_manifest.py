"""Tests for the manifest CSV builder / loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from safewatch.data.manifest import (
    MANIFEST_COLUMNS,
    build_manifest,
    load_manifest,
)


def _populate_raw(raw: Path) -> None:
    (raw / "rwf_2000" / "Fight").mkdir(parents=True)
    (raw / "rwf_2000" / "NonFight").mkdir()
    (raw / "hockey_fight" / "fight").mkdir(parents=True)
    (raw / "hockey_fight" / "nofight").mkdir()
    (raw / "rwf_2000" / "Fight" / "fi001.mp4").touch()
    (raw / "rwf_2000" / "Fight" / "fi002.mp4").touch()
    (raw / "rwf_2000" / "NonFight" / "nf001.mp4").touch()
    (raw / "hockey_fight" / "fight" / "f001.avi").touch()
    (raw / "hockey_fight" / "nofight" / "n001.avi").touch()


def test_build_manifest_writes_required_columns(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _populate_raw(raw)
    output = tmp_path / "manifest.csv"

    build_manifest(raw, ["rwf2000", "hockey_fight"], output, probe=False)
    frame = load_manifest(output)

    assert list(frame.columns) == list(MANIFEST_COLUMNS)
    assert len(frame) == 5
    assert set(frame["source_dataset"]) == {"rwf2000", "hockey_fight"}
    assert set(frame["label"]) == {"aggressive", "normal"}
    assert frame["label"].str.contains("aggressive").sum() == 3


def test_manifest_paths_are_portable_relatives(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _populate_raw(raw)
    output = tmp_path / "manifest.csv"

    build_manifest(raw, ["rwf2000"], output, probe=False)
    frame = load_manifest(output)

    assert frame["video_path"].iloc[0] == "rwf_2000/Fight/fi001.mp4"
    for path in frame["video_path"]:
        assert not Path(path).is_absolute()


def test_manifest_skips_missing_videos(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _populate_raw(raw)
    (raw / "rwf_2000" / "Fight" / "fi001.mp4").unlink()
    output = tmp_path / "manifest.csv"

    build_manifest(raw, ["rwf2000"], output, probe=False)
    assert len(load_manifest(output)) == 2


def test_manifest_limit_per_source(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _populate_raw(raw)
    output = tmp_path / "manifest.csv"

    build_manifest(
        raw, ["rwf2000", "hockey_fight"], output, probe=False, limit_per_source=1
    )
    frame = load_manifest(output)
    assert len(frame) == 2
    assert frame["source_dataset"].nunique() == 2


def test_manifest_probe_degrades_on_tiny_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "rwf_2000" / "Fight").mkdir(parents=True)
    (raw / "rwf_2000" / "Fight" / "fi001.mp4").touch()  # 0 bytes -> unreadable
    output = tmp_path / "manifest.csv"

    build_manifest(raw, ["rwf2000"], output, probe=True)
    frame = load_manifest(output)

    assert len(frame) == 1
    assert frame["label"].iloc[0] == "aggressive"
    assert frame["duration_s"].isna().iloc[0]


def test_manifest_dedupes_same_underlying_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "rwf_2000" / "Fight").mkdir(parents=True)
    real = raw / "rwf_2000" / "Fight" / "fi001.mp4"
    real.touch()
    alias = raw / "rwf_2000" / "Fight" / "fi_alias.mp4"
    try:
        os.symlink(real, alias)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    output = tmp_path / "manifest.csv"

    build_manifest(raw, ["rwf2000"], output, probe=False)
    assert len(load_manifest(output)) == 1


def test_load_manifest_rejects_bad_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_manifest(bad)
