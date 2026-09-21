"""Tests for external dataset adapters (RWF-2000, Hockey Fight, UCF101)."""

from __future__ import annotations

from pathlib import Path

import pytest

from safewatch.data.adapters import (
    build_adapters,
    enabled_sources,
)
from safewatch.data.adapters.base import Clip
from safewatch.data.adapters.hockey_fight import HockeyFightAdapter
from safewatch.data.adapters.rwf2000 import RWF2000Adapter
from safewatch.data.adapters.ucf101 import UCF101Adapter
from safewatch.data.labels import DatasetClass


def _touch(directory: Path, name: str) -> None:
    (directory / name).touch()


def test_rwf2000_layout(tmp_path: Path) -> None:
    root = tmp_path / "rwf_2000"  # within datasets/raw
    (root / "Fight").mkdir(parents=True)
    (root / "NonFight").mkdir()
    _touch(root / "Fight", "fi0001.mp4")
    _touch(root / "Fight", "fi0002.mp4")
    _touch(root / "NonFight", "nf0001.mp4")

    clips = list(RWF2000Adapter().iter_clips(tmp_path))
    assert len(clips) == 3

    by_id = {clip.video_id: clip for clip in clips}
    assert by_id["rwf2000-rwf_2000-Fight-fi0001.mp4"].video_path.exists()
    assert by_id["rwf2000-rwf_2000-NonFight-nf0001.mp4"].native_label == "nofight"
    assert sorted(clip.native_label for clip in clips) == [
        "fight",
        "fight",
        "nofight",
    ]
    assert all(clip.source == "rwf2000" for clip in clips)


def test_rwf2000_case_and_nesting_tolerant(tmp_path: Path) -> None:
    (tmp_path / "rwf_2000" / "fight" / "deeper").mkdir(parents=True)
    (tmp_path / "rwf_2000" / "NOFIGHT").mkdir()
    _touch(tmp_path / "rwf_2000" / "fight" / "deeper", "a.mp4")
    _touch(tmp_path / "rwf_2000" / "NOFIGHT", "b.avi")

    clips = list(RWF2000Adapter().iter_clips(tmp_path))
    assert {clip.native_label for clip in clips} == {"fight", "nofight"}


def test_rwf2000_unmapped_video_skipped(tmp_path: Path) -> None:
    (tmp_path / "rwf_2000" / "Other").mkdir(parents=True)
    _touch(tmp_path / "rwf_2000" / "Other", "misc.mp4")
    assert list(RWF2000Adapter().iter_clips(tmp_path)) == []


def test_hockey_fight_layout(tmp_path: Path) -> None:
    (tmp_path / "hockey_fight" / "fights" / "fight").mkdir(parents=True)
    (tmp_path / "hockey_fight" / "fights" / "nofight").mkdir(parents=True)
    _touch(tmp_path / "hockey_fight" / "fights" / "fight", "f001.avi")
    _touch(tmp_path / "hockey_fight" / "fights" / "nofight", "n001.avi")

    clips = list(HockeyFightAdapter().iter_clips(tmp_path))
    assert {clip.native_label for clip in clips} == {"fight", "nofight"}
    assert all(clip.source == "hockey_fight" for clip in clips)


def test_adapter_unified_mapping() -> None:
    assert RWF2000Adapter().unified_label("Fight") == DatasetClass.AGGRESSIVE
    assert RWF2000Adapter().unified_label("nofight") == DatasetClass.NORMAL
    assert HockeyFightAdapter().unified_label("fight") == DatasetClass.AGGRESSIVE
    assert HockeyFightAdapter().unified_label("NonFight") == DatasetClass.NORMAL


def test_adapter_unified_rejects_unknown_native() -> None:
    with pytest.raises(ValueError, match="unknown native label"):
        RWF2000Adapter().unified_label("strolling")


def test_clip_dataclass(tmp_path: Path) -> None:
    clip = Clip(
        video_path=tmp_path, native_label="fight", source="rwf2000", video_id="v"
    )
    assert clip.video_path == tmp_path
    assert clip.video_id == "v"


def test_enabled_sources_excludes_scaffold() -> None:
    assert enabled_sources() == ["rwf2000", "hockey_fight"]


def test_build_adapters_resolves_names() -> None:
    adapters = build_adapters(["rwf2000", "HOCKEY_FIGHT"])
    assert [adapter.source for adapter in adapters] == ["rwf2000", "hockey_fight"]


def test_build_adapters_unknown_source() -> None:
    with pytest.raises(ValueError, match="unknown dataset source"):
        build_adapters(["rlvs"])


def test_ucf101_scaffold_raises_until_mapped(tmp_path: Path) -> None:
    adapter = UCF101Adapter()
    assert adapter.enabled is False
    with pytest.raises(NotImplementedError, match="TODO\\(implementation\\)"):
        adapter.iter_clips(tmp_path)
    with pytest.raises(NotImplementedError, match="TODO\\(implementation\\)"):
        adapter.unified_label("Hugging")
