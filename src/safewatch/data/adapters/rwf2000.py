"""RWF-2000 adapter.

Public layout (mirrored into ``datasets/raw/rwf_2000/``)::

    rwf_2000/
      Fight/     fi*.mp4
      NonFight/  nf*.mp4

The dataset is binary (fight / non-fight), so only ``aggressive`` and
``normal`` are populated.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from safewatch.data.adapters.base import (
    Clip,
    DatasetAdapter,
    iter_videos,
    stable_id,
)
from safewatch.data.labels import DatasetClass

FIGHT_TOKENS = {"fight", "fights", "fighting"}
NOFIGHT_TOKENS = {
    "nofight",
    "nofights",
    "nonfight",
    "non_fight",
    "non-fight",
    "no_fight",
}


class RWF2000Adapter(DatasetAdapter):
    """Indexes RWF-2000 real-world fight clips."""

    source = "rwf2000"
    raw_dir_name = "rwf_2000"
    native_classes = ("fight", "nofight")

    @property
    def enabled(self) -> bool:
        return True

    def _native_to_unified(self) -> dict[tuple[str, ...], DatasetClass]:
        return {
            ("fight", "fights", "fighting"): DatasetClass.AGGRESSIVE,
            (
                "nofight",
                "nofights",
                "nonfight",
                "non_fight",
                "non-fight",
                "no_fight",
            ): DatasetClass.NORMAL,
        }

    def iter_clips(self, raw_root: Path) -> Iterator[Clip]:
        base = raw_root / self.raw_dir_name
        if not base.is_dir():
            return
        for video in iter_videos(base):
            label = self._label_from_parents(video, base)
            if label is None:
                continue
            rel = video.relative_to(raw_root)
            yield Clip(
                video_path=video,
                native_label=label,
                source=self.source,
                video_id=stable_id(self.source, rel),
            )

    def _label_from_parents(self, video: Path, base: Path) -> str | None:
        for part in reversed(video.relative_to(base).parts[:-1]):
            token = part.strip().lower()
            if token in FIGHT_TOKENS:
                return "fight"
            if token in NOFIGHT_TOKENS:
                return "nofight"
        return None
