"""Behavioral-dataset data contracts.

A :class:`DatasetRecord` is one labeling unit: a per-interaction-window feature
vector (``FeatureRecord`` era rows, eventually the full canonical catalog), its
behavior label, and the provenance metadata that powers video-level splits and
audit (``docs/dataset_strategy.md``, ``docs/labeling_guide.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from safewatch.core.constants import BehaviorClass


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One scored interaction window as a (features, label, metadata) triple."""

    names: tuple[str, ...]
    values: tuple[float, ...]
    label: BehaviorClass | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) "
                "must have equal length"
            )

    @property
    def dimension(self) -> int:
        return len(self.values)

    @property
    def video_id(self) -> str | None:
        value = self.metadata.get("video_id")
        return value if isinstance(value, str) else None

    @property
    def window_start(self) -> int | None:
        value = self.metadata.get("window_start")
        return value if isinstance(value, int) else None

    @property
    def window_end(self) -> int | None:
        value = self.metadata.get("window_end")
        return value if isinstance(value, int) else None


@dataclass(frozen=True, slots=True)
class Annotation:
    """One manual annotation in the ``docs/labeling_guide.md`` JSON format."""

    id: str
    label: BehaviorClass
    window_start: int
    window_end: int
    tracks: tuple[str, ...] = ()
    fps: int = 30
    annotator: str = ""
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> Annotation:
        missing = {"id", "window_start", "window_end"} - set(raw)
        if missing:
            raise ValueError(f"annotation missing required fields: {sorted(missing)}")
        tracks_raw = raw.get("tracks", ())
        tracks = (
            tuple(tracks_raw)
            if isinstance(tracks_raw, (tuple, list))
            else tuple(str(tracks_raw).split(","))
            if tracks_raw
            else ()
        )
        return cls(
            id=str(raw["id"]),
            label=_parse_annotation_label(raw["label"]),
            window_start=int(str(raw["window_start"])),
            window_end=int(str(raw["window_end"])),
            tracks=tracks,
            fps=int(str(raw["fps"])) if "fps" in raw else 30,
            annotator=str(raw["annotator"]) if "annotator" in raw else "",
            confidence=float(str(raw["confidence"])) if "confidence" in raw else 1.0,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "fps": self.fps,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "tracks": list(self.tracks),
            "label": self.label.value,
            "annotator": self.annotator,
            "confidence": self.confidence,
        }


def _parse_annotation_label(raw: object) -> BehaviorClass:
    if isinstance(raw, BehaviorClass):
        return raw
    if isinstance(raw, bool):
        raise TypeError(f"invalid label: {raw!r}")
    if isinstance(raw, int):
        return BehaviorClass(raw)
    if isinstance(raw, str):
        try:
            return BehaviorClass(int(raw.strip()))
        except ValueError:
            try:
                return BehaviorClass[raw.strip().upper()]
            except KeyError:
                raise ValueError(f"invalid label: {raw!r}") from None
    raise ValueError(f"invalid label: {raw!r}")


__all__ = ["Annotation", "DatasetRecord"]
