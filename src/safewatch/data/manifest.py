"""Manifest CSV tooling for indexed raw datasets.

The manifest is the single-source table consumed by splitting, statistics,
and experiments: one row per clip with its unified label and source. Video
paths are stored relative to the dataset root (``datasets/raw``) so a
manifest stays portable across machines.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import pandas as pd

from safewatch.data.adapters import build_adapters
from safewatch.data.adapters.base import Clip
from safewatch.data.labels import parse_unified_label

logger = logging.getLogger("safewatch.data.manifest")

MANIFEST_COLUMNS = ("video_path", "label", "source_dataset")
"""Required manifest columns documented in ``datasets/README.md``."""

PROBE_COLUMNS = ("duration_s", "fps")
"""Optional columns populated by the probe (metadata only, may be NaN)."""

MIN_PROBE_BYTES = 4096
"""Probing skipped for files this small (avoids noisy decoder failures)."""


def _probe_video(path: Path) -> tuple[float, float] | None:
    """Return ``(duration_s, fps)`` or ``None`` when the clip is unreadable."""

    try:
        if not path.is_file() or path.stat().st_size < MIN_PROBE_BYTES:
            return None
        capture = cv2.VideoCapture(str(path))
        try:
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
    except (cv2.error, OSError):
        return None
    if frame_count <= 0 or fps <= 0:
        return None
    return (frame_count / fps, fps)


def build_manifest(
    raw_root: Path,
    sources: Iterable[str],
    output: Path,
    *,
    limit_per_source: int | None = None,
    probe: bool = True,
) -> pd.DataFrame:
    """Index clips from ``sources`` into ``output`` (a CSV manifest).

    File existence is enforced; unreadable probes degrade to NaN rather than
    aborting. ``limit_per_source`` caps how many clips each source contributes
    (useful for pipeline smoke runs).
    """

    rows: list[dict[str, object]] = []
    seen: set[Path] = set()
    for adapter in build_adapters(sources):
        emitted = 0
        for clip in adapter.iter_clips(raw_root):
            if limit_per_source is not None and emitted >= limit_per_source:
                break
            if not clip.video_path.is_file():
                logger.warning("missing video, skipped: %s", clip.video_path)
                continue
            resolved = clip.video_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            label = parse_unified_label(adapter.unified_label(clip.native_label))
            row: dict[str, object] = {
                "video_path": _portable_path(clip, raw_root),
                "label": label.value,
                "source_dataset": clip.source,
            }
            if probe:
                duration, fps = _probe_video(clip.video_path) or (
                    float("nan"),
                    float("nan"),
                )
                row["duration_s"] = duration
                row["fps"] = fps
            rows.append(row)
            emitted += 1

    columns = [*MANIFEST_COLUMNS, *PROBE_COLUMNS] if probe else list(MANIFEST_COLUMNS)
    manifest = pd.DataFrame(rows, columns=columns)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    logger.info("manifest written: %s (%d clips)", output, len(manifest))
    return manifest


def _portable_path(clip: Clip, raw_root: Path) -> str:
    try:
        relative = clip.video_path.relative_to(raw_root)
    except ValueError:
        relative = Path(os.path.basename(str(clip.video_path)))
    return relative.as_posix()


def load_manifest(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a manifest CSV back as a DataFrame with stable dtypes."""

    frame = pd.read_csv(
        path,
        dtype={
            "video_path": str,
            "label": str,
            "source_dataset": str,
        },
    )
    missing = [col for col in MANIFEST_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"manifest {path} missing required columns: {missing}")
    return frame


__all__ = [
    "MANIFEST_COLUMNS",
    "MIN_PROBE_BYTES",
    "PROBE_COLUMNS",
    "build_manifest",
    "load_manifest",
]
