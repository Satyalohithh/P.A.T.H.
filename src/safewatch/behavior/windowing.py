"""Turning per-frame :class:`FeatureRecord` streams into scored interaction windows.

Per ``docs/labeling_guide.md`` interactions are annotated over 30-frame
windows. :func:`build_pair_windows` buckets the per-frame Phase-4 records into
non-overlapping windows and emits one 14-dimensional row per (video, window,
track pair): both persons' 5 ego-feature window means followed by the 4
pair-feature window means. NaN values are skipped per feature so a missing
keypoint in a few frames does not void the whole window (XGBoost consumes the
remaining NaNs natively).
"""

from __future__ import annotations

import math
from collections import defaultdict

from safewatch.core.schemas.dataset import DatasetRecord
from safewatch.core.schemas.features import FeatureRecord
from safewatch.features.behavioral_features import (
    EGO_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
)

FEATURE_NAMES_EGO = EGO_FEATURE_NAMES
FEATURE_NAMES_PAIR = PAIR_FEATURE_NAMES


def build_pair_windows(
    records: list[FeatureRecord],
    video_id: str,
    window_frames: int = 30,
) -> list[DatasetRecord]:
    """Aggregate per-frame records into pair-window rows.

    Records are bucketed by ``frame_index // window_frames``. A row is emitted
    for every window where both tracks of a pair contribute records and the
    pair features are present; ego values are the per-track mean of that
    track's ego features, pair values the mean of the shared pair features.
    Records carrying a ``video_id`` are restricted to ``video_id`` so a mixed
    clip stream cannot leak frames from other videos into a window.
    """

    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")

    scoped = [
        record
        for record in records
        if record.video_id is None or record.video_id == video_id
    ]

    ego_sums: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    ego_counts: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    pair_sums: dict[tuple[tuple[int, int], int], dict[str, float]] = defaultdict(dict)
    pair_counts: dict[tuple[tuple[int, int], int], dict[str, int]] = defaultdict(dict)

    for record in scoped:
        window = record.frame_index // window_frames
        if record.pair_track_id is None:
            _fold(
                ego_sums[(record.track_id, window)],
                ego_counts[(record.track_id, window)],
                record.names,
                record.values,
            )
        else:
            pair_key = _pair_key(record.track_id, record.pair_track_id)
            _fold(
                pair_sums[(pair_key, window)],
                pair_counts[(pair_key, window)],
                record.names,
                record.values,
            )

    rows: list[DatasetRecord] = []
    for (pair_key, window), pair_values in sorted(pair_sums.items()):
        if not pair_values:
            continue
        track_a, track_b = pair_key
        ego_a = ego_sums.get((track_a, window))
        ego_b = ego_sums.get((track_b, window))
        if ego_a is None or ego_b is None:
            continue
        names_a = _person_names(FEATURE_NAMES_EGO, "A")
        names_b = _person_names(FEATURE_NAMES_EGO, "B")
        names = (*names_a, *names_b, *FEATURE_NAMES_PAIR)
        values = (
            *(
                _mean(ego_a, ego_counts[(track_a, window)], name)
                for name in FEATURE_NAMES_EGO
            ),
            *(
                _mean(ego_b, ego_counts[(track_b, window)], name)
                for name in FEATURE_NAMES_EGO
            ),
            *(
                _mean(pair_values, pair_counts[(pair_key, window)], name)
                for name in FEATURE_NAMES_PAIR
            ),
        )
        window_start = window * window_frames
        rows.append(
            DatasetRecord(
                names=names,
                values=values,
                label=None,
                metadata={
                    "video_id": video_id,
                    "window_start": window_start,
                    "window_end": window_start + window_frames - 1,
                    "tracks": (str(track_a), str(track_b)),
                },
            )
        )
    return rows


def _person_names(base: tuple[str, ...], person: str) -> tuple[str, ...]:
    return tuple(f"{name}.{person}" for name in base)


def _pair_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _fold(
    sums: dict[str, float],
    counts: dict[str, int],
    names: tuple[str, ...],
    values: tuple[float, ...],
) -> None:
    for name, value in zip(names, values, strict=True):
        if math.isnan(value):
            continue
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1


def _mean(sums: dict[str, float], counts: dict[str, int], name: str) -> float:
    count = counts.get(name, 0)
    if count == 0:
        return float("nan")
    return sums[name] / count


__all__ = [
    "FEATURE_NAMES_EGO",
    "FEATURE_NAMES_PAIR",
    "build_pair_windows",
]
