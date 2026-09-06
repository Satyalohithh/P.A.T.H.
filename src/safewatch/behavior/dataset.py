"""Behavioral dataset: CSV/Parquet persistence, labeling, and splitting.

Datasets are stored in a wide tabular layout: one row per scored interaction
window, with ``label`` plus the standard provenance columns
(``id``, ``window_start``, ``window_end``, ``tracks``, ``confidence``,
``annotator``, ``fps``) and one column per feature name (in ``names`` order).
Unrecognized metadata keys are preserved in a JSON ``_extra`` column so
round-trips are lossless.

Feature columns are identified as every column not in the reserved set, so
datasets built from new feature catalogs (e.g. the full 36 canonical features)
load without any format change.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

from safewatch.behavior.label_schema import label_name, parse_label
from safewatch.core.constants import BehaviorClass
from safewatch.core.schemas.dataset import Annotation, DatasetRecord

RESERVED_COLUMNS: tuple[str, ...] = (
    "label",
    "id",
    "window_start",
    "window_end",
    "tracks",
    "confidence",
    "annotator",
    "fps",
    "_extra",
)
"""Columns that are not feature columns in the wide layout."""

EXTRA_COLUMN = "_extra"


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Train/test partition of a labeled dataset."""

    train: tuple[DatasetRecord, ...]
    test: tuple[DatasetRecord, ...]

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)


def feature_names(records: list[DatasetRecord]) -> tuple[str, ...]:
    """Return the shared feature-name order, raising if it is inconsistent."""

    if not records:
        raise ValueError("cannot infer feature names from an empty dataset")
    expected = records[0].names
    for record in records[1:]:
        if record.names != expected:
            raise ValueError(
                "inconsistent feature names across dataset rows; all rows must "
                "share the same feature order"
            )
    return expected


def save_dataset(records: list[DatasetRecord], path: str | Path) -> None:
    """Serialize ``records`` to CSV or Parquet based on the file extension."""

    if not records:
        raise ValueError(
            "cannot serialize an empty dataset: feature names are unavailable"
        )
    rows: list[dict[str, object]] = []
    for record in records:
        tracks_raw = record.metadata.get("tracks", ())
        if isinstance(tracks_raw, (tuple, list)):
            tracks: object = ",".join(str(t) for t in tracks_raw)
        else:
            tracks = tracks_raw
        row: dict[str, object] = {
            "label": label_name(record.label) if record.label is not None else None,
            "id": record.metadata.get("video_id"),
            "window_start": record.metadata.get("window_start"),
            "window_end": record.metadata.get("window_end"),
            "tracks": tracks,
            "confidence": record.metadata.get("confidence"),
            "annotator": record.metadata.get("annotator"),
            "fps": record.metadata.get("fps"),
        }
        extra = {
            key: value
            for key, value in record.metadata.items()
            if key
            not in {
                "video_id",
                "window_start",
                "window_end",
                "tracks",
                "confidence",
                "annotator",
                "fps",
            }
        }
        if extra:
            row[EXTRA_COLUMN] = json.dumps(extra)
        for name, value in zip(record.names, record.values, strict=True):
            row[name] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    suffix = _check_suffix(path).lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def load_dataset(path: str | Path) -> list[DatasetRecord]:
    """Load a dataset written by :func:`save_dataset` from CSV or Parquet."""

    suffix = _check_suffix(path).lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    else:
        frame = pd.read_parquet(path)
    records: list[DatasetRecord] = []
    for row in frame.itertuples(index=False, name=None):
        row_dict = dict(zip(frame.columns, row, strict=True))
        _load_record(row_dict, records)
    return records


def _load_record(row: dict[str, object], records: list[DatasetRecord]) -> None:
    names = tuple(name for name in row if name not in RESERVED_COLUMNS)
    values = tuple(_as_float(row[name]) for name in names)
    label_raw = row.get("label")
    label: BehaviorClass | None = None
    if label_raw is not None and not _is_blank(label_raw):
        label = parse_label(label_raw)
    metadata: dict[str, object] = {}
    video_id = _unblank(row.get("id"))
    if video_id is not None:
        metadata["video_id"] = video_id
    window_start = _as_optional_int(row.get("window_start"))
    if window_start is not None:
        metadata["window_start"] = window_start
    window_end = _as_optional_int(row.get("window_end"))
    if window_end is not None:
        metadata["window_end"] = window_end
    tracks = _unblank(row.get("tracks"))
    if tracks is not None:
        text = str(tracks)
        metadata["tracks"] = tuple(
            part.strip() for part in text.split(",") if part.strip()
        )
    for key in ("confidence", "annotator", "fps"):
        value = _unblank(row.get(key))
        if value is not None:
            metadata[key] = value
    extra_raw = _unblank(row.get(EXTRA_COLUMN))
    if extra_raw is not None:
        unpacked = json.loads(str(extra_raw))
        if not isinstance(unpacked, dict):
            raise ValueError(f"invalid {EXTRA_COLUMN!r} value in dataset")
        metadata.update({str(key): value for key, value in unpacked.items()})
    records.append(
        DatasetRecord(names=names, values=values, label=label, metadata=metadata)
    )


def _as_float(value: object) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def _as_optional_int(value: object) -> object:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return int(text) if text else None
    except ValueError:
        return None


def _unblank(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        return None
    if _is_blank(value):
        return None
    return value


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return bool(isinstance(value, str) and not value.strip())


def _check_suffix(path: str | Path) -> str:
    suffix = Path(path).suffix
    if suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(
            f"unsupported dataset format {suffix!r}; expected '.csv' or '.parquet'"
        )
    return suffix


def assign_labels(
    records: list[DatasetRecord],
    annotations: list[Annotation] | dict[str, object],
) -> list[DatasetRecord]:
    """Attach labels to unlabeled rows from annotations keyed by video id.

    ``annotations`` is either a list of :class:`Annotation` objects or a
    mapping from video id to a label-coercible value (:func:`parse_label`).
    Rows whose ``video_id`` has no annotation are left unlabeled; annotations
    against unknown video ids raise :class:`ValueError` to catch typos.
    """

    if isinstance(annotations, list):
        label_by_id = {annotation.id: annotation.label for annotation in annotations}
    else:
        label_by_id = {
            str(key): parse_label(value) for key, value in annotations.items()
        }
    unknown = set(label_by_id) - {r.metadata.get("video_id") for r in records}
    if unknown:
        raise ValueError(f"annotations reference unknown video ids: {sorted(unknown)}")
    labeled: list[DatasetRecord] = []
    for record in records:
        video_id = record.metadata.get("video_id")
        label = label_by_id.get(video_id) if isinstance(video_id, str) else None
        labeled.append(
            DatasetRecord(
                names=record.names,
                values=record.values,
                label=record.label if record.label is not None else label,
                metadata=record.metadata,
            )
        )
    return labeled


def validate_labels(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """Return ``records`` filtered to fully scored rows, raising on gaps.

    Rows missing a label are dropped; if a label cannot be interpreted the
    error is raised, so a silent labelling gap fails loudly.
    """

    missing = [i for i, record in enumerate(records) if record.label is None]
    if missing:
        raise ValueError(
            f"{len(missing)} dataset rows are missing a label "
            f"(rows {missing[:10]}{'...' if len(missing) > 10 else ''})"
        )
    return records


def train_test_split(
    records: list[DatasetRecord],
    test_size: float = 0.2,
    seed: int = 42,
    by_video: bool = True,
) -> DatasetSplit:
    """Partition labeled records into train/test.

    When ``by_video`` is true the split happens at the video level (every row
    of a video lands on one side), stratified by each video's dominant label,
    matching ``docs/dataset_strategy.md``. All rows must be labeled.
    """

    validate_labels(records)
    label_values = np.asarray(
        [
            record.label.value  # type: ignore[union-attr]
            for record in records
        ],
        dtype=np.int64,
    )
    if by_video:
        return _video_level_split(records, label_values, test_size, seed)
    train_idx, test_idx = sk_train_test_split(
        np.arange(len(records)),
        test_size=test_size,
        random_state=seed,
        stratify=label_values,
    )
    return _split_from_index(records, train_idx, test_idx)


def _video_level_split(
    records: list[DatasetRecord],
    label_values: np.ndarray,
    test_size: float,
    seed: int,
) -> DatasetSplit:
    video_ids, indices = np.unique(
        np.asarray([record.metadata.get("video_id") or "" for record in records]),
        return_inverse=True,
    )
    if len(video_ids) < 2:
        raise ValueError("by_video splitting requires at least two videos")
    majority: list[int] = []
    by_index: dict[int, list[int]] = {video: [] for video in range(len(video_ids))}
    for i, video_idx in enumerate(indices):
        by_index[int(video_idx)].append(i)
    for video in range(len(video_ids)):
        counts: Counter[int] = Counter()
        for i in by_index[video]:
            counts[int(label_values[i])] += 1
        majority.append(counts.most_common(1)[0][0])
    majority_array = np.asarray(majority, dtype=np.int64)
    positions = np.arange(len(video_ids))
    try:
        video_train, video_test = sk_train_test_split(
            positions,
            test_size=test_size,
            random_state=seed,
            stratify=majority_array,
        )
    except ValueError as exc:
        raise ValueError(
            "could not build a stratified video-level split: each behavior "
            "class needs enough videos so that the test set can hold every "
            "class. Increase the number of videos or lower 'test_size'."
        ) from exc
    train_idx = [i for j in video_train for i in by_index[int(j)]]
    test_idx = [i for j in video_test for i in by_index[int(j)]]
    return _split_from_index(records, np.asarray(train_idx), np.asarray(test_idx))


def _split_from_index(
    records: list[DatasetRecord],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> DatasetSplit:
    return DatasetSplit(
        train=tuple(records[i] for i in train_idx),
        test=tuple(records[i] for i in test_idx),
    )


__all__ = [
    "EXTRA_COLUMN",
    "RESERVED_COLUMNS",
    "DatasetSplit",
    "assign_labels",
    "feature_names",
    "load_dataset",
    "save_dataset",
    "train_test_split",
    "validate_labels",
]
