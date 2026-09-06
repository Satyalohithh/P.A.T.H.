"""Tests for the behavioral dataset schema and persistence."""

from __future__ import annotations

import math

import pytest

from safewatch.behavior.dataset import (
    assign_labels,
    feature_names,
    load_dataset,
    save_dataset,
    train_test_split,
    validate_labels,
)
from safewatch.core.constants import BehaviorClass
from safewatch.core.schemas.dataset import Annotation, DatasetRecord

NAMES = (
    "pose.torso_lean.A",
    "motion.arm_angular_vel.L.A",
    "inter.distance",
)


def _record(
    video_id: str,
    label: BehaviorClass | None = None,
    values: tuple[float, ...] = (1.0, 2.0, float("nan")),
    extra: dict[str, object] | None = None,
) -> DatasetRecord:
    metadata: dict[str, object] = {
        "video_id": video_id,
        "window_start": 0,
        "window_end": 29,
        "tracks": ("1", "2"),
        "confidence": 0.9,
        "annotator": "u1",
        "fps": 30,
    }
    if extra:
        metadata.update(extra)
    return DatasetRecord(names=NAMES, values=values, label=label, metadata=metadata)


def _labeled_records(videos: int = 8, windows: int = 4) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for video in range(videos):
        label = BehaviorClass(video % 4)
        for window in range(windows):
            records.append(
                _record(
                    video_id=f"v{video:02d}",
                    label=label,
                    values=(float(video), float(window), float("nan")),
                )
            )
    return records


def test_names_values_length_must_match() -> None:
    with pytest.raises(ValueError, match="equal length"):
        DatasetRecord(names=("a", "b"), values=(1.0,))


@pytest.mark.parametrize("suffix", ["csv", "parquet"])
def test_round_trip_lossless(tmp_path, suffix: str) -> None:
    records = _labeled_records()
    path = tmp_path / f"dataset.{suffix}"
    save_dataset(records, path)
    loaded = load_dataset(path)
    assert len(loaded) == len(records)
    for original, restored in zip(records, loaded, strict=True):
        assert restored.names == original.names
        assert restored.label == original.label
        assert restored.metadata == original.metadata
        assert _values_equal(restored.values, original.values)


def _values_equal(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        if math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


def test_unknown_extension_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported dataset format"):
        save_dataset(_labeled_records(1, 1), tmp_path / "dataset.txt")
    with pytest.raises(ValueError, match="unsupported dataset format"):
        load_dataset(tmp_path / "dataset.txt")


def test_empty_dataset_serialization_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        save_dataset([], tmp_path / "dataset.csv")


def test_feature_names_inconsistent() -> None:
    records = [
        DatasetRecord(names=("a", "b"), values=(1.0, 2.0), label=BehaviorClass.NORMAL),
        DatasetRecord(names=("a",), values=(1.0,), label=BehaviorClass.NORMAL),
    ]
    with pytest.raises(ValueError, match="inconsistent feature names"):
        feature_names(records)


def test_feature_names_empty() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        feature_names([])


def test_assign_labels_from_annotations() -> None:
    records = [
        _record("v01", label=None),
        _record("v02", label=None),
    ]
    annotations = [
        Annotation(
            id="v01",
            label=BehaviorClass.AGGRESSIVE,
            window_start=0,
            window_end=29,
            confidence=0.8,
        )
    ]
    labeled = assign_labels(records, annotations)
    assert labeled[0].label == BehaviorClass.AGGRESSIVE
    assert labeled[1].label is None


def test_assign_labels_from_mapping() -> None:
    records = [_record("v9", label=None), _record("v10", label=None)]
    labeled = assign_labels(records, {"v9": "playful", "v10": 3})
    assert labeled[0].label == BehaviorClass.PLAYFUL
    assert labeled[1].label == BehaviorClass.AGGRESSIVE


def test_assign_labels_unknown_video_raises() -> None:
    with pytest.raises(ValueError, match="unknown video ids"):
        assign_labels([_record("v1", label=None)], {"ghost": "normal"})


def test_validate_labels_raises_on_missing() -> None:
    records = [_record("v1", label=BehaviorClass.NORMAL), _record("v2", label=None)]
    with pytest.raises(ValueError, match="missing a label"):
        validate_labels(records)


def test_validate_labels_passes_when_complete() -> None:
    records = [_record("v1", label=BehaviorClass.NORMAL)]
    assert validate_labels(records) == records


def test_train_test_split_stratified_deterministic() -> None:
    records = _labeled_records(videos=40, windows=5)
    first = train_test_split(records, test_size=0.25, seed=7, by_video=False)
    second = train_test_split(records, test_size=0.25, seed=7, by_video=False)
    assert [r.metadata["video_id"] for r in first.train] == [
        r.metadata["video_id"] for r in second.train
    ]
    assert 0 < first.test_size < len(records)


def test_train_test_split_respects_test_size() -> None:
    records = _labeled_records(videos=40, windows=5)
    split = train_test_split(records, test_size=0.2, seed=3, by_video=False)
    expected = round(len(records) * 0.2)
    assert abs(split.test_size - expected) <= 1


def test_train_test_split_by_video_prevents_leakage() -> None:
    records = _labeled_records(videos=24, windows=3)
    split = train_test_split(records, test_size=0.25, seed=1, by_video=True)
    train_videos = {r.metadata["video_id"] for r in split.train}
    test_videos = {r.metadata["video_id"] for r in split.test}
    assert train_videos.isdisjoint(test_videos)
    assert len(train_videos) + len(test_videos) == 24


def test_train_test_split_by_video_stratified_by_dominant_label() -> None:
    records = _labeled_records(videos=32, windows=4)
    split = train_test_split(records, test_size=0.25, seed=11, by_video=True)
    train_videos = {r.metadata["video_id"] for r in split.train}
    train_labels = [
        r.label for r in split.train if r.metadata["video_id"] in train_videos
    ]
    dominant = {label.value for label in train_labels}
    assert dominant == {0, 1, 2, 3}


def test_train_test_split_requires_labels() -> None:
    records = [_record("v1", label=None)]
    with pytest.raises(ValueError, match="missing a label"):
        train_test_split(records)


def test_train_test_split_requires_two_videos() -> None:
    records = _labeled_records(videos=1, windows=2)
    with pytest.raises(ValueError, match="at least two videos"):
        train_test_split(records, by_video=True)


def test_annotation_from_dict_and_back() -> None:
    annotation = Annotation.from_dict(
        {
            "id": "rwf-0001",
            "fps": 30,
            "window_start": 42,
            "window_end": 72,
            "tracks": ["a", "b"],
            "label": 1,
            "annotator": "u1",
            "confidence": 0.9,
        }
    )
    assert annotation.label == BehaviorClass.PLAYFUL
    assert annotation.tracks == ("a", "b")
    assert annotation.to_dict()["label"] == 1


def test_annotation_from_dict_label_names_accepted() -> None:
    annotation = Annotation.from_dict(
        {"id": "x", "window_start": 0, "window_end": 29, "label": "aggressive"}
    )
    assert annotation.label == BehaviorClass.AGGRESSIVE


def test_annotation_from_dict_missing_id_raises() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        Annotation.from_dict({"window_start": 0, "window_end": 29, "label": 1})
