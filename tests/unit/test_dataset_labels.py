"""Tests for the unified acquisition label schema."""

from __future__ import annotations

import pytest

from safewatch.core.constants import BehaviorClass
from safewatch.data.labels import (
    UNIFIED_ORDER,
    DatasetClass,
    active_classes,
    from_behavior_class,
    label_name,
    label_value,
    missing_classes,
    parse_unified_label,
    to_behavior_class,
)


def test_unified_order_is_three_class() -> None:
    assert UNIFIED_ORDER == (
        DatasetClass.NORMAL,
        DatasetClass.PLAYFUL,
        DatasetClass.AGGRESSIVE,
    )


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (DatasetClass.NORMAL, "normal"),
        (DatasetClass.PLAYFUL, "playful"),
        (DatasetClass.AGGRESSIVE, "aggressive"),
    ],
)
def test_label_name(cls: DatasetClass, expected: str) -> None:
    assert label_name(cls) == expected
    assert label_value(cls) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("normal", DatasetClass.NORMAL),
        ("PLAYFUL", DatasetClass.PLAYFUL),
        (" Aggressive ", DatasetClass.AGGRESSIVE),
        (DatasetClass.NORMAL, DatasetClass.NORMAL),
    ],
)
def test_parse_unified_label_accepts_valid_forms(
    raw: object, expected: DatasetClass
) -> None:
    assert parse_unified_label(raw) == expected


@pytest.mark.parametrize("raw", ["unknown", "suspicious", "", None, 1.5, 2, ["normal"]])
def test_parse_unified_label_rejects_invalid(raw: object) -> None:
    with pytest.raises(ValueError, match="invalid unified label"):
        parse_unified_label(raw)


def test_parse_unified_label_rejects_bool() -> None:
    with pytest.raises(TypeError, match="invalid unified label"):
        parse_unified_label(True)


def test_to_behavior_class_mapping() -> None:
    assert to_behavior_class(DatasetClass.NORMAL) == BehaviorClass.NORMAL
    assert to_behavior_class(DatasetClass.PLAYFUL) == BehaviorClass.PLAYFUL
    assert to_behavior_class(DatasetClass.AGGRESSIVE) == BehaviorClass.AGGRESSIVE


def test_from_behavior_class_roundtrip() -> None:
    for cls in DatasetClass:
        assert from_behavior_class(to_behavior_class(cls)) == cls


def test_from_behavior_class_suspicious_is_none() -> None:
    assert from_behavior_class(BehaviorClass.SUSPICIOUS) is None


def test_active_classes_in_frozen_order() -> None:
    labels = [DatasetClass.AGGRESSIVE, DatasetClass.NORMAL, DatasetClass.NORMAL]
    assert active_classes(labels) == (DatasetClass.NORMAL, DatasetClass.AGGRESSIVE)


def test_missing_classes_complements_active() -> None:
    active = (DatasetClass.NORMAL, DatasetClass.AGGRESSIVE)
    assert missing_classes(active) == (DatasetClass.PLAYFUL,)
    assert active_classes(missing_classes(active)) == (DatasetClass.PLAYFUL,)


def test_active_classes_rejects_non_members() -> None:
    with pytest.raises(TypeError, match="DatasetClass"):
        active_classes([BehaviorClass.NORMAL])  # type: ignore[list-item]
