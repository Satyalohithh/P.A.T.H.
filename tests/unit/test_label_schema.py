"""Tests for the behavior label schema."""

from __future__ import annotations

import pytest

from safewatch.behavior.label_schema import (
    LABEL_NAMES,
    LABELS,
    label_name,
    label_value,
    parse_label,
)
from safewatch.core.constants import BehaviorClass


def test_labels_are_frozen_four_class() -> None:
    assert LABELS == (
        BehaviorClass.NORMAL,
        BehaviorClass.PLAYFUL,
        BehaviorClass.SUSPICIOUS,
        BehaviorClass.AGGRESSIVE,
    )
    assert list(LABEL_NAMES.values()) == [
        "normal",
        "playful",
        "suspicious",
        "aggressive",
    ]


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (BehaviorClass.NORMAL, "normal"),
        (BehaviorClass.PLAYFUL, "playful"),
        (BehaviorClass.SUSPICIOUS, "suspicious"),
        (BehaviorClass.AGGRESSIVE, "aggressive"),
    ],
)
def test_label_name(cls: BehaviorClass, expected: str) -> None:
    assert label_name(cls) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("normal", BehaviorClass.NORMAL),
        ("PLAYFUL", BehaviorClass.PLAYFUL),
        (" Suspicious ", BehaviorClass.SUSPICIOUS),
        ("aggressive", BehaviorClass.AGGRESSIVE),
        ("0", BehaviorClass.NORMAL),
        ("3", BehaviorClass.AGGRESSIVE),
        (3, BehaviorClass.AGGRESSIVE),
        (BehaviorClass.PLAYFUL, BehaviorClass.PLAYFUL),
    ],
)
def test_parse_label_accepts_valid_forms(raw: object, expected: BehaviorClass) -> None:
    assert parse_label(raw) == expected


@pytest.mark.parametrize("raw", ["unknown", "4", "-1", "", None, 1.5, ["normal"]])
def test_parse_label_rejects_invalid(raw: object) -> None:
    with pytest.raises(ValueError, match="invalid behavior label"):
        parse_label(raw)


def test_parse_label_rejects_bool() -> None:
    with pytest.raises(TypeError, match="invalid behavior label"):
        parse_label(True)


def test_label_value() -> None:
    assert label_value("aggressive") == BehaviorClass.AGGRESSIVE.value
