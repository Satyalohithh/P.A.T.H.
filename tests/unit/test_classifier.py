"""Tests for the XGBoost baseline behavior classifier."""

from __future__ import annotations

import numpy as np
import pytest

from safewatch.behavior import BehaviorClassifier
from safewatch.core.constants import BehaviorClass
from safewatch.core.schemas.features import FeatureVector

NAMES = ("f0", "f1", "f2", "f3")


@pytest.fixture(scope="module")
def synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    features: list[list[float]] = []
    labels: list[int] = []
    centers = [(-3.0, -3.0), (-1.0, 2.0), (1.0, -1.0), (3.0, 3.0)]
    for label, (cx, cy) in enumerate(centers):
        for _ in range(60):
            features.append(
                [
                    cx + rng.normal(0, 0.6),
                    cy + rng.normal(0, 0.6),
                    float(rng.normal()),
                    float(rng.normal()),
                ]
            )
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


@pytest.fixture(scope="module")
def fitted(synthetic: tuple[np.ndarray, np.ndarray]) -> BehaviorClassifier:
    features, labels = synthetic
    return BehaviorClassifier(num_classes=4, feature_names=NAMES).fit(
        features, labels, feature_names=NAMES, n_estimators=30, max_depth=3
    )


def test_predict_returns_behavior_class_with_normalized_probabilities(
    fitted: BehaviorClassifier, synthetic: tuple[np.ndarray, np.ndarray]
) -> None:
    features, _ = synthetic
    result = fitted.predict(FeatureVector(names=NAMES, values=tuple(features[0])))
    assert isinstance(result.label, BehaviorClass)
    assert len(result.probabilities) == 4
    assert sum(result.probabilities) == pytest.approx(1.0, abs=1e-5)
    assert result.label.value == int(np.argmax(result.probabilities))


def test_predict_proba(fitted: BehaviorClassifier) -> None:
    probabilities = fitted.predict_proba(
        FeatureVector(names=NAMES, values=(0.0, 0.0, 0.0, 0.0))
    )
    assert len(probabilities) == 4
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)


def test_predict_batch_shapes(fitted: BehaviorClassifier) -> None:
    batch = np.ones((7, 4), dtype=np.float32)
    labels, probabilities = fitted.predict_batch(batch)
    assert labels.shape == (7,)
    assert probabilities.shape == (7, 4)
    assert (probabilities.sum(axis=1) - 1.0).max() < 1e-4


def test_fit_rejects_misaligned_feature_names(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    features, labels = synthetic
    with pytest.raises(ValueError, match="feature_names"):
        BehaviorClassifier().fit(features, labels, feature_names=("x", "y"))


def test_fit_rejects_ragged_row_counts(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    features, labels = synthetic
    with pytest.raises(ValueError, match="must match"):
        BehaviorClassifier().fit(features, labels[:-1])


def test_fit_rejects_out_of_range_labels(
    synthetic: tuple[np.ndarray, np.ndarray],
) -> None:
    features, _ = synthetic
    with pytest.raises(ValueError, match="in \\[0,.*\\)"):
        BehaviorClassifier(num_classes=2).fit(
            features, np.array([0, 5] * (len(features) // 2), dtype=int)
        )


def test_fit_rejects_single_class(synthetic: tuple[np.ndarray, np.ndarray]) -> None:
    features, _ = synthetic
    with pytest.raises(ValueError, match="at least two labels"):
        BehaviorClassifier().fit(features, np.zeros(len(features), dtype=int))


def test_fit_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        BehaviorClassifier().fit(np.empty((0, 4), dtype=np.float32), [])


def test_predict_requires_model() -> None:
    classifier = BehaviorClassifier()
    with pytest.raises(RuntimeError, match="not loaded"):
        classifier.predict(FeatureVector(names=NAMES, values=(0.0, 0.0, 0.0, 0.0)))


def test_predict_enforces_feature_alignment(fitted: BehaviorClassifier) -> None:
    with pytest.raises(ValueError, match="do not match"):
        fitted.predict(FeatureVector(names=("x", "y"), values=(0.0, 0.0)))


def test_save_load_round_trip(
    fitted: BehaviorClassifier, tmp_path, synthetic: tuple[np.ndarray, np.ndarray]
) -> None:
    features, _ = synthetic
    model_path = tmp_path / "baseline.json"
    fitted.save(model_path)
    assert model_path.is_file()
    assert tmp_path.joinpath("baseline.json.meta.json").is_file()

    restored = BehaviorClassifier(model_path=str(model_path))
    restored.load()
    assert restored.feature_names == NAMES
    assert restored.num_classes == 4
    before = fitted.predict_batch(features[:50])[0]
    after = restored.predict_batch(features[:50])[0]
    assert (before == after).all()


def test_load_without_model_path_raises() -> None:
    with pytest.raises(ValueError, match="model_path"):
        BehaviorClassifier().load()


def test_unload_releases_model(fitted: BehaviorClassifier) -> None:
    assert fitted.is_loaded
    fitted.unload()
    assert not fitted.is_loaded
    with pytest.raises(RuntimeError):
        fitted.predict_batch(np.ones((1, 4), dtype=np.float32))
