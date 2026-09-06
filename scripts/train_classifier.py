#!/usr/bin/env python3
"""Train the Phase-5 baseline behavior classifier (XGBoost, tabular).

Loads a labeled CSV/Parquet dataset, splits it into train/test at the video
level (``docs/dataset_strategy.md``), fits ``multi:softprob`` XGBoost with the
hyperparameters from ``configs/models/xgboost_baseline.yaml``, and persists:

- ``<output-dir>/xgb_baseline.json`` + ``.meta.json`` (the saved model)
- ``<output-dir>/metrics_report.json`` (train + test metrics)
- ``<output-dir>/confusion_matrix.png``
- ``<output-dir>/feature_importance.csv`` and ``feature_importance.png``

Usage:
    uv run python scripts/train_classifier.py --data datasets/processed/labeled.parquet
    uv run python scripts/train_classifier.py --data labeled.csv \\
        --output-dir models/checkpoints --test-size 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from safewatch.behavior import BehaviorClassifier, XGBoostBaselineConfig
from safewatch.behavior.dataset import (
    DatasetRecord,
    feature_names,
    load_dataset,
    train_test_split,
    validate_labels,
)
from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.core.constants import BehaviorClass
from safewatch.evaluation.confusion_analysis import save_confusion_matrix
from safewatch.evaluation.feature_importance import FeatureImportance
from safewatch.evaluation.metrics import Metrics, MetricsReport
from safewatch.utils.logging import configure_logging, get_logger

MODEL_NAME = "xgb_baseline.json"
METRICS_REPORT = "metrics_report.json"
CONFUSION_MATRIX_PNG = "confusion_matrix.png"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = "feature_importance.png"


def _matrix(records: list[DatasetRecord]) -> np.ndarray:
    return np.asarray([record.values for record in records], dtype=np.float32)


def _labels(records: list[DatasetRecord]) -> list[int]:
    return [record.label.value for record in records]


def _report_to_dict(report: MetricsReport) -> dict[str, object]:
    """Serialize a :class:`MetricsReport` to JSON-compatible structures."""

    per_class = {
        str(label): {
            "precision": round(scores.precision, 4),
            "recall": round(scores.recall, 4),
            "f1": round(scores.f1, 4),
            "support": scores.support,
        }
        for label, scores in report.per_class.items()
    }
    return {
        "accuracy": round(report.accuracy, 4),
        "macro_f1": round(report.macro_f1, 4),
        "per_class": per_class,
        "confusion": report.confusion.tolist(),
        "class_order": [LABEL_NAMES[cls] for cls in BehaviorClass],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="labeled CSV/Parquet dataset")
    parser.add_argument(
        "--config",
        default="configs/models/xgboost_baseline.yaml",
        help="baseline hyperparameter configuration",
    )
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints",
        help="directory for the model and evaluation artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-video-split",
        action="store_true",
        help="split per row instead of per video (default is video-level)",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.train")

    config = XGBoostBaselineConfig.from_file(args.config)
    records = validate_labels(load_dataset(args.data))
    split = train_test_split(
        records,
        test_size=args.test_size,
        seed=args.seed,
        by_video=not args.no_video_split,
    )
    names = feature_names(records)
    if len(split.train) == 0 or len(split.test) == 0:
        raise SystemExit("train/test split produced an empty side; adjust --test-size")

    classifier = BehaviorClassifier(
        num_classes=len(BehaviorClass), feature_names=names
    ).fit(
        _matrix(split.train),
        _labels(split.train),
        feature_names=names,
        **config.to_hyperparameters(),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_NAME
    classifier.save(model_path)

    metrics = Metrics(num_classes=len(BehaviorClass))
    train_labels = _labels(split.train)
    test_labels = _labels(split.test)
    train_report = metrics.report(
        train_labels, classifier.predict_batch(_matrix(split.train))[0].tolist()
    )
    test_report = metrics.report(
        test_labels, classifier.predict_batch(_matrix(split.test))[0].tolist()
    )
    artifacts = {
        "model": str(model_path),
        "train": _report_to_dict(train_report),
        "test": _report_to_dict(test_report),
        "split": {
            "by_video": not args.no_video_split,
            "seed": args.seed,
            "test_size": args.test_size,
            "train_rows": len(split.train),
            "test_rows": len(split.test),
        },
        "features": list(names),
    }
    with open(output_dir / METRICS_REPORT, "w", encoding="utf-8") as handle:
        json.dump(artifacts, handle, indent=2)

    save_confusion_matrix(test_report.confusion, str(output_dir / CONFUSION_MATRIX_PNG))

    importance = FeatureImportance(names)
    gain = importance.gain_importance(classifier.model)
    importance_frame = pd.DataFrame(
        {"feature": list(gain), "gain": list(gain.values())}
    ).sort_values("gain", ascending=False)
    importance_frame.to_csv(output_dir / FEATURE_IMPORTANCE_CSV, index=False)
    importance.plot_importance(
        classifier.model, output_dir / FEATURE_IMPORTANCE_PNG, top_k=14
    )

    logger.info(
        "baseline training complete",
        extra={
            "safewatch_extra": {
                "train_f1": round(train_report.macro_f1, 4),
                "test_f1": round(test_report.macro_f1, 4),
                "test_accuracy": round(test_report.accuracy, 4),
                "output_dir": str(output_dir),
            }
        },
    )


if __name__ == "__main__":
    main()
