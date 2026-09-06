#!/usr/bin/env python3
"""Evaluate a trained baseline classifier on a labeled CSV/Parquet dataset.

Loads the model (XGBoost native JSON, see ``scripts/train_classifier.py``),
predicts every row, and writes the metric set (accuracy, macro-F1, per-class
precision/recall/F1, confusion matrix) to ``<output-dir>/evaluation_report.json``
plus a ``confusion_matrix.png`` render.

Usage:
    uv run python scripts/evaluate_classifier.py \\
        --model models/checkpoints/xgb_baseline.json \\
        --data datasets/processed/labeled.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from safewatch.behavior import BehaviorClassifier
from safewatch.behavior.dataset import DatasetRecord, load_dataset, validate_labels
from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.core.constants import BehaviorClass
from safewatch.evaluation.confusion_analysis import save_confusion_matrix
from safewatch.evaluation.metrics import Metrics, MetricsReport
from safewatch.utils.logging import configure_logging, get_logger

REPORT_NAME = "evaluation_report.json"
CONFUSION_MATRIX_PNG = "confusion_matrix.png"


def _matrix(records: list[DatasetRecord]) -> np.ndarray:
    return np.asarray([record.values for record in records], dtype=np.float32)


def _report_to_dict(report: MetricsReport) -> dict[str, object]:
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
    parser.add_argument("--model", required=True, help="saved XGBoost baseline model")
    parser.add_argument("--data", required=True, help="labeled CSV/Parquet dataset")
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints",
        help="directory for evaluation artifacts",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.evaluation")

    classifier = BehaviorClassifier(model_path=args.model)
    classifier.load()
    records = validate_labels(load_dataset(args.data))
    if classifier.feature_names is not None:
        expected = tuple(classifier.feature_names)
        mismatched = [r for r in records if r.names != expected]
        if mismatched:
            raise SystemExit(
                f"{len(mismatched)} rows do not match the model feature order"
            )
    labels = np.asarray([record.label.value for record in records])
    predictions, _ = classifier.predict_batch(_matrix(records))
    report = Metrics(num_classes=len(BehaviorClass)).report(
        labels.tolist(), predictions.tolist()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": args.model,
        "rows": len(records),
        **(
            {"feature_order": list(classifier.feature_names)}
            if classifier.feature_names is not None
            else {}
        ),
        "metrics": _report_to_dict(report),
    }
    with open(output_dir / REPORT_NAME, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2)
    save_confusion_matrix(report.confusion, str(output_dir / CONFUSION_MATRIX_PNG))

    logger.info(
        "evaluation complete",
        extra={
            "safewatch_extra": {
                "rows": len(records),
                "macro_f1": round(report.macro_f1, 4),
                "accuracy": round(report.accuracy, 4),
                "report": str(output_dir / REPORT_NAME),
                "confusion_png": str(output_dir / CONFUSION_MATRIX_PNG),
            }
        },
    )


if __name__ == "__main__":
    main()
