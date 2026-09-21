"""Phase-6 experiment orchestrator.

Runs the end-to-end experiment pipeline over the indexed raw dataset:

    manifest -> per-clip feature extraction (cached) -> pair windows
    -> active-class BehaviorClassifier (XGBoost) -> evaluation snapshot
    -> feature importance / confusion matrix artifacts -> markdown report

Publications under ``reports/<name>/`` (config, dataset snapshot, metrics,
confusion matrix, feature importance), model output under ``models/checkpoints/``,
feature caches under ``artifacts/features/``, and the assembled labeled dataset
under ``artifacts/dataset/``. ``datasets/raw/`` is never modified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import TypeAdapter
from pydantic_core import ValidationError

from safewatch.behavior.classifier import BehaviorClassifier
from safewatch.behavior.config import XGBoostBaselineConfig
from safewatch.behavior.dataset import (
    assign_labels,
    save_dataset,
    train_test_split,
    validate_labels,
)
from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.behavior.windowing import build_pair_windows
from safewatch.core.constants import BehaviorClass
from safewatch.core.schemas.dataset import DatasetRecord
from safewatch.core.schemas.features import FeatureRecord
from safewatch.data.extraction import (
    extract_clip_features,
    load_records,
    save_records,
)
from safewatch.data.labels import (
    DatasetClass,
    parse_unified_label,
    to_behavior_class,
)
from safewatch.data.manifest import load_manifest
from safewatch.evaluation.confusion_analysis import save_confusion_matrix
from safewatch.evaluation.feature_importance import FeatureImportance
from safewatch.evaluation.metrics import Metrics, MetricsReport
from safewatch.features.behavioral_features import (
    EGO_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
)

DEFAULT_WINDOW_FRAMES = 30
"""Window length used by the dataset generation (matches the label guide)."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Resolved experiment configuration."""

    name: str
    description: str = ""
    seed: int = 42
    manifest: str = "datasets/manifest.csv"
    raw_root: str = "datasets/raw"
    splits_dir: str = "datasets/splits"
    val_size: float = 0.15
    test_size: float = 0.15
    sources: tuple[str, ...] = ()
    window_frames: int = DEFAULT_WINDOW_FRAMES
    frames_cap: int | None = None
    device: str = "auto"
    training_config: str = "configs/models/xgboost_baseline.yaml"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    models_dir: str = "models/checkpoints"
    max_clips: int | None = None
    synthetic_videos: int = 8
    """Per-class clip count used by ``synthetic=True`` smoke runs."""

    @classmethod
    def from_file(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        defaults = {field for field in cls.__dataclass_fields__}
        unknown = set(raw) - defaults
        if unknown:
            raise ValueError(f"unknown experiment config keys: {sorted(unknown)}")
        try:
            return TypeAdapter(ExperimentConfig).validate_python(dict(raw))
        except ValidationError as exc:
            raise ValueError(f"invalid experiment configuration: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ExperimentResults:
    """What an experiment run produced (paths + headline numbers)."""

    cfg: ExperimentConfig
    clips_processed: int
    windows_total: int
    n_train: int
    n_test: int
    active_classes: tuple[BehaviorClass, ...]
    feature_catalog: tuple[str, ...]
    metrics: MetricsReport
    outputs: dict[str, Path]

    def summary(self) -> str:
        classes = ", ".join(LABEL_NAMES[c] for c in self.active_classes)
        return (
            f"experiment={self.cfg.name} classes=[{classes}] "
            f"windows={self.windows_total} train={self.n_train} test={self.n_test} "
            f"accuracy={self.metrics.accuracy:.3f} macro_f1={self.metrics.macro_f1:.3f} "
            f"report={self.outputs['report']}"
        )


def run_experiment(
    cfg: ExperimentConfig,
    *,
    synthetic: bool = False,
    force: bool = False,
    limit_videos: int | None = None,
    skip_extraction: bool = False,
    engine_factory: Any = None,
) -> ExperimentResults:
    """Execute the full experiment; see module docstring for the pipeline.

    ``engine_factory`` is a callable returning a feature engine (tests and
    smoke stubs inject one) while ``synthetic`` replaces extraction with a
    deterministic generator, bypassing ``datasets/raw`` entirely.
    """

    if cfg.test_size <= 0 or cfg.test_size >= 1:
        raise ValueError(f"test_size must be in (0, 1), got {cfg.test_size}")
    raw_root = Path(cfg.raw_root)
    manifest = (
        _select_manifest(cfg, limit_videos)
        if not synthetic
        else _synthetic_manifest(cfg.synthetic_videos)
    )

    report_dir = _prepare_dir(Path(cfg.reports_dir) / cfg.name)
    features_dir = _prepare_dir(Path(cfg.artifacts_dir) / "features")
    dataset_dir = _prepare_dir(Path(cfg.artifacts_dir) / "dataset")
    models_dir = _prepare_dir(Path(cfg.models_dir))

    records_all: list[DatasetRecord] = []
    processed = 0
    for _, row in manifest.iterrows():
        video_id = str(row["video_path"])
        behavior = to_behavior_class(parse_unified_label(row["label"]))
        clip_records = _records_for(
            cfg,
            row,
            video_id,
            raw_root=raw_root,
            features_dir=features_dir,
            synthetic=synthetic,
            force=force,
            skip_extraction=skip_extraction,
            engine_factory=engine_factory,
        )
        labeled = assign_labels(
            build_pair_windows(clip_records, video_id, cfg.window_frames),
            {video_id: behavior},
        )
        records_all.extend(validate_labels(labeled))
        processed += 1

    if not records_all:
        raise ValueError("no labeled windows were produced; check manifest/sources")
    save_dataset(records_all, dataset_dir / f"{cfg.name}.parquet")
    split = train_test_split(
        records_all, test_size=cfg.test_size, seed=cfg.seed, by_video=True
    )
    active = _active_classes(split.train)
    index = {behavior: i for i, behavior in enumerate(active)}
    y_train = np.asarray(
        [_label_index(index, r.label) for r in split.train], dtype=np.int64
    )
    y_test = np.asarray(
        [_label_index(index, r.label) for r in split.test], dtype=np.int64
    )
    X_train = np.asarray([r.values for r in split.train], dtype=np.float32)
    X_test = np.asarray([r.values for r in split.test], dtype=np.float32)
    names = split.train[0].names

    hyperparameters = XGBoostBaselineConfig.from_file(
        cfg.training_config
    ).to_hyperparameters()
    classifier = BehaviorClassifier(num_classes=len(active), feature_names=names)
    classifier.fit(X_train, y_train, feature_names=names, **hyperparameters)

    y_pred, _ = classifier.predict_batch(X_test)
    metrics = Metrics(num_classes=len(active)).report(
        list(y_test), [int(p) for p in y_pred]
    )

    model_path = models_dir / f"xgb_{cfg.name}.json"
    classifier.save(model_path)
    _patch_model_meta(model_path, active)

    importance_raw = FeatureImportance(names).gain_importance(classifier.model)
    importance_path = report_dir / "feature_importance.csv"
    importance_json = report_dir / "feature_importance.json"
    _write_json(
        importance_json,
        {
            name: round(value, 6)
            for name, value in sorted(
                importance_raw.items(), key=lambda kv: kv[1], reverse=True
            )
        },
    )
    pd.DataFrame(
        [
            {"feature": name, "gain": value}
            for name, value in sorted(
                importance_raw.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
    ).to_csv(importance_path, index=False)
    FeatureImportance(names).plot_importance(
        classifier.model, report_dir / "feature_importance.png", top_k=10
    )

    _write_json(report_dir / "confusion_matrix.json", metrics.confusion.tolist())
    save_confusion_matrix(metrics.confusion, str(report_dir / "confusion_matrix.png"))
    _write_metrics_report(report_dir, metrics, active)

    snapshot_manifest = manifest.copy()
    snapshot_manifest.to_csv(report_dir / "manifest_snapshot.csv", index=False)
    config_snapshot = report_dir / "experiment_config.yaml"
    snapshot = {key: _plain(value) for key, value in asdict(cfg).items()}
    with open(config_snapshot, "w", encoding="utf-8") as handle:
        yaml.safe_dump(snapshot, handle, sort_keys=False)

    report_path = _render_report(
        cfg,
        report_dir,
        metrics,
        active,
        processed,
        len(records_all),
        len(split.train),
        len(split.test),
        names,
        importance_raw,
    )

    return ExperimentResults(
        cfg=cfg,
        clips_processed=processed,
        windows_total=len(records_all),
        n_train=len(split.train),
        n_test=len(split.test),
        active_classes=active,
        feature_catalog=names,
        metrics=metrics,
        outputs={
            "report": report_path,
            "model": model_path,
            "dataset": dataset_dir / f"{cfg.name}.parquet",
            "confusion": report_dir / "confusion_matrix.png",
            "importance": importance_path,
        },
    )


def _select_manifest(cfg: ExperimentConfig, limit: int | None) -> pd.DataFrame:
    manifest = load_manifest(cfg.manifest)
    if cfg.sources:
        available = set(manifest["source_dataset"])
        manifest = manifest[manifest["source_dataset"].isin(cfg.sources)]
        missing = set(cfg.sources) - available
        if missing:
            raise ValueError(f"manifest lacks requested sources: {sorted(missing)}")
    if manifest.empty:
        raise ValueError(f"manifest {cfg.manifest} selected no clips")
    if limit is not None:
        manifest = manifest.head(limit)
    return manifest.reset_index(drop=True)


def _records_for(
    cfg: ExperimentConfig,
    row: pd.Series,
    video_id: str,
    *,
    raw_root: Path,
    features_dir: Path,
    synthetic: bool,
    force: bool,
    skip_extraction: bool,
    engine_factory: Any,
) -> list[FeatureRecord]:
    if synthetic:
        seed = int.from_bytes(hashlib.sha1(video_id.encode()).digest()[:8], "big")
        return _synthetic_records(
            video_id, to_behavior_class(parse_unified_label(row["label"])), seed
        )
    cache = features_dir / _cache_name(video_id)
    if cache.is_file() and not force:
        if skip_extraction:
            return load_records(cache)
        cached = load_records(cache)
        if len(cached) > 0:
            return cached
    if skip_extraction:
        raise FileNotFoundError(f"feature cache missing for {video_id}: {cache}")
    resolved = raw_root / str(row["video_path"])
    if not resolved.is_file():
        raise FileNotFoundError(f"video clip not found: {resolved}")
    settings = _experiment_settings(cfg)
    extracted = extract_clip_features(
        resolved,
        settings,
        video_id=video_id,
        frames_cap=cfg.frames_cap,
        engine_factory=engine_factory,
    )
    save_records(extracted, cache)
    return extracted


def _cache_name(video_id: str) -> str:
    safe = video_id.replace("\\", "_").replace("/", "__")
    return f"{safe.replace(' ', '_')}.parquet"


def _synthetic_manifest(videos_per_class: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cls in DatasetClass:
        for index in range(videos_per_class):
            rows.append(
                {
                    "video_path": f"synthetic/{cls.value}_{index}.mp4",
                    "label": cls.value,
                    "source_dataset": "synthetic",
                }
            )
    return pd.DataFrame(rows, columns=("video_path", "label", "source_dataset"))


def _synthetic_records(
    video_id: str, behavior: BehaviorClass, seed: int
) -> list[FeatureRecord]:
    """Deterministically separable per-class record stream (smoke mode)."""

    rng = np.random.default_rng(seed)
    frames = 90
    ego = _ego_base(behavior)
    pair = _pair_base(behavior)
    records: list[FeatureRecord] = []
    for frame in range(frames):
        timestamp = round(frame / 30.0, 6)
        for track_id in (1, 2):
            noise = rng.normal(0.0, 0.05, len(EGO_FEATURE_NAMES))
            values = tuple(float(v + n) for v, n in zip(ego, noise, strict=True))
            records.append(
                FeatureRecord(
                    track_id=track_id,
                    frame_index=frame,
                    timestamp=timestamp,
                    names=EGO_FEATURE_NAMES,
                    values=values,
                    video_id=video_id,
                )
            )
        pair_values = tuple(
            float(v + n) for v, n in zip(pair, rng.normal(0.0, 0.06, 4), strict=True)
        )
        records.append(
            FeatureRecord(
                track_id=1,
                frame_index=frame,
                timestamp=timestamp,
                names=PAIR_FEATURE_NAMES,
                values=pair_values,
                pair_track_id=2,
                video_id=video_id,
            )
        )
    return records


def _ego_base(behavior: BehaviorClass) -> tuple[float, ...]:
    bases = {
        BehaviorClass.NORMAL: (0.18, 0.05, 0.05, 0.04, 0.04),
        BehaviorClass.PLAYFUL: (0.35, 0.35, 0.35, 0.18, 0.18),
        BehaviorClass.AGGRESSIVE: (0.55, 0.28, 0.28, 0.12, 0.12),
    }
    return bases[behavior]


def _pair_base(behavior: BehaviorClass) -> tuple[float, ...]:
    bases = {
        BehaviorClass.NORMAL: (2.5, 0.0, 0.1, 0.25),
        BehaviorClass.PLAYFUL: (1.7, 0.08, 0.45, 0.55),
        BehaviorClass.AGGRESSIVE: (0.8, 0.35, 0.75, 0.45),
    }
    return bases[behavior]


def _active_classes(
    records: Iterable[DatasetRecord],
) -> tuple[BehaviorClass, ...]:
    present = sorted({int(r.label) for r in records if r.label is not None})
    if len(present) < 2:
        raise ValueError("training data must contain at least two behavior classes")
    return tuple(BehaviorClass(value) for value in present)


def _label_index(index: dict[BehaviorClass, int], label: BehaviorClass | None) -> int:
    if label is None:
        raise ValueError("dataset contains an unlabeled window at experiment time")
    return index[label]


def _patch_model_meta(model_path: Path, active: tuple[BehaviorClass, ...]) -> None:
    meta_path = Path(str(model_path) + ".meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta: dict[str, Any] = json.load(handle)
    meta["labels"] = [LABEL_NAMES[cls] for cls in active]
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _write_metrics_report(
    report_dir: Path, metrics: MetricsReport, active: tuple[BehaviorClass, ...]
) -> None:
    per_class = {
        LABEL_NAMES[active[int(cls)]]: {
            "precision": scores.precision,
            "recall": scores.recall,
            "f1": scores.f1,
            "support": scores.support,
        }
        for cls, scores in metrics.per_class.items()
    }
    _write_json(
        report_dir / "metrics_report.json",
        {
            "accuracy": metrics.accuracy,
            "macro_f1": metrics.macro_f1,
            "num_classes": metrics.num_classes,
            "active_classes": [LABEL_NAMES[cls] for cls in active],
            "per_class": per_class,
        },
    )


def _render_report(
    cfg: ExperimentConfig,
    report_dir: Path,
    metrics: MetricsReport,
    active: tuple[BehaviorClass, ...],
    clips: int,
    windows: int,
    n_train: int,
    n_test: int,
    feature_catalog: tuple[str, ...],
    importance: dict[str, float],
) -> Path:
    per_class = "\n".join(
        f"| {name} | {metrics.per_class[i].precision:.3f} | "
        f"{metrics.per_class[i].recall:.3f} | {metrics.per_class[i].f1:.3f} | "
        f"{metrics.per_class[i].support} |"
        for i, name in enumerate(LABEL_NAMES[cls] for cls in active)
    )
    top = "\n".join(
        f"1. `{name}` ({value:.3f})"
        for name, value in sorted(
            importance.items(), key=lambda kv: kv[1], reverse=True
        )[:5]
    )
    body = f"""# Experiment {cfg.name}

*Generated: {_timestamp()}*

{cfg.description}

## Data

- Sources: {", ".join(cfg.sources) if cfg.sources else "all manifest sources"}
- Clips processed: {clips}
- Windows extracted: {windows} (training {n_train}, test {n_test})
- Active classes: {", ".join(LABEL_NAMES[c] for c in active)}

## Results

| Metric | Value |
| --- | --- |
| Accuracy | {metrics.accuracy:.3f} |
| Macro F1 | {metrics.macro_f1:.3f} |

### Per-class

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
{per_class}

## Feature importance (top 5)

{top}

## Artifacts

- Model: `models/checkpoints/xgb_{cfg.name}.json` (.meta.json)
- Dataset: `artifacts/dataset/{cfg.name}.parquet`
- Confusion matrix: `reports/{cfg.name}/confusion_matrix.png` (+ .json)
- Metrics: `reports/{cfg.name}/metrics_report.json`
- Feature importance: `reports/{cfg.name}/feature_importance.csv` (+ .png/.json)
- Config snapshot: `reports/{cfg.name}/experiment_config.yaml`
- Manifest snapshot: `reports/{cfg.name}/manifest_snapshot.csv`
"""
    path = report_dir / f"{cfg.name}.md"
    path.write_text(body, encoding="utf-8")
    return path


def _prepare_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _experiment_settings(cfg: ExperimentConfig) -> Any:
    from safewatch.core.config import load_settings

    config_path = Path("configs/default.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(
            "configs/default.yaml not found; run experiments from the repo root"
        )
    return load_settings(config_path=str(config_path))


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


__all__ = [
    "DEFAULT_WINDOW_FRAMES",
    "ExperimentConfig",
    "ExperimentResults",
    "run_experiment",
]
