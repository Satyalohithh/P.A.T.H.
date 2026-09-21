"""End-to-end tests for the Phase-6 experiment orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from safewatch.core.constants import BehaviorClass
from safewatch.data.experiment import ExperimentConfig, run_experiment


def _config(tmp_path: Path, **overrides: object) -> ExperimentConfig:
    kwargs: dict[str, object] = {
        "name": "smoke",
        "description": "synthetic smoke run",
        "seed": 7,
        "test_size": 0.25,
        "synthetic_videos": 6,
        "artifacts_dir": str(tmp_path / "artifacts"),
        "reports_dir": str(tmp_path / "reports"),
        "models_dir": str(tmp_path / "models"),
    }
    kwargs.update(overrides)
    return ExperimentConfig.from_dict(kwargs)


def test_run_experiment_synthetic_end_to_end(tmp_path: Path) -> None:
    results = run_experiment(_config(tmp_path), synthetic=True, force=True)

    assert results.n_train > 0
    assert results.n_test > 0
    assert results.windows_total == results.n_train + results.n_test
    assert results.clips_processed == 3 * 6
    assert results.active_classes == (
        BehaviorClass.NORMAL,
        BehaviorClass.PLAYFUL,
        BehaviorClass.AGGRESSIVE,
    )
    assert results.feature_catalog
    assert results.metrics.macro_f1 > 0.9

    for key in ("report", "model", "dataset", "confusion", "importance"):
        assert results.outputs[key].is_file(), f"missing output {key}"

    report = results.outputs["report"].read_text(encoding="utf-8")
    assert report.startswith("# Experiment smoke")


def test_report_snapshots_written(tmp_path: Path) -> None:
    run_experiment(_config(tmp_path), synthetic=True, force=True)

    report_dir = tmp_path / "reports" / "smoke"
    assert (report_dir / "experiment_config.yaml").is_file()
    assert (report_dir / "manifest_snapshot.csv").is_file()
    assert (report_dir / "metrics_report.json").is_file()
    assert (report_dir / "confusion_matrix.png").is_file()
    assert (report_dir / "confusion_matrix.json").is_file()
    assert (report_dir / "feature_importance.csv").is_file()
    assert (report_dir / "feature_importance.png").is_file()
    assert (report_dir / "smoke.md").is_file()

    metrics = json.loads(
        (report_dir / "metrics_report.json").read_text(encoding="utf-8")
    )
    assert set(metrics["active_classes"]) == {"normal", "playful", "aggressive"}
    assert set(metrics["per_class"]) == set(metrics["active_classes"])

    snapshot = yaml.safe_load(
        (report_dir / "experiment_config.yaml").read_text(encoding="utf-8")
    )
    assert snapshot["name"] == "smoke"
    assert snapshot["synthetic_videos"] == 6


def test_model_meta_reflects_active_classes(tmp_path: Path) -> None:
    results = run_experiment(_config(tmp_path), synthetic=True, force=True)

    meta_path = Path(str(results.outputs["model"]) + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["num_classes"] == 3
    assert meta["labels"] == ["normal", "playful", "aggressive"]
    assert len(meta["feature_names"]) == 14


def test_config_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "from_file",
                "seed": 1,
                "test_size": 0.2,
                "synthetic_videos": 4,
            }
        ),
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_file(config_path)
    assert cfg.name == "from_file"
    assert cfg.window_frames == 30


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown experiment config keys"):
        _config(tmp_path, bogus_key=1)
