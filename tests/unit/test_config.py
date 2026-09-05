"""Tests for the Settings object and config-loading contracts."""

from __future__ import annotations

import pytest

from safewatch.core.config import (
    DetectionConfig,
    Settings,
    TrackingConfig,
    YamlConfigLoader,
    load_settings,
)
from safewatch.core.exceptions import SafeWatchConfigError


class TestSettingsFromMapping:
    def test_coerces_scalars(self) -> None:
        settings = Settings.from_mapping(
            {
                "log_level": "DEBUG",
                "keypoint_confidence_threshold": "0.5",
                "smoothing_kernel_width": "7",
            }
        )
        assert settings.log_level == "DEBUG"
        assert settings.keypoint_confidence_threshold == 0.5
        assert settings.smoothing_kernel_width == 7

    def test_unknown_keys_go_to_extra(self) -> None:
        settings = Settings.from_mapping({"nonsense_key": "value"})
        assert settings.extra["nonsense_key"] == "value"

    def test_invalid_value_raises_config_error(self) -> None:
        with pytest.raises(SafeWatchConfigError):
            Settings.from_mapping({"interaction_window_frames": "not-a-number"})

    def test_detection_keys_collected_in_defaults(self) -> None:
        settings = Settings.from_mapping({})
        assert settings.detection == DetectionConfig()

    def test_detection_prefix_keys_fold_into_nested(self) -> None:
        settings = Settings.from_mapping(
            {"detection_conf_threshold": "0.6", "detection_device": "cpu"}
        )
        assert settings.detection.conf_threshold == 0.6
        assert settings.detection.device == "cpu"

    def test_tracking_keys_collected_in_defaults(self) -> None:
        settings = Settings.from_mapping({})
        assert settings.tracking == TrackingConfig()

    def test_tracking_prefix_keys_fold_into_nested(self) -> None:
        settings = Settings.from_mapping(
            {"tracking_track_thresh": "0.6", "tracking_match_thresh": "0.75"}
        )
        assert settings.tracking.track_thresh == 0.6
        assert settings.tracking.match_thresh == 0.75


class TestSettingsFromEnvironment:
    def test_prefix_filtering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEWATCH_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("UNRELATED", "ignored")
        settings = Settings.from_environment()
        assert settings.log_level == "WARNING"

    def test_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SW_LOG_LEVEL", "ERROR")
        settings = Settings.from_environment(prefix="SW_")
        assert settings.log_level == "ERROR"

    def test_detection_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEWATCH_DETECTION_CONF_THRESHOLD", "0.7")
        assert Settings.from_environment().detection.conf_threshold == 0.7

    def test_tracking_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEWATCH_TRACKING_TRACK_BUFFER", "60")
        assert Settings.from_environment().tracking.track_buffer == 60


class TestSettingsSerialization:
    def test_as_dict_excludes_extra(self, settings: Settings) -> None:
        data = settings.as_dict()
        assert "extra" not in data
        assert data["environment"] == "test"

    def test_as_dict_flattens_nested_models(self) -> None:
        data = Settings.from_mapping({}).as_dict()
        assert data["detection"] == DetectionConfig().model_dump()
        assert "conf_threshold" in data["detection"]
        assert data["tracking"] == TrackingConfig().model_dump()
        assert "track_thresh" in data["tracking"]


class TestYamlConfigLoader:
    def test_missing_file_raises_config_error(self, tmp_path) -> None:
        with pytest.raises(SafeWatchConfigError):
            YamlConfigLoader().load(str(tmp_path / "nope.yaml"))

    def test_invalid_yaml_raises_config_error(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(":: not: [valid", encoding="utf-8")
        with pytest.raises(SafeWatchConfigError):
            YamlConfigLoader().load(str(bad))

    def test_flattens_nested_yaml(self, tmp_path) -> None:
        cfg = tmp_path / "ok.yaml"
        cfg.write_text(
            """environment: test
logging:
  level: WARNING
  format: plain
detection:
  confidence_threshold: 0.55
tracking:
  backend: bytetrack
  track_thresh: 0.6
  match_thresh: 0.75
  track_buffer: 40
  min_hits: 3
pose:
  keypoint_confidence_threshold: 0.5
  min_valid_keypoints: 8
interaction:
  window_frames: 42
  zones:
    intimate_h: 0.6
smoothing:
  sigma_frames: 3.0
  kernel_width: 7
""",
            encoding="utf-8",
        )
        got = YamlConfigLoader().load(str(cfg))
        assert got["environment"] == "test"
        assert got["log_level"] == "WARNING"
        assert got["detection"]["conf_threshold"] == 0.55
        assert got["tracking"]["track_thresh"] == 0.6
        assert got["tracking"]["match_thresh"] == 0.75
        assert got["tracking"]["track_buffer"] == 40
        assert got["tracking"]["min_hits"] == 3
        assert got["min_valid_keypoints_for_pose"] == 8
        assert got["interaction_window_frames"] == 42
        assert got["intimate_zone_h"] == 0.6
        assert got["smoothing_sigma_frames"] == 3.0
        assert got["smoothing_kernel_width"] == 7


class TestLoadSettings:
    def test_loads_defaults(self) -> None:
        settings = load_settings("configs/default.yaml")
        assert settings.log_level == "INFO"
        assert settings.detection.conf_threshold == 0.45
        assert settings.tracking == TrackingConfig()

    def test_env_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAFEWATCH_LOG_LEVEL", "DEBUG")
        settings = load_settings("configs/default.yaml")
        assert settings.log_level == "DEBUG"

    def test_explicit_overrides_have_highest_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SAFEWATCH_DETECTION_CONF_THRESHOLD", "0.9")
        settings = load_settings(
            "configs/default.yaml",
            overrides={"detection_conf_threshold": "0.3"},
        )
        assert settings.detection.conf_threshold == 0.3
        assert settings.detection.weights == "yolov8n.pt"

    def test_missing_config_raises_config_error(self, tmp_path) -> None:
        with pytest.raises(SafeWatchConfigError):
            load_settings(str(tmp_path / "missing.yaml"))
