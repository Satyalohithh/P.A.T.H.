"""Tests for the Settings object and config-loading contracts."""

from __future__ import annotations

import pytest

from safewatch.core.config import ConfigLoader, Settings, YamlConfigLoader
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


class TestSettingsSerialization:
    def test_as_dict_excludes_extra(self, settings: Settings) -> None:
        data = settings.as_dict()
        assert "extra" not in data
        assert data["environment"] == "test"


class TestConfigLoaders:
    def test_loaders_are_stubs(self) -> None:
        for loader in (ConfigLoader(), YamlConfigLoader()):
            with pytest.raises(NotImplementedError):
                loader.load("configs/default.yaml")
