"""Tests for structured logging bootstrap."""

from __future__ import annotations

import json
import logging

from safewatch.utils.logging import JsonFormatter, configure_logging, get_logger


class TestJsonFormatter:
    def test_emits_valid_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello"
        assert parsed["ts"]

    def test_includes_exc_info(self) -> None:
        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=True,
            )
            parsed = json.loads(formatter.format(record))
            assert "ValueError: boom" in parsed["exc_info"]


class TestConfigureLogging:
    def test_json_default_configures_root(self) -> None:
        configure_logging(level="INFO", fmt="json")
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)

    def test_plain_format(self) -> None:
        configure_logging(level="DEBUG", fmt="plain")
        root = logging.getLogger()
        assert not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers)


class TestGetLogger:
    def test_returns_named_logger(self) -> None:
        assert get_logger("safewatch.test").name == "safewatch.test"
