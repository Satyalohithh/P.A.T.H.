"""Ensures every safewatch module imports with the standard library only.

Third-party dependencies are not installed in the CI fast path; this guards
the import-stability contract of the package.
"""

from __future__ import annotations

import importlib
import pkgutil

import safewatch

# Alembic runtime glue is loaded by the migration CLI, not by the package.
EXCLUDED_MODULES = {"safewatch.storage.migrations.env"}

ALL_MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(safewatch.__path__, "safewatch.")
    if info.name not in EXCLUDED_MODULES
)


def test_package_metadata() -> None:
    assert safewatch.__version__ == "0.1.0"


def test_all_modules_importable() -> None:
    assert ALL_MODULES, "expected at least one safewatch module"
    for module_name in ALL_MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, module_name
