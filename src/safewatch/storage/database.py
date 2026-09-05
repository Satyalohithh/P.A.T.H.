"""Database connection management (SQLAlchemy + Alembic)."""

from __future__ import annotations

from safewatch.core.exceptions import StorageError


class Database:
    """Wraps the SQLAlchemy engine and session factory."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url

    def connect(self) -> None:
        raise NotImplementedError("TODO(implementation): Database.connect")

    def disconnect(self) -> None:
        raise NotImplementedError("TODO(implementation): Database.disconnect")

    def create_all(self) -> None:
        raise NotImplementedError("TODO(implementation): Database.create_all")

    def migrate(self) -> None:
        """Run Alembic migrations up to head."""

        raise NotImplementedError("TODO(implementation): Database.migrate")


def database_engine(url: str) -> object:
    """Return a SQLAlchemy engine for ``url`` (declared)."""

    raise NotImplementedError("TODO(implementation): database_engine")


__all__ = ["Database", "StorageError", "database_engine"]
