"""Migration test for the pricing-provenance backfill.

Rows written before provenance existed have an unknowable origin, so the
backfill assigns ``unresolved`` rather than asserting a trusted source: a
persisted zero-price row relabelled ``openrouter`` would carry a trusted label
into the guard that has to hold it back. A row that somehow already carries a
source keeps it.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import sqlalchemy as sa

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "a9c3f5b7d1e4_add_pricing_provenance_to_models.py"
)
_spec = importlib.util.spec_from_file_location(
    "pricing_provenance_migration", _MIGRATION_PATH
)
assert _spec is not None and _spec.loader is not None
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


def test_backfill_assigns_unresolved_and_preserves_an_existing_source() -> None:
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE models ("
                "id VARCHAR PRIMARY KEY, "
                "pricing_source VARCHAR NULL"
                ")"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO models (id, pricing_source) VALUES "
                "('a', NULL), "
                "('b', 'manual')"
            )
        )

        migration._backfill_pricing_source(conn)

        rows = conn.execute(
            sa.text("SELECT id, pricing_source FROM models ORDER BY id")
        ).all()

    assert rows == [("a", "unresolved"), ("b", "manual")]


def test_backfill_does_not_touch_whether_a_row_is_enabled() -> None:
    """The column is inert metadata. Which rows a node serves is decided by the
    running node, never rewritten underneath the operator by a migration."""
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE models ("
                "id VARCHAR PRIMARY KEY, "
                "pricing VARCHAR, enabled BOOLEAN, pricing_source VARCHAR NULL"
                ")"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO models (id, pricing, enabled, pricing_source) VALUES "
                "('free-enabled', '{\"prompt\": 0, \"completion\": 0}', 1, NULL), "
                "('priced-enabled', "
                '\'{"prompt": 0.000001, "completion": 0}\', 1, NULL)'
            )
        )

        migration._backfill_pricing_source(conn)

        rows: dict[str, int] = {
            row[0]: row[1]
            for row in conn.execute(sa.text("SELECT id, enabled FROM models")).all()
        }

    assert rows == {"free-enabled": 1, "priced-enabled": 1}


def _run_alembic(root: Path, database_url: str, command: str, revision: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", command, revision],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _model_columns(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        return {row[1] for row in connection.execute("PRAGMA table_info(models)")}


def test_the_column_arrives_and_leaves_with_the_migration(tmp_path: Path) -> None:
    """The chain has to run both ways from the head this migration was written
    against, so an operator who upgrades can also step back off it."""
    root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "pricing-provenance-migration.db"
    database_url = f"sqlite+aiosqlite:///{database_path}"
    previous_head = "b4f7a1c9d2e3"

    _run_alembic(root, database_url, "upgrade", previous_head)
    assert "pricing_source" not in _model_columns(database_path)

    _run_alembic(root, database_url, "upgrade", "a9c3f5b7d1e4")
    assert "pricing_source" in _model_columns(database_path)

    _run_alembic(root, database_url, "downgrade", previous_head)
    assert "pricing_source" not in _model_columns(database_path)
