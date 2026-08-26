"""Migration test for the pricing-provenance backfill.

Rows written before provenance existed have an unknowable origin, so the
backfill assigns ``unresolved`` rather than asserting a trusted source: a
persisted zero-price row relabelled ``openrouter`` would carry a trusted label
into the guard that has to hold it back. A row that somehow already carries a
source keeps it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "f0e1d2c3b4a5_add_pricing_provenance_to_models.py"
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
