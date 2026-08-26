"""add pricing provenance to models

Revision ID: f0e1d2c3b4a5
Revises: b4f7a1c9d2e3
Create Date: 2026-08-26 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "f0e1d2c3b4a5"
down_revision = "b4f7a1c9d2e3"
branch_labels = None
depends_on = None


def _backfill_pricing_source(conn: sa.Connection) -> None:
    # Existing rows predate provenance tracking, so their true origin is
    # unknowable. Backfill "unresolved" rather than assert a trusted source: a
    # persisted zero-price row relabelled "openrouter" would carry a trusted
    # label past the write guard that has to hold it back.
    conn.execute(
        sa.text(
            "UPDATE models SET pricing_source = 'unresolved' "
            "WHERE pricing_source IS NULL"
        )
    )


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("models")}

    if "pricing_source" not in columns:
        op.add_column(
            "models",
            sa.Column("pricing_source", sa.String(), nullable=True),
        )

    _backfill_pricing_source(conn)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {c["name"] for c in inspector.get_columns("models")}

    if "pricing_source" in columns:
        op.drop_column("models", "pricing_source")
