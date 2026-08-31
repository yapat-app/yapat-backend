"""Add is_label_space_version to custom_taxonomies

Distinguishes genuine team label-space versions (created via the submit/freeze flow)
from internal per-label taxonomies auto-created by Active Learning (which are also
status="active"). Backfills existing rows: a row is a label-space version iff its
taxonomy_data.metadata.created_from_conversation is present.

Revision ID: 2026_08_07_is_labelspace_version
Revises: 2026_08_07_team_active_labelspace
Create Date: 2026-08-07
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "2026_08_07_is_labelspace_version"
down_revision = "2026_08_07_team_active_labelspace"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "custom_taxonomies",
        sa.Column(
            "is_label_space_version",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        # taxonomy_data is JSONB on Postgres — use JSON path operators directly.
        op.execute(
            """
            UPDATE custom_taxonomies
            SET is_label_space_version = TRUE
            WHERE taxonomy_data -> 'metadata' ->> 'created_from_conversation' IS NOT NULL
            """
        )
    else:
        # Portable fallback (e.g. SQLite in tests): inspect JSON in Python.
        import json

        rows = bind.execute(
            sa.text("SELECT id, taxonomy_data FROM custom_taxonomies")
        ).fetchall()
        version_ids = []
        for row in rows:
            data = row[1]
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (ValueError, TypeError):
                    data = None
            if isinstance(data, dict):
                meta = data.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("created_from_conversation") is not None:
                    version_ids.append(row[0])
        for cid in version_ids:
            bind.execute(
                sa.text(
                    "UPDATE custom_taxonomies SET is_label_space_version = 1 WHERE id = :id"
                ),
                {"id": cid},
            )


def downgrade() -> None:
    op.drop_column("custom_taxonomies", "is_label_space_version")
