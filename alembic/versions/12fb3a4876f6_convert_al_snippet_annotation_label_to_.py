"""convert al_snippet_annotation label to labels array

Revision ID: 12fb3a4876f6
Revises: a1b2c3d4e5f6
Create Date: 2026-07-08 14:05:05.473323

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '12fb3a4876f6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add the new column alongside the old one.
    op.add_column(
        "al_snippet_annotation",
        sa.Column("labels", sa.JSON(), nullable=True),
    )

    # 2. Consolidate: for each (snippet_id, source, user_id, model_checkpoint_id)
    #    group, collect all existing `label` values into one row's `labels` array.
    conn = op.get_bind()
    conn.execute(sa.text("""
        WITH grouped AS (
            SELECT snippet_id, source, user_id, model_checkpoint_id,
                   array_agg(DISTINCT label) AS labels_arr,
                   min(id) AS keep_id
            FROM al_snippet_annotation
            GROUP BY snippet_id, source, user_id, model_checkpoint_id
        )
        UPDATE al_snippet_annotation a
        SET labels = to_jsonb(g.labels_arr)
        FROM grouped g
        WHERE a.id = g.keep_id
    """))

    # 3. Delete the now-redundant duplicate rows (everything not the "keep_id").
    conn.execute(sa.text("""
        DELETE FROM al_snippet_annotation a
        USING al_snippet_annotation b
        WHERE a.snippet_id = b.snippet_id
          AND a.source = b.source
          AND COALESCE(a.user_id, -1) = COALESCE(b.user_id, -1)
          AND COALESCE(a.model_checkpoint_id, -1) = COALESCE(b.model_checkpoint_id, -1)
          AND a.id > b.id
    """))

    # 4. Drop the old column/index/constraint, finalize the new column/constraint.
    op.drop_index("ix_al_snippet_annotation_label", table_name="al_snippet_annotation")
    op.drop_constraint(
        "uq_al_snippet_label_source_user_ckpt", "al_snippet_annotation", type_="unique"
    )
    op.drop_column("al_snippet_annotation", "label")
    op.alter_column("al_snippet_annotation", "labels", nullable=False)
    op.create_unique_constraint(
        "uq_al_snippet_source_user_ckpt",
        "al_snippet_annotation",
        ["snippet_id", "source", "user_id", "model_checkpoint_id"],
    )


def downgrade() -> None:
    # Reverse: add `label` back, explode `labels` arrays into one row per label,
    # drop the `labels` column and its constraint, restore the old constraint.
    op.add_column(
        "al_snippet_annotation",
        sa.Column("label", sa.String(), nullable=True),
    )

    conn = op.get_bind()
    conn.execute(sa.text("""
        INSERT INTO al_snippet_annotation
            (dataset_id, snippet_id, label, source, user_id, model_checkpoint_id, created_at)
        SELECT a.dataset_id, a.snippet_id, lbl, a.source, a.user_id, a.model_checkpoint_id, a.created_at
        FROM al_snippet_annotation a,
             jsonb_array_elements_text(a.labels::jsonb) AS lbl
        WHERE jsonb_array_length(a.labels::jsonb) > 1
    """))

    conn.execute(sa.text("""
        UPDATE al_snippet_annotation
        SET label = labels::jsonb ->> 0
        WHERE jsonb_array_length(labels::jsonb) >= 1
    """))

    op.alter_column("al_snippet_annotation", "label", nullable=False)
    op.drop_constraint(
        "uq_al_snippet_source_user_ckpt", "al_snippet_annotation", type_="unique"
    )
    op.drop_column("al_snippet_annotation", "labels")
    op.create_unique_constraint(
        "uq_al_snippet_label_source_user_ckpt",
        "al_snippet_annotation",
        ["snippet_id", "label", "source", "user_id", "model_checkpoint_id"],
    )
    op.create_index(
        "ix_al_snippet_annotation_label", "al_snippet_annotation", ["label"]
    )
