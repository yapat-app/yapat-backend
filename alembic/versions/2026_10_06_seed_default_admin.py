"""Seed a default admin user

Fresh installs previously had to register the first admin through Swagger UI.
This creates one automatically if no ADMIN user exists yet. Credentials come
from DEFAULT_ADMIN_USERNAME / DEFAULT_ADMIN_PASSWORD (defaults: admin /
admin12345) -- override the password for any real deployment.

Revision ID: 2026_10_06_seed_default_admin
Revises: 2026_08_07_custom_tax_dataset_scope
Create Date: 2026-10-06
"""
import logging
import os

from alembic import op
import sqlalchemy as sa

from app.core.security import get_password_hash


# revision identifiers, used by Alembic.
revision = "2026_10_06_seed_default_admin"
down_revision = "2026_08_07_custom_tax_dataset_scope"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin12345"

# Lightweight table definition so the migration doesn't depend on app models.
users = sa.table(
    "users",
    sa.column("username", sa.String),
    sa.column("hashed_password", sa.String),
    sa.column("full_name", sa.String),
    sa.column("role", sa.String),
    sa.column("is_active", sa.Boolean),
)


def seed_default_admin(conn) -> None:
    """Create the default admin unless an ADMIN user already exists."""
    if conn.execute(sa.select(sa.func.count()).select_from(users).where(users.c.role == "ADMIN")).scalar():
        return

    username = os.environ.get("DEFAULT_ADMIN_USERNAME") or DEFAULT_USERNAME
    password = os.environ.get("DEFAULT_ADMIN_PASSWORD") or DEFAULT_PASSWORD

    if conn.execute(sa.select(users.c.username).where(users.c.username == username)).first():
        logger.warning("Default admin not created: username %r is already taken by a non-admin user.", username)
        return

    conn.execute(
        users.insert().values(
            username=username,
            hashed_password=get_password_hash(password),
            full_name="Default Admin",
            role="ADMIN",
            is_active=True,
        )
    )
    logger.info("Created default admin user %r.", username)
    if password == DEFAULT_PASSWORD:
        logger.warning("Default admin is using the default password; set DEFAULT_ADMIN_PASSWORD or change it after login.")


def upgrade() -> None:
    seed_default_admin(op.get_bind())


def downgrade() -> None:
    # Intentionally a no-op: never delete an admin account on downgrade.
    pass
