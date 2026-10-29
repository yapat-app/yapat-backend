"""Seed a default admin user on first setup

Fresh installs previously had to register an admin by hand through Swagger UI.
This data migration creates one if (and only if) no ADMIN user exists yet.

Credentials come from ``DEFAULT_ADMIN_USERNAME`` / ``DEFAULT_ADMIN_PASSWORD``
(defaults: ``admin`` / ``admin12345``, see app/config.py).

Safe to re-run: an existing ADMIN means no change (its password is never
reset), and a username already taken by a non-admin user is never promoted or
modified -- seeding is skipped with a warning instead.

Revision ID: 2026_10_07_seed_default_admin
Revises: 2026_09_22_al_pred_composite_idx
Create Date: 2026-10-07
"""

from alembic import op
import sqlalchemy as sa

from app.config import settings
from app.core.security import get_password_hash

revision = "2026_10_07_seed_default_admin"
down_revision = "2026_09_22_al_pred_composite_idx"
branch_labels = None
depends_on = None

# Same rules as UserCreate.validate_password_length in app/schemas/user.py.
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72  # bcrypt limit

# Lightweight table stub so the migration does not depend on the ORM model.
users = sa.table(
    "users",
    sa.column("username", sa.String),
    sa.column("hashed_password", sa.String),
    sa.column("role", sa.Enum("ADMIN", "TEAM_OWNER", "USER", name="userrole", create_type=False)),
    sa.column("is_active", sa.Boolean),
)

_RULE = "=" * 60

# The built-in default may be shown for first-run convenience; a custom
# password is a secret and must never be written to the logs.
BUILTIN_DEFAULT_PASSWORD = type(settings).model_fields["DEFAULT_ADMIN_PASSWORD"].default


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"DEFAULT_ADMIN_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters long"
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"DEFAULT_ADMIN_PASSWORD cannot be longer than {MAX_PASSWORD_BYTES} bytes"
        )


def _say(*lines: str) -> None:
    print("\n".join(("", _RULE, " YAPAT Database Setup", _RULE, "", *lines, "",
                     _RULE, " Database Setup Complete", _RULE, "")))


def seed_default_admin(conn, username: str, password: str) -> str:
    """Create the default admin if none exists.

    Returns "exists", "username_taken" or "created".
    """
    if not username:
        raise ValueError("DEFAULT_ADMIN_USERNAME must not be empty")

    if conn.execute(sa.select(users.c.username).where(users.c.role == "ADMIN").limit(1)).first():
        _say(
            "✓ Database tables are ready.",
            "✓ Existing administrator account found.",
            "✓ No changes were made to existing accounts.",
        )
        return "exists"

    if conn.execute(sa.select(users.c.username).where(users.c.username == username)).first():
        _say(
            "✓ Database tables are ready.",
            "✓ No administrator account was found.",
            "! WARNING: the default administrator could not be created, because the",
            f"  username '{username}' is already in use by a non-admin account.",
            "  That account was NOT modified (not promoted, password unchanged).",
            "",
            "  To get an administrator, set DEFAULT_ADMIN_USERNAME to an unused name",
            "  and re-run the migration, or register one manually via the API.",
        )
        return "username_taken"

    _validate_password(password)
    conn.execute(
        sa.insert(users).values(
            username=username,
            hashed_password=get_password_hash(password),
            role="ADMIN",
            is_active=True,
        )
    )
    shown_password = (
        password if password == BUILTIN_DEFAULT_PASSWORD else "configured via DEFAULT_ADMIN_PASSWORD"
    )
    _say(
        "✓ Database tables are ready.",
        "✓ No administrator account was found.",
        "✓ Creating the default administrator account...",
        "",
        f"  Username: {username}",
        f"  Password: {shown_password}",
        "  Role:     ADMIN",
        "",
        "✓ Default administrator account created successfully.",
        "",
        "You can now log in to YAPAT with these credentials.",
        "Please change the password after your first login.",
    )
    return "created"


def upgrade() -> None:
    seed_default_admin(
        op.get_bind(),
        settings.DEFAULT_ADMIN_USERNAME,
        settings.DEFAULT_ADMIN_PASSWORD,
    )


def downgrade() -> None:
    # No-op: the admin may own or be linked to data by now.
    pass
