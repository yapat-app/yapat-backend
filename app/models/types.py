"""
Shared, dialect-aware column types.

Same rationale as VectorType in app/models/embedding.py: some columns use a
Postgres-only type with no SQLite equivalent, which is fine for production
(Postgres always) but breaks any test that creates tables against SQLite --
not just tests for the table itself, since Base.metadata is a single shared,
cumulative object across a whole pytest session. Once any test imports a
model using the raw type, every other test in that session that creates
tables via Base.metadata.create_all() fails with
"Compiler ... can't render element of type JSONB", regardless of whether
that test touches the offending table at all.
"""
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator


class PortableJSONB(TypeDecorator):
    """Real JSONB on Postgres, plain JSON on SQLite (and anything else)."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())
