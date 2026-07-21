"""
Database connection
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    # Per-process pool sized so all DB-connected processes together stay under
    # Postgres max_connections (100). With the API run at 4 workers plus the 3
    # celery processes (worker, pam-al worker, beat), 7 processes x 10 = 70,
    # leaving headroom for admin/psql sessions. Do not raise without checking
    # (worker_count + 3) * (pool_size + max_overflow) <= max_connections.
    pool_size=5,   # Connections maintained per process
    max_overflow=5,  # Extra connections created on demand per process
    pool_timeout=30,  # Seconds to wait before giving up on getting a connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    # Safety net against connection leaks: Postgres terminates any session left
    # idle inside an open transaction beyond this window, returning it to the
    # pool. Without it, a request that opens a transaction (e.g. the auth user
    # lookup) but is then stalled leaves the connection checked out forever;
    # enough of these exhaust the pool and require a manual restart. Legitimate
    # transactions complete in milliseconds, so 60s only ever reaps leaks.
    connect_args={"options": "-c idle_in_transaction_session_timeout=60000"},
    echo=False  # Set to True for SQL query logging
)

# Register pgvector type adapter with psycopg2 so that vector columns
# are properly decoded instead of raising "Unknown PG numeric type".
try:
    from pgvector.psycopg2 import register_vector

    @event.listens_for(engine, "connect")
    def _register_pgvector(dbapi_connection, connection_record):
        register_vector(dbapi_connection)
except ImportError:
    pass  # pgvector not installed – skip

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

