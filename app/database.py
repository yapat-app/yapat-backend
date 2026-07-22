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
    pool_size=5,  # per process
    max_overflow=5,  # per process
    pool_timeout=30,  # Seconds to wait before giving up on getting a connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={"options": "-c idle_in_transaction_session_timeout=60000"},  # Postgres auto-terminates leaked idle-in-transaction sessions after 60s
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

