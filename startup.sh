#!/bin/bash
set -e

echo "Waiting for database to be ready..."
until pg_isready -h db -U yapat_user -d yapat; do
  echo "Database is unavailable - sleeping"
  sleep 1
done

# psql connects with the same credentials as the app, so a deployment that
# changes the database password only has to change DATABASE_URL. libpq does not
# understand SQLAlchemy driver suffixes such as "+psycopg2", so strip them.
PSQL_URL=$(printf '%s' "${DATABASE_URL:?DATABASE_URL must be set}" | sed -E 's#^postgresql\+[A-Za-z0-9_]+://#postgresql://#')

echo "Refreshing database collation version to suppress warnings..."
psql "$PSQL_URL" -c "ALTER DATABASE yapat REFRESH COLLATION VERSION;" >/dev/null 2>&1 || true

echo "Ensuring alembic_version can store long revision ids..."
psql "$PSQL_URL" -v ON_ERROR_STOP=1 -c "\
CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(128) NOT NULL PRIMARY KEY);\
ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128);\
"

echo "Running database migrations..."
# If the DB points to a revision that no longer exists in the codebase (e.g. a
# migration was applied locally but never committed), stamp it to the current
# head so `upgrade head` doesn't crash.  This is safe because stamping only
# updates the version pointer — it never drops or alters schema.
if ! alembic current 2>&1 | grep -q "(head)"; then
  if alembic current 2>&1 | grep -q "Can't locate revision"; then
    echo "WARNING: DB points to an unknown revision. Stamping to current head..."
    alembic stamp head
  fi
fi
alembic upgrade head

echo "Starting application..."
exec "$@"

