#!/bin/bash
set -e

echo "Waiting for database to be ready..."
until pg_isready -h db -U yapat_user -d yapat; do
  echo "Database is unavailable - sleeping"
  sleep 1
done

echo "Refreshing database collation version to suppress warnings..."
PGPASSWORD=yapat_password psql -h db -U yapat_user -d yapat -c "ALTER DATABASE yapat REFRESH COLLATION VERSION;" >/dev/null 2>&1 || true

echo "Running database migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"

