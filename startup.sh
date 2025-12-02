#!/bin/bash
set -e

echo "Waiting for database to be ready..."
until pg_isready -h db -U yapat_user -d yapat; do
  echo "Database is unavailable - sleeping"
  sleep 1
done

echo "Running database migrations..."
alembic upgrade head

echo "Starting application..."
exec "$@"

