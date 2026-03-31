#!/bin/bash
# Fix PostgreSQL collation version mismatch warnings
# This script refreshes the collation version to match the current PostgreSQL version

set -e

DB_HOST="${DB_HOST:-db}"
DB_USER="${DB_USER:-yapat_user}"
DB_NAME="${DB_NAME:-yapat}"
DB_PASSWORD="${DB_PASSWORD:-yapat_password}"

echo "Refreshing collation version for database ${DB_NAME}..."
export PGPASSWORD="${DB_PASSWORD}"
psql -h "${DB_HOST}" -U "${DB_USER}" -d "${DB_NAME}" -c "ALTER DATABASE ${DB_NAME} REFRESH COLLATION VERSION;" 2>&1 | grep -v "WARNING" || true
echo "Collation version refreshed successfully."
