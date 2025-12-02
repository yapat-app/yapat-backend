#!/bin/bash

# Start Celery Beat scheduler for periodic tasks
# This is useful for scheduled maintenance, cleanup, or monitoring tasks

echo "Starting Celery Beat scheduler..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Celery Beat
celery -A app.celery_app beat \
    --loglevel=info \
    --scheduler=celery.beat:PersistentScheduler

