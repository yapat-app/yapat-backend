#!/bin/bash

# Start Celery worker for YAPAT backend
# This script starts a Celery worker with appropriate configuration

echo "Starting Celery worker for YAPAT..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Celery worker with multiple queues
celery -A app.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=default,embeddings,processing,exports \
    --max-tasks-per-child=100 \
    --task-events \
    --without-gossip \
    --without-mingle

# Alternative: Start multiple specialized workers
# Uncomment to run specialized workers for different task types

# High priority processing queue
# celery -A app.celery_app worker --loglevel=info --concurrency=2 --queues=processing -n processing@%h &

# Embedding generation queue (can be GPU-intensive)
# celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=embeddings -n embeddings@%h &

# Low priority export queue
# celery -A app.celery_app worker --loglevel=info --concurrency=2 --queues=exports -n exports@%h &

# Default queue for everything else
# celery -A app.celery_app worker --loglevel=info --concurrency=2 --queues=default -n default@%h &

