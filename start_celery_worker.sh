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

# Start Celery worker for all general queues (embeddings, processing, exports, default)
# PAM Active Learning tasks run in a separate worker — see below.
celery -A app.celery_app worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=default,embeddings,processing,exports \
    --max-tasks-per-child=100 \
    --task-events \
    --without-gossip \
    --without-mingle

# ──────────────────────────────────────────────────────────────────────────────
# PAM Active Learning worker
# ──────────────────────────────────────────────────────────────────────────────
# Run this in a separate terminal.  Concurrency is intentionally 1 because
# training loads large embedding matrices and PyTorch models into RAM/VRAM.
# Use --queues=pam_al to keep heavy training tasks isolated from other workers.
#
# CPU:
#   celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=pam_al -n pam_al@%h
#
# GPU (set PAM_DEFAULT_DEVICE=cuda in your .env first):
#   celery -A app.celery_app worker --loglevel=info --concurrency=1 --queues=pam_al -n pam_al@%h
# ──────────────────────────────────────────────────────────────────────────────

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

