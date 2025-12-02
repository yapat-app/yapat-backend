#!/bin/bash

# Start Flower - Celery monitoring tool
# Access the web UI at http://localhost:5555

echo "Starting Flower monitoring dashboard..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Flower
celery -A app.celery_app flower \
    --port=5555 \
    --broker=redis://localhost:6379/0 \
    --basic_auth=admin:yapat123

