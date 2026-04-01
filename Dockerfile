FROM python:3.12-slim

WORKDIR /app

# for faster builds
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    postgresql-client \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --prefer-binary --progress-bar off -r requirements.txt

# Copy application code
COPY . .

# Normalize line endings and make startup script executable
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

