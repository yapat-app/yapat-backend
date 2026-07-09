FROM python:3.12-slim

WORKDIR /app

# cpu (default) or gpu — set via DEVICE in .env
ARG DEVICE=cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    postgresql-client \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install all dependencies except torch first (separate layer for better caching)
RUN grep -v '^torch' requirements.txt > /tmp/req-base.txt && \
    pip install --prefer-binary --progress-bar off -r /tmp/req-base.txt

# Install torch: CPU uses --no-deps to skip ~900 MB of NVIDIA CUDA runtime libs
# that are unnecessary for CPU inference. GPU gets the full CUDA build.
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cu128; \
    else \
        pip install torch --no-deps --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install TensorFlow: GPU build ships CUDA kernels via tensorflow[and-cuda] wheels
# (no host CUDA install required — all CUDA libs bundled). CPU build uses the
# regular `tensorflow` package — `tensorflow-cpu` has no published wheels for
# linux/aarch64 (or recent CPython versions) and CUDA is loaded lazily anyway,
# so the plain package is CPU-safe with no extra size/dependency cost here.
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install "tensorflow[and-cuda]"; \
    else \
        pip install tensorflow; \
    fi

# For GPU: register the pip-bundled CUDA libs with the dynamic linker so TF can
# dlopen them without a hardcoded LD_LIBRARY_PATH. Computed dynamically so it
# works regardless of Python version or package layout changes.
RUN if [ "$DEVICE" = "gpu" ]; then \
        python3 -c "import site,glob,os;paths=[p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp,'nvidia','*','lib'))];print('\n'.join(paths))" \
            > /etc/ld.so.conf.d/nvidia-pip.conf && ldconfig; \
    fi

# Copy application code
COPY . .

# Normalize line endings and make startup script executable
RUN sed -i 's/\r$//' /app/startup.sh && chmod +x /app/startup.sh

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

