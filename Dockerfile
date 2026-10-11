# Inference API image.
#
# The image intentionally ships WITHOUT weights. artifacts/ is a volume, so a
# container started without a trained checkpoint exits with a clear error
# rather than silently serving an untrained network.
#
#   docker compose --profile train run --rm train      # produce a checkpoint
#   docker compose up api                              # serve it

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libglib is the only system library opencv-python-headless needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels: the default PyPI torch wheel bundles ~2.5 GB of CUDA libraries.
# Swap the index URL for the CUDA build if you deploy to a GPU host.
COPY requirements.txt requirements-export.txt ./
RUN pip install torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -r requirements-export.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app
USER appuser

VOLUME ["/app/artifacts"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "per_camera.api:app", "--host", "0.0.0.0", "--port", "8000"]
