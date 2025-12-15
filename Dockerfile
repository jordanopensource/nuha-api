# =============================================================================
# EgyNuha API - Multi-stage Dockerfile
# =============================================================================
# Builds a production-ready image with the classification model baked in.
# Model is downloaded from HuggingFace during the build process.
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1: Dependencies Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# -----------------------------------------------------------------------------
# Stage 2: Model Downloader
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS model-downloader

# Build arguments for model download
ARG HF_MODEL_REPO="SafwanLjd/egynuha-classifier"
ARG HF_TOKEN=""

# Make ARG available as ENV for the RUN command
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /model-download

# Install huggingface_hub for downloading models
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub

# Download the model from HuggingFace using Python API
RUN python -c "from huggingface_hub import snapshot_download; import os; snapshot_download(repo_id='${HF_MODEL_REPO}', local_dir='/model', token=os.environ.get('HF_TOKEN') or None)"


# -----------------------------------------------------------------------------
# Stage 3: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# Build-time metadata arguments (set by CI)
# Only define ARGs that are actually used in LABELs
ARG CI_COMMIT_SHA="unknown"
ARG CI_REPO_URL="unknown"
ARG CI_PIPELINE_CREATED=""

# OCI Image Labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL org.opencontainers.image.title="EgyNuha API" \
      org.opencontainers.image.description="Egyptian-Arabic Text Classification API" \
      org.opencontainers.image.source="${CI_REPO_URL}" \
      org.opencontainers.image.revision="${CI_COMMIT_SHA}" \
      org.opencontainers.image.created="${CI_PIPELINE_CREATED}" \
      org.opencontainers.image.vendor="JOSA"

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy model from downloader stage
COPY --from=model-downloader --chown=appuser:appuser /model ./model/

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Container-specific path - must match where model is copied
# All other env vars use defaults from Python code and can be
# overridden at runtime via: docker run --env-file .env
# or docker-compose with env_file directive
ENV MODEL_PATH="/home/appuser/model"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\",\"8000\")}/health')" || exit 1

# Use shell form with defaults for uvicorn configuration
ENTRYPOINT ["sh", "-c", "uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-1} --timeout-keep-alive ${TIMEOUT:-120}"]
