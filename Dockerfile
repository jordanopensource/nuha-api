# =============================================================================
# Nuha API - Multi-stage Dockerfile
# =============================================================================
# Builds a production-ready image with all classification models baked in.
# Models are downloaded from HuggingFace during the build process.
# At runtime, DIALECT env var selects which model to load.
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1: Dependencies Builder
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements.lock is the generated, fully-pinned, hashed lock (resolved from
# requirements.txt). Installing it with --require-hashes makes the build
# reproducible and verifies every wheel. It carries the +cpu torch pin and the
# PyTorch CPU index, so torch resolves CPU-only and never the CUDA wheel.
# Regenerate it whenever requirements.txt changes (see the lock file's header).
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --require-hashes -r requirements.lock


# -----------------------------------------------------------------------------
# Stage 2: Model Downloader
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS model-downloader

WORKDIR /model-download

# Install huggingface_hub for downloading models
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub

# app/dialects/<code>.json is the single source of truth for which models to
# download and from which HuggingFace repo. Adding a dialect is a one-file
# change — no edits here. All repos are public, so models download anonymously.
COPY app/dialects/ /model-download/dialects/
RUN python -c "\
import json, glob, os; from huggingface_hub import snapshot_download; \
files = sorted(glob.glob('/model-download/dialects/*.json')); \
[( \
    print('Downloading ' + cfg['hf_repo'] + ' -> /models/' + code, flush=True), \
    snapshot_download(repo_id=cfg['hf_repo'], local_dir='/models/' + code) \
) for code, cfg in ( \
    (os.path.splitext(os.path.basename(f))[0], json.load(open(f))) for f in files \
)]"


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
LABEL org.opencontainers.image.title="Nuha API" \
      org.opencontainers.image.description="Text Classification API" \
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

# Copy models from downloader stage
COPY --from=model-downloader --chown=appuser:appuser /models ./models/

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# DIALECT must be set at runtime (e.g. DIALECT=arz in compose.yml).
# MODEL_PATH auto-derives from DIALECT if not set (defaults to ./models/{DIALECT}).
# Other env vars use defaults from Python code and can be
# overridden at runtime via: docker run --env-file .env

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# exec replaces sh with uvicorn as PID 1 for proper signal handling
ENTRYPOINT ["sh", "-c", "exec uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-1} --timeout-keep-alive ${TIMEOUT:-120}"]
