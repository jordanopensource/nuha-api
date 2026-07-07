# =============================================================================
# Nuha API - Multi-stage Dockerfile
# =============================================================================
# Builds a production-ready, PER-DIALECT image with a SINGLE classification
# model baked in. The dialect is chosen at build time with --build-arg DIALECT
# and the model for that dialect (only) is downloaded from HuggingFace during
# the build. At runtime the DIALECT env var must match the baked-in dialect so
# the app loads the model that is actually present.
#
# Build one image per dialect, e.g.:
#   docker build --build-arg DIALECT=arz -t nuha-api:arz .
#   docker build --build-arg DIALECT=acm -t nuha-api:acm .
#   docker build --build-arg DIALECT=ckb -t nuha-api:ckb .
# (compose does this for you: see compose.yml's per-service build section.)
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
# reproducible and verifies every wheel. Inference runs on ONNX Runtime (CPU), a
# normal PyPI package, so no custom wheel index is needed here.
# Regenerate it whenever requirements.txt changes (see the lock file's header).
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --require-hashes -r requirements.lock


# -----------------------------------------------------------------------------
# Stage 2: Model Downloader
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS model-downloader

# DIALECT selects WHICH single model to bake into this image. It is required:
# per-dialect images are the supported model now, so a missing DIALECT fails
# the build early with a clear message rather than producing an empty image.
ARG DIALECT

WORKDIR /model-download

# Install huggingface_hub for downloading the model
RUN pip install --no-cache-dir --root-user-action=ignore huggingface_hub

# app/dialects/<code>.json is the single source of truth for which model to
# download and from which HuggingFace repo. We read hf_repo for THIS dialect
# dynamically from the JSON (never hardcoded), download just that one model into
# /models/${DIALECT}, and bake only it. All repos are public, so it downloads
# anonymously. Adding/retargeting a dialect is a one-file change in app/dialects.
COPY app/dialects/ /model-download/dialects/
RUN test -n "${DIALECT}" || { \
        echo "ERROR: build-arg DIALECT is required (e.g. --build-arg DIALECT=arz)." >&2; \
        echo "       Per-dialect images each bake a single model; pick one of:" >&2; \
        ls /model-download/dialects/*.json | sed 's#.*/##; s#\.json$##' | sed 's/^/         /' >&2; \
        exit 1; \
    }
RUN DIALECT="${DIALECT}" python -c "\
import json, os, sys; from huggingface_hub import snapshot_download; \
code = os.environ['DIALECT']; \
path = '/model-download/dialects/' + code + '.json'; \
sys.exit('ERROR: unknown DIALECT ' + repr(code) + ' (no ' + path + ')') if not os.path.isfile(path) else None; \
repo = json.load(open(path))['hf_repo']; \
print('Downloading ' + repo + ' -> /models/' + code, flush=True); \
snapshot_download(repo_id=repo, local_dir='/models/' + code)"


# -----------------------------------------------------------------------------
# Stage 3: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim

# DIALECT is needed again here for the per-dialect COPY path and to bake a
# default DIALECT env so the image is self-describing. Re-declared because each
# stage gets its own ARG scope.
ARG DIALECT

# Build-time metadata arguments (set by CI)
# Only define ARGs that are actually used in LABELs
ARG CI_COMMIT_SHA="unknown"
ARG CI_REPO_URL="unknown"
ARG CI_PIPELINE_CREATED=""

# OCI Image Labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL org.opencontainers.image.title="Nuha API" \
      org.opencontainers.image.description="Text Classification API (dialect: ${DIALECT})" \
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

# Copy ONLY this dialect's model from the downloader stage. The downloader put
# it at /models/${DIALECT}; keep the same models/<code> layout the app expects
# (classifier.py derives MODEL_PATH=./models/{DIALECT} from the DIALECT env).
COPY --from=model-downloader --chown=appuser:appuser /models/${DIALECT} ./models/${DIALECT}/

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

# Bake the dialect this image was built for as the default DIALECT. compose
# still sets DIALECT per service (and it MUST match this baked value: the
# build-arg controls which model is PRESENT, the env controls which the app
# loads). The default just makes a bare `docker run` of this image work.
# MODEL_PATH auto-derives from DIALECT (./models/{DIALECT}) if not set.
ENV DIALECT=${DIALECT}

USER appuser

EXPOSE 8000

# Generous timeout: under a CPU-saturated burst the event loop that serves
# /health is briefly starved; a tight timeout would falsely mark a busy-but-
# healthy backend unhealthy. (compose.yml's healthcheck mirrors this and, when
# running via Compose, overrides it.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# exec replaces sh with uvicorn as PID 1 for proper signal handling
ENTRYPOINT ["sh", "-c", "exec uvicorn app.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-1} --timeout-keep-alive ${TIMEOUT:-120}"]
