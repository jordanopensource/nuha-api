# =============================================================================
# Nuha API - Dockerfile
# =============================================================================
# ONE stateless image for ALL dialects. Models are NOT baked in: the api
# service loads every dialect from the models volume at startup (MODELS_DIR),
# and the same image runs the compose `fetch` service that populates that
# volume. There is deliberately no ARG DIALECT and no download stage; the
# image is identical regardless of which dialects exist.
#
#   docker build -t nuha-api:local .
# =============================================================================


# -----------------------------------------------------------------------------
# Stage 1: Dependencies Builder
# -----------------------------------------------------------------------------
# Base pinned by DIGEST (a tag is mutable). Bump both stages together via
# `docker buildx imagetools inspect python:3.12-slim`.
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# requirements.lock is the generated, fully-pinned, hashed lock (resolved from
# requirements.txt). --require-hashes makes the build reproducible and verifies
# every wheel. huggingface_hub (the fetch service's downloader) is already in
# the lock as a transitive dependency of transformers.
COPY requirements.lock .
RUN pip install --no-cache-dir pip==26.1.2 \
    && pip install --no-cache-dir --require-hashes -r requirements.lock


# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

ARG CI_COMMIT_SHA="unknown"
ARG CI_REPO_URL="unknown"
ARG CI_PIPELINE_CREATED=""

LABEL org.opencontainers.image.title="Nuha API" \
      org.opencontainers.image.description="Text Classification API (models loaded from a volume)" \
      org.opencontainers.image.source="${CI_REPO_URL}" \
      org.opencontainers.image.revision="${CI_COMMIT_SHA}" \
      org.opencontainers.image.created="${CI_PIPELINE_CREATED}" \
      org.opencontainers.image.vendor="JOSA"

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Code is ROOT-owned (no --chown): the appuser process can read but never
# rewrite what it executes. app/dialects ships the repo's reviewed configs; the
# RUNTIME never reads them (the volume's dialect.json files are the live
# config), they are the fetch command's install source. The two scripts make
# this image self-sufficient as the compose `fetch`/debug tool.
COPY app/__init__.py app/main.py app/classifier.py app/registry.py ./app/
COPY app/common/ ./app/common/
COPY app/dialects/ ./app/dialects/
COPY scripts/fetch_models.py scripts/validate_dialects.py ./scripts/

# The models volume mounts here (compose sets MODELS_DIR=/models). Owned by
# appuser so Docker's first-use volume initialization inherits that ownership
# and the non-root fetch service can write; the api service mounts it :ro.
RUN install -d -o appuser -g appuser /models
ENV MODELS_DIR=/models

USER appuser

EXPOSE 8000

# /health is liveness only; /ready (the compose healthcheck target during
# startup gating) flips after the model scan. start-period covers N sequential
# model loads at boot, not just process start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# --workers is a hard-coded 1, not an env knob: inference runs in a thread pool
# over shared models (the forward pass releases the GIL), so a second uvicorn
# process adds no throughput while duplicating every model in memory and
# running a second admission gate that quietly undercuts the CLASSIFIER_WORKERS
# concurrency tuning. Concurrency scales with CLASSIFIER_WORKERS (in-process)
# and replicas (across containers). The container ALWAYS binds 0.0.0.0:8000
# internally (the healthcheck and the compose port map target :8000); PORT is
# only the host-published port in compose.
ENTRYPOINT ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive ${TIMEOUT:-120}"]
