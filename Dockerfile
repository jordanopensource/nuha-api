# Build stage
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies in a virtual environment for clean copying
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Runtime stage
FROM python:3.12-slim

# Build-time metadata
ARG CI_COMMIT_SHA="unknown"
ARG CI_REPO_URL="unknown"

LABEL org.opencontainers.image.source="${CI_REPO_URL}" \
      org.opencontainers.image.revision="${CI_COMMIT_SHA}"

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /home/appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser app/ ./app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
