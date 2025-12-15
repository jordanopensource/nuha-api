# EgyNuha API

Egyptian-Arabic text classification API powered by a fine-tuned transformer model.

## Overview

EgyNuha classifies Egyptian-Arabic text into categories for detection of gender-based hate speech. It identifies neutral content, objections, discriminatory language, sexual content, and violence — with detailed sub-classifications for each category.

## Features

- **Single and batch classification** — Classify one text or up to 1000 texts in a single request
- **Multi-language responses** — Get classification labels in Arabic or English
- **High performance** — Batched inference with configurable worker threads
- **Production ready** — Docker support, health checks, structured logging

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/classify` | Classify a single text |
| `POST` | `/classify/batch` | Classify multiple texts |

### Classification Request

```bash
# Single text (Arabic labels)
curl -X POST "http://localhost:8000/classify?lang=ar" \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا كيف حالك"}'

# Single text (English labels)
curl -X POST "http://localhost:8000/classify?lang=en" \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا كيف حالك"}'

# Batch classification
curl -X POST "http://localhost:8000/classify/batch?lang=en" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["نص اول", "نص تاني", "نص تالت"]}'
```

### Response Format

```json
{
  "is_valid": true,
  "sub_class": "Neutral",
  "main_class": "Neutral",
  "confidence": 0.9842
}
```

### Classification Categories

| Main Class (EN) | Main Class (AR) | Sub-classes |
|-----------------|-----------------|-------------|
| Neutral | محايد | Neutral |
| Objection/Rejection | اعتراض/رفض | Objection/Rejection |
| Discriminatory or Offensive Language | لغة تمييزية او مهينة | Insults or Bullying, Harmful Stereotypes, Blame and Accusation |
| Sexual Content | المحتوى الجنسي | Sexual Insults, Verbal Sexual Harassment |
| Violence | العنف | Sexual Violence, Incitement/Invoking Authorities, Threats |

## Quick Start

### Using Docker (recommended)

```bash
# Pull and run the latest image
docker run -p 8000:8000 josaorg/egynuha-api:stable

# Or with custom configuration
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  -e CLASSIFIER_WORKERS=2 \
  josaorg/egynuha-api:stable
```

### Using Docker Compose

```bash
# Copy and configure environment
cp .sample.env .env

# Start the service
docker compose up -d
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the model
pip install huggingface_hub
huggingface-cli download SafwanLjd/egynuha-classifier --local-dir ./model

# Run the API
MODEL_PATH=./model uvicorn app.main:app --reload
```

## Configuration

All configuration is done through environment variables. See [`.sample.env`](.sample.env) for the complete list with descriptions.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./model` | Path to the classification model |
| `CLASSIFIER_WORKERS` | `4` | Number of inference worker threads |
| `MAX_BATCH_SIZE` | `1000` | Maximum texts per batch request |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `text` | Log format (`text` or `json`) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn worker processes |
| `TIMEOUT` | `120` | Request timeout in seconds |

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Building from Source

```bash
# Build Docker image (downloads model during build)
docker build -t egynuha-api:local .

# Build with custom model repository
docker build \
  --build-arg HF_MODEL_REPO=your-org/your-model \
  --build-arg HF_TOKEN=your-token \
  -t egynuha-api:local .
```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── classifier.py      # ML model loading and inference
│   └── main.py            # FastAPI application and endpoints
├── .woodpecker/
│   ├── build-latest-image.yaml
│   └── build-stable-image.yaml
├── .dockerignore
├── .gitignore
├── .sample.env            # Environment variable documentation
├── docker-compose.yml     # Local deployment configuration
├── Dockerfile             # Multi-stage Docker build
└── requirements.txt       # Python dependencies
```