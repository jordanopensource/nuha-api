# Nuha API

Nuha is a text classification API for detecting gender-based hate speech in
Arabic and Kurdish. It serves three dialects, each with its own fine-tuned
transformer model: Egyptian Arabic, Iraqi Arabic, and Sorani Kurdish. You send
it text, it tells you whether the text is hateful and which category it falls
into.

It is built for teams that need to screen or moderate user content at scale. The
classification taxonomy comes from the SAFA project, and the labels are the
authoritative names from that taxonomy. You can get those labels back in Arabic,
English, or Kurdish without changing which model runs.

## How a request flows

A single nginx proxy sits in front of three backend containers, one per dialect.
The proxy reads the `?dialect=` query parameter and routes the request to the
matching backend. If you leave `dialect` off, it falls back to Egyptian so older
clients keep working.

```
                         ┌────────────────────┐
   client ──────────────▶│     nginx proxy    │ :8000
                         │  routes ?dialect=  │
                         └──────────┬─────────┘
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
       ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
       │ nuha-api-arz │      │ nuha-api-acm │      │ nuha-api-ckb │  :8000 each
       │ DIALECT=arz  │      │ DIALECT=acm  │      │ DIALECT=ckb  │
       │  × replicas  │      │  × replicas  │      │  × replicas  │
       └──────────────┘      └──────────────┘      └──────────────┘
```

Inside a backend, one request goes through these steps:

1. Preprocess the text with the dialect's own cleaning rules.
2. Look the cleaned text up in an in-memory cache.
3. On a cache miss, tokenize, run the model, and store the raw prediction.
4. Derive the main class from the sub class using a fixed mapping.
5. Look up the label strings in the language you asked for and return them.

## Why it is built this way

I build one image per dialect, each baking in only that dialect's model. A single
`Dockerfile` does all three: it takes a `DIALECT` build argument, reads that
dialect's model repo from its dialect file, and downloads just that one model.
The dialect code becomes the image tag (`nuha-api:arz`, `nuha-api:acm`,
`nuha-api:ckb`). At runtime the `DIALECT` environment variable selects the same
dialect, so the container loads the model that is actually present.

Each image carries exactly one model, so it is far smaller than a single image
holding all three would be (about 1.0 GB for a BERT dialect, 1.6 GB for the
larger Kurdish model). It also means I can build, ship, and scale each dialect on
its own. If Iraqi traffic spikes, I add Iraqi replicas without touching the
others.

Everything that is specific to a dialect lives in one file: `app/dialects/<code>.json`.
That file holds the dialect's name, its HuggingFace model repo, its languages, its
preprocessing rules, its memory limit, its replica count, and its labels. The
application code carries no hardcoded dialect knowledge. Adding a dialect is a
one-file change, which I cover below.

## Dialects and models

| Code  | Dialect         | Model       | HuggingFace repo          | Response languages |
|-------|-----------------|-------------|---------------------------|--------------------|
| `arz` | Egyptian Arabic | BERT        | `thejosango/nuha-arz-sub` | `ar`, `en`         |
| `acm` | Iraqi Arabic    | BERT        | `thejosango/safa-acm-sub` | `ar`, `en`         |
| `ckb` | Sorani Kurdish  | XLM-RoBERTa | `thejosango/safa-ckb-sub` | `ar`, `en`, `ckb`  |

The dialect codes are [ISO 639-3](https://iso639-3.sil.org/) language codes:
`arz` (Egyptian Arabic), `acm` (Mesopotamian/Iraqi Arabic), and `ckb` (Central
Kurdish/Sorani). I use one standard for all three so the codes stay consistent.

The model repos follow the upstream projects: `nuha-` for Egyptian models,
`safa-` for Iraqi and Kurdish. The repos are public, so the build downloads them
without a token.

The Egyptian taxonomy has 10 sub classes that roll up into 5 main classes. The
Iraqi and Kurdish dialects share the SAFA taxonomy: 13 sub classes into 6 main
classes. The label strings and the sub-to-main mapping differ per dialect and
live in each dialect file.

Iraqi and Kurdish share one preprocessing function. The differences (Iraqi
decodes leetspeak and normalizes ى to ي, Kurdish does neither) are two boolean
flags set in the dialect files, not two copies of the code.

## Running it

### The full stack with Docker Compose

This is the way to run all three dialects behind the proxy.

```bash
# Optional: copy the sample env if you want to override any defaults.
# Every variable already has a sensible default, so this step is optional.
cp .sample.env .env

# Build the three per-dialect images (each bakes in only its own model).
docker compose build

# Start the three dialect backends and the nginx proxy.
docker compose up -d
```

The API is then on port 8000 (change it with `PORT` in `.env`). The proxy waits
for all three backends to report healthy before it accepts traffic.

Compose tags the images it builds `nuha-api:arz`, `nuha-api:acm`, and
`nuha-api:ckb`. To run images from a registry instead of building locally, set
`NUHA_IMAGE_PREFIX` to the repository path; Compose appends `:<code>` per
dialect. For the JOSA registry, where CI publishes the per-dialect images:

```bash
NUHA_IMAGE_PREFIX=registry.cloud.josa.ngo/library/nuha-api docker compose up -d
```

That resolves to `…/nuha-api:arz`, `…/nuha-api:acm`, and `…/nuha-api:ckb`. CI
also pushes channel- and commit-pinned tags per dialect (`arz-latest`,
`arz-stable`, `arz-stable-<sha>`, and so on) for rollback; pin a specific tag by
setting the image directly in your own override file.

### One dialect in a single container

If you only need one dialect, run that dialect's image directly. `DIALECT` is
required, and it must match the dialect the image was built for (the image bakes
in only that one model). Each image already defaults `DIALECT` to the dialect it
was built for, so a bare run works.

```bash
docker run -p 8000:8000 nuha-api:arz

# An image from the registry, with a couple of overrides:
docker run -p 8000:8000 \
  -e DIALECT=acm \
  -e LOG_LEVEL=DEBUG \
  -e CLASSIFIER_WORKERS=2 \
  registry.cloud.josa.ngo/library/nuha-api:acm
```

### Locally for development

```bash
# Create and activate a virtual environment.
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Install the runtime dependencies.
pip install -r requirements.txt

# Download the ONNX model for the dialect you want to run.
pip install huggingface_hub
hf download thejosango/nuha-arz-sub-onnx --local-dir ./models/arz

# Run it. DIALECT is required; the model must exist at ./models/{DIALECT}.
DIALECT=arz uvicorn app.main:app --reload
```

Repeat the download step with `thejosango/safa-acm-sub-onnx` into `./models/acm`
or `thejosango/safa-ckb-sub-onnx` into `./models/ckb` for the other dialects. The
repos hold the exported ONNX graph and its tokenizer, which is all the runtime
needs. If your models live somewhere else, point `MODEL_PATH` at the directory.

### Tests

The tests mock the ML imports, so you do not need the models present to run them.
`DIALECT` is still required because the app reads it at import time.

```bash
pip install -r requirements-test.txt

DIALECT=arz pytest    # 268 passed, a few skipped
DIALECT=acm  pytest    # 268 passed, a few skipped
DIALECT=ckb pytest    # 267 passed, a few skipped
```

The `ckb` suite skips a few extra cases: they cover the Kurdish-only `lang=ckb`
path, which the Arabic-and-English dialects do not have.

## The API

| Method | Path              | What it does                |
|--------|-------------------|-----------------------------|
| `GET`  | `/health`         | Liveness check              |
| `POST` | `/classify`       | Classify one text           |
| `POST` | `/classify/batch` | Classify a list of texts    |

Interactive docs are at `/docs` (Swagger UI) and `/redoc` when the stack is up.

### Query parameters

Both classify endpoints take the same two query parameters.

- **`dialect`** is optional. It is `arz`, `acm`, or `ckb`. The proxy uses it to
  pick the backend. Each backend also checks it: if you send a `dialect` that
  does not match the container, you get a 422. Leave it off and the request goes
  to the default dialect (Egyptian).
- **`lang`** defaults to `ar`. It sets the language of the labels in the
  response, not which model runs. It is `ar`, `en`, or `ckb`. `ckb` is only valid
  on the Kurdish dialect; ask for it anywhere else and you get a 422.

### Request and response shapes

`/classify` takes one text:

```json
{ "text": "نص للتصنيف" }
```

and returns:

```json
{
  "is_valid": true,
  "sub_class": "Neutral",
  "main_class": "Neutral",
  "confidence": 0.9984
}
```

`is_valid` is `false` when the text does not survive preprocessing (for example
it is empty after cleaning, or it is only emojis). In that case the three other
fields are `null`.

`/classify/batch` takes a list and returns one result per input, in order:

```json
{ "texts": ["نص أول", "نص ثاني"] }
```

```json
{
  "results": [
    { "is_valid": true, "sub_class": "Neutral", "main_class": "Neutral", "confidence": 0.4863 },
    { "is_valid": true, "sub_class": "Neutral", "main_class": "Neutral", "confidence": 0.9798 }
  ]
}
```

### curl examples

These run against the proxy on port 8000. I verified each one against a running
container.

```bash
# Egyptian Arabic, Arabic labels (lang defaults to ar).
curl -X POST "http://localhost:8000/classify?dialect=arz" \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا كيف حالك"}'
# {"is_valid":true,"sub_class":"محايد","main_class":"محايد","confidence":0.9984}

# Iraqi Arabic, English labels.
curl -X POST "http://localhost:8000/classify?dialect=acm&lang=en" \
  -H "Content-Type: application/json" \
  -d '{"text": "شلونك"}'

# Sorani Kurdish, Kurdish labels.
curl -X POST "http://localhost:8000/classify?dialect=ckb&lang=ckb" \
  -H "Content-Type: application/json" \
  -d '{"text": "چۆنی"}'

# A batch, Egyptian, English labels.
curl -X POST "http://localhost:8000/classify/batch?dialect=arz&lang=en" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["نص اول", "نص تاني", "نص تالت"]}'
```

### Validation and status codes

- `text` on `/classify` must be 1 to 50000 characters. Empty text is a 422.
- `texts` on `/classify/batch` must be a non-empty list, up to `MAX_BATCH_SIZE`
  items (1000 by default). Each text is capped at 50000 characters, but there is
  no minimum: an empty string in a batch comes back with `is_valid=false` rather
  than failing the whole request.

The status codes you can get:

| Code | When                                                                       |
|------|----------------------------------------------------------------------------|
| 200  | Success                                                                     |
| 400  | The proxy received an unknown `dialect` value                               |
| 422  | Bad input, a `dialect` that does not match the backend, or an invalid `lang`|
| 500  | An unexpected error (the body is generic, no stack trace leaks)            |
| 503  | Every inference slot is busy (the service sheds load instead of queuing)   |
| 504  | An inference ran past `INFERENCE_TIMEOUT`, which means something is wrong   |

The 400, 500, 503, and 504 codes are additions. The original 200 and 422
behavior is unchanged, so existing clients keep working.

## Configuration

Configuration is by environment variable. `.env` holds the settings that are the
same across all three dialects; the per-dialect settings live in the dialect
files instead. See [`.sample.env`](.sample.env) for the full list with comments.
Anything dialect-specific (memory limit, replica count) is in
`app/dialects/<code>.json`, not here, so shared config stays shared.

| Variable             | Default                | What it does                                                                 |
|----------------------|------------------------|------------------------------------------------------------------------------|
| `DIALECT`            | *(required)*           | Which dialect this container serves: `arz`, `acm`, or `ckb`. No default.       |
| `MODEL_PATH`         | `./models/{DIALECT}`   | Where to load the model from. Derived from `DIALECT` if unset.                |
| `CLASSIFIER_WORKERS` | `2`                    | Concurrent inference slots. Sizes both the thread pool and the 503 gate.      |
| `ORT_INTRA_OP_THREADS`| `1`                   | ONNX Runtime threads per inference. Keep at 1 (see Deployment notes).         |
| `INFERENCE_TIMEOUT`  | `120`                  | Per-request inference timeout in seconds. A safety backstop, not a tuning knob.|
| `MAX_BATCH_SIZE`     | `1000`                 | Most texts allowed in one batch request.                                      |
| `CACHE_SIZE`         | `1024`                 | LRU capacity of the prediction cache. 0 disables it.                          |
| `EXPOSE_CACHE_STATS` | *(off)*                | If set, `/health` includes cache hit and miss stats. Off so it leaks nothing. |
| `DISABLE_DOCS`       | *(off)*                | If set, turns off `/docs`, `/redoc`, and `/openapi.json`.                     |
| `LOG_LEVEL`          | `INFO`                 | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.                           |
| `LOG_FORMAT`         | `text`                 | `text` for humans, `json` for log aggregation.                               |
| `HOST`               | `0.0.0.0`              | Bind address.                                                                 |
| `PORT`               | `8000`                 | Port the proxy publishes.                                                     |
| `WORKERS`            | `1`                    | Uvicorn worker processes. Keep at 1 (see Deployment notes).                   |
| `TIMEOUT`            | `120`                  | Uvicorn keep-alive timeout in seconds. Not a request timeout.                 |

A couple of proxy-only variables: `DEFAULT_DIALECT` (default `arz`) is the
dialect the proxy uses for no-dialect requests and for the docs pages.
`NUHA_IMAGE_PREFIX` (default `nuha-api`) is the repository prefix for the
per-dialect backend images; Compose appends `:<code>` to it. `BACKEND_CPU_LIMIT`,
`PROXY_CPU_LIMIT`, `PROXY_MEM_LIMIT`, and `BACKEND_MEM_RESERVATION` set Compose
resource caps.

## Deployment notes

### The proxy

nginx terminates client connections and routes by dialect. It also:

- Rate limits per IP: 100 requests a second to `/classify`, 20 a second to
  `/classify/batch`, since batches are heavier. Over the limit gets a 429.
- Sets `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`.
- Serves its own `/health` with a 200 instead of proxying to a backend.
- Serves the docs from the default dialect's backend, since docs do not depend on
  the dialect.
- Accepts request bodies up to 25 MiB, which covers a maximal batch of Arabic or
  Kurdish text.
- Keeps client-facing timeouts tight, but allows a long read timeout (125s) on
  the classify routes so a slow batch is not cut off. That timeout sits just
  above `INFERENCE_TIMEOUT` so the app returns its own clean 504 first and nginx
  is only a fallback.

The routing config is generated from the dialect files, so the proxy never has a
hardcoded list of dialects (more on that below).

### Inference, concurrency, and overload

Inference is CPU bound, so the tuning rule is one in-flight inference per CPU the
container is allowed. Keep `CLASSIFIER_WORKERS` equal to the backend's CPU limit
and keep `ORT_INTRA_OP_THREADS` at 1. The product of the two should stay near
the CPU limit: that runs `CLASSIFIER_WORKERS` single-threaded inferences, one per
core. To favour fewer but faster (multi-threaded) batches over concurrency, raise
`ORT_INTRA_OP_THREADS` and lower `CLASSIFIER_WORKERS` to keep that product the
same.

The reason for pinning the per-inference threads is a real trap, and it is the
same one the old torch build had. Left to its default, ONNX Runtime sizes its
intra-op thread pool to the host core count and ignores the Docker CPU cap. Under
that cap the kernel throttles the container and every inference slows down. With
`CLASSIFIER_WORKERS` inferences in flight, each also trying to use every host
core, they oversubscribe the CPU badly. Pinning `ORT_INTRA_OP_THREADS` to 1 keeps
each inference on a single core, so the slots run one per core with no
oversubscription. (This single variable replaces the two torch-only
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` knobs the PyTorch build used.)

Each backend protects itself with a non-blocking gate sized to
`CLASSIFIER_WORKERS`. When every slot is busy, the next request gets an immediate
503 rather than piling up in a queue. There is also a per-request timeout: an
inference that runs past `INFERENCE_TIMEOUT` returns a 504. That timeout is a
backstop for something genuinely stuck, not a tuning knob, so set it well above
your worst-case batch time. A full 1000-text batch is a single inference that can
take tens of seconds on a 2-CPU cap, and the default 120s leaves wide margin.

Keep `WORKERS` (uvicorn processes) at 1. The model forward pass releases the GIL,
so one process already saturates the CPU the container has. A second worker is a
full copy of the model in memory for no throughput gain. Scale with replicas, not
workers.

### Memory

Each dialect's memory limit is the `mem_limit` field in its dialect file,
rendered into `compose.yml`. They are all set to `4g` right now. That is a
deliberate, comfortable ceiling, not a measured requirement. Under a heavy stress
load the real peaks are around 1.5 GiB for the BERT dialects (`arz`, `acm`) and a
bit over 2 GiB for `ckb`, which is heavier because XLM-RoBERTa has a larger
multilingual embedding matrix. A limit is a ceiling, not a reservation, so
choosing `4g` does not reserve 12 GiB. Size the host to the real peak (roughly
1.5 to 2 GiB per dialect plus the OS), not to the sum of the ceilings.

### Scaling

Each dialect is a Compose service that scales to N replicas. The proxy
round-robins across them automatically: a scaled service name resolves through
Docker DNS to all of its replica IPs, and nginx spreads requests across them. No
proxy change is needed to add capacity. The load balancing is round-robin rather
than least-connections, which is fine here because each replica has its own 503
gate, so an overloaded replica sheds and the retry lands on another.

The baseline replica count is the `replicas` field in each dialect file (default
1). To change it for good, edit that field and re-render the config. For a
temporary burst, for example during a workshop, override it at runtime with no
rebuild:

```bash
docker compose up -d --scale nuha-api-arz=10   # ramp up
docker compose up -d --scale nuha-api-arz=1    # back to baseline
```

Sizing: each replica runs `CLASSIFIER_WORKERS` concurrent inferences, so to serve
about N batches at once on a dialect, set its replicas to roughly
N / `CLASSIFIER_WORKERS`. The host needs `sum(replicas) × BACKEND_CPU_LIMIT`
cores at peak. Keep at least 1 replica per dialect, since a dialect at 0 replicas
fails its requests until you scale it back up.

This is on-demand scaling that you drive by hand or by script. It fits a bursty
workload like scheduled events well. True load-based autoscaling, including
scale-to-zero, needs an orchestrator (Kubernetes with an HPA, or KEDA or
Knative). The per-replica unit tuned here is exactly what such an autoscaler would
replicate, so that work carries over. I have not built it; it is larger infra
than the Compose model.

## Adding a dialect

The whole design points at this being easy. To add a dialect you drop one file
and rebuild. No application code changes.

1. Create `app/dialects/<code>.json`, where `<code>` is the new dialect's code
   (it becomes the dialect's identifier). Give it a `name`, an `hf_repo`, a
   `languages` list, a `preprocessing` block, a `mem_limit`, a `replicas` count,
   and a `labels` block (sub and main labels per language, plus the sub-to-main
   mapping). Copy an existing dialect file as a starting point.
2. Re-render the generated config:

   ```bash
   python scripts/render_config.py
   ```

   This regenerates `compose.yml` (a new backend service for your dialect) and
   the nginx routing (a new route for `?dialect=<code>`). The `render-config`
   pre-commit hook also does this for you on commit, so you usually do not run it
   by hand.
3. Build the image for the new dialect with `docker build --build-arg
   DIALECT=<code> -t nuha-api:<code> .` (or `docker compose build` to build them
   all). The model-download stage reads the dialect file and pulls your new
   `hf_repo` automatically.

The classifier globs `app/dialects/*.json` at import, so it picks up the new file
with no edit. The only time you touch Python is if the dialect needs a brand new
preprocessing family. In that case add a function to `_PREPROCESS_REGISTRY` in
`app/classifier.py` and reference it by name in the dialect file's
`preprocessing.type`. The existing `nuha` and `safa` preprocessors cover the
current dialects.

`compose.yml` and `nginx/dialects.conf.template` are generated. Do not hand-edit
them. Edit the dialect files (or `compose.template.yml` for proxy-level changes)
and re-render.

## Build

The Dockerfile builds one image per dialect. It takes a `DIALECT` build argument
that selects which single model to bake in, in three stages:

1. **Dependencies.** Install the Python packages into a virtual environment from
   `requirements.lock`.
2. **Model download.** Read the chosen dialect's `app/dialects/<code>.json`,
   download that one model from its `hf_repo`, and bake only it. The repos are
   public, so this needs no token.
3. **Runtime.** A slim image with the virtual environment, the one model, and the
   app, running as a non-root user. uvicorn runs as PID 1 for clean signal
   handling. The image defaults `DIALECT` to the one it was built for.

```bash
# Build one dialect's image (downloads just that dialect's model).
docker build --build-arg DIALECT=arz -t nuha-api:arz .

# Run what you built.
docker run -p 8000:8000 nuha-api:arz
```

`docker compose build` does this for all three dialects at once, producing
`nuha-api:arz`, `nuha-api:acm`, and `nuha-api:ckb`. The images are CPU-only and
sized to their one model: about 1.0 GB for `arz`, 1.1 GB for `acm`, and 1.6 GB
for the larger XLM-RoBERTa `ckb` model.

### Dependencies and the lockfile

`requirements.txt` is the human-edited list of direct dependencies. Inference
runs on ONNX Runtime, whose `onnxruntime` wheel is a normal PyPI package, so the
file needs no custom wheel index (unlike the old PyTorch `+cpu` build, which had
to declare PyTorch's CPU wheel index). `transformers` is still a dependency, but
only for its `AutoTokenizer`; the models are exported to ONNX and loaded through
`onnxruntime`, not through transformers' model classes.

`requirements.lock` is generated from `requirements.txt`. It is a fully pinned,
fully hashed lock of the whole dependency tree. The Dockerfile installs the lock
with `pip install --require-hashes`, so every build pulls the exact same versions
and verifies each wheel's hash. The build is reproducible and tamper-evident.

Regenerate the lock whenever you change `requirements.txt`. The command is in the
lock file's header:

```bash
docker run --rm \
  -v "$PWD/requirements.txt:/in/requirements.txt:ro" -v "$PWD:/out" \
  python:3.12-slim sh -c \
  "pip install pip-tools==7.5.3 && cd /in && pip-compile --generate-hashes \
   --allow-unsafe --no-strip-extras --output-file=/out/requirements.lock requirements.txt"
```

`transformers` is pinned to `5.2.0`. Do not bump it without re-validating against
the real models; the model files and the library version are matched.

## Project layout

```
app/
  classifier.py        Model loading, preprocessing, inference cache, active config
  main.py              FastAPI app, endpoints, validation, exception handlers
  dialects/            One self-contained file per dialect (arz/acm/ckb).json:
                       name, hf_repo, languages (each with a display name and
                       aliases), preprocessing, mem_limit, replicas, and labels.
                       The single source of truth.
tests/                 pytest suite (the ML imports are mocked)
scripts/
  render_config.py     Regenerates compose.yml and the nginx routing from dialects/
nginx.conf             Static proxy config; includes the generated routing
nginx/
  dialects.conf.template  Generated routing map (one entry per dialect)
Dockerfile             Three-stage per-dialect build (one model baked in)
compose.template.yml   Hand-edited source for compose.yml
compose.yml            Generated: three backends plus the proxy
requirements.txt       Direct dependencies (onnxruntime, transformers, ...)
requirements.lock      Generated, hashed lock installed by the Dockerfile
requirements-test.txt  Test dependencies
.env                   Shared runtime config (per-dialect config is in dialects/)
.sample.env            Generated from .env by samplr
```

## Pre-commit hooks

The repo uses [pre-commit](https://pre-commit.com/).

```bash
pip install pre-commit
pre-commit install --install-hooks
pre-commit run --all-files          # optional, run on everything once
```

The hooks cover Ruff (Python lint and format), YAML and TOML syntax, trailing
whitespace and merge markers, secret detection (Gitleaks), and Conventional
Commit messages. The `render-config` hook keeps `compose.yml` and the nginx
routing in step with the dialect files, and `run-samplr` keeps `.sample.env` in
step with `.env`.

## Backward compatibility

The API stays compatible with the existing frontend:

- A request with no `dialect` routes to Egyptian.
- The response schema is unchanged: `is_valid`, `sub_class`, `main_class`,
  `confidence`.
- The original status codes (200, 422) behave the same. The new codes (400, 500,
  503, 504) are additions.

## Next steps

- Explore container orchestration and autoscaling. Scaling today is manual: I set
  the replica counts by hand or with a script. An orchestrator like Kubernetes
  (with an HPA) or KEDA could scale each dialect automatically on load, including
  scaling to zero between bursts. The per-replica unit is already tuned, so that
  work carries straight over.
