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

One stateless api service answers everything. It reads the dialect from the
path (`/<dialect>/classify`) and dispatches to that dialect's model, which it
loaded at startup from the models volume.

```
   client ──────────────▶ ┌─────────────────────┐
                          │       nuha-api      │ :8000
                          │  routes /<dialect>/…│
                          │  arz + acm + ckb    │
                          └──────────┬──────────┘
                                     │ read-only
                          ┌──────────▼──────────┐
                          │    models volume    │  /models/<code>/
                          │  dialect.json + the │  written only by the
                          │    model snapshot   │  fetch service
                          └─────────────────────┘
```

Inside the service, one request goes through these steps:

1. Look the path's dialect up in the registry of loaded models (unknown -> 400).
2. Preprocess the text with the dialect's own cleaning rules.
3. Look the cleaned text up in that dialect's in-memory cache.
4. On a cache miss, tokenize, run the model, and store the raw prediction.
5. Derive the main class from the sub class using a fixed mapping.
6. Look up the label strings in the language you asked for and return them.

## Why it is built this way

Models are DATA, not image layers. One image serves every dialect and carries
no model; the models live on a Docker volume, one directory per dialect,
installed and removed at runtime by the fetch service. The api scans that
volume once at startup, so the operator flow for changing what the stack
serves is:

```bash
docker compose run --rm fetch add <code>    # or: remove <code>
docker compose restart api
```

No image rebuild, no compose edit, no repo change. Which dialects exist is
decided by the volume's contents, the same way the rest of the stack treats
configuration as data. A dialect whose directory is broken stays out of
service while its siblings load, and the service starts (and answers its
contract) even with an empty volume, so bootstrapping is: bring it up, fetch,
restart.

Everything that is specific to a dialect lives in one file. In the repo it is
`app/dialects/<code>.json` (the reviewed source); on the volume the fetch
command installs the same file next to the model as `dialect.json`, so the
config travels with the artifact it describes. The file holds the dialect's
name, its HuggingFace model repo, its languages, its preprocessing rules, and
its labels. The application code carries no hardcoded dialect knowledge.

## Dialects and models

| Code  | Dialect         | Model       | HuggingFace repo          | Response languages |
|-------|-----------------|-------------|---------------------------|--------------------|
| `arz` | Egyptian Arabic | BERT        | `thejosango/nuha-arz-sub-onnx` | `ar`, `en`         |
| `acm` | Iraqi Arabic    | BERT        | `thejosango/safa-acm-sub-onnx` | `ar`, `en`, `ckb`  |
| `ckb` | Sorani Kurdish  | XLM-RoBERTa | `thejosango/safa-ckb-sub-onnx` | `ar`, `en`, `ckb`  |

The dialect codes are [ISO 639-3](https://iso639-3.sil.org/) language codes:
`arz` (Egyptian Arabic), `acm` (Mesopotamian/Iraqi Arabic), and `ckb` (Central
Kurdish/Sorani). I use one standard for all three so the codes stay consistent.

The model repos follow the upstream projects: `nuha-` for Egyptian models,
`safa-` for Iraqi and Kurdish. The repos are public, so the fetch command
downloads them without a token.

The Egyptian taxonomy has 10 sub classes that roll up into 5 main classes. The
Iraqi and Kurdish dialects share the SAFA taxonomy: 13 sub classes into 6 main
classes. The label strings and the sub-to-main mapping differ per dialect and
live in each dialect file.

Iraqi and Kurdish share one preprocessing function. The differences (Iraqi
decodes leetspeak and normalizes ى to ي, Kurdish does neither) are two boolean
flags set in the dialect files, not two copies of the code.

## Running it

### The full stack with Docker Compose

```bash
# Optional: copy the sample env if you want to override any defaults.
# Every variable already has a sensible default, so this step is optional.
cp .sample.env .env

# Build the one image and start the api (the models volume starts empty).
docker compose up -d --build api

# Install the models onto the volume, then restart so the api loads them.
docker compose run --rm fetch add --all
docker compose restart api
```

The API is then on port 8000 (change it with `PORT` in `.env`). `restart`
honors the 160s stop grace, so on a live stack it drains in-flight requests
before the fresh startup scan.

Compose tags the image it builds locally `nuha-api:local`. To run a
CI-published image from the registry instead, point `NUHA_API_IMAGE` at the
repository and pick a release channel with `NUHA_API_TAG`:

```bash
NUHA_API_IMAGE=registry.cloud.josa.ngo/library/nuha-api \
NUHA_API_TAG=stable-api docker compose up -d
```

Tags are channel-first: CI publishes `stable-api` from `main` and `latest-api`
from other branches, each with a checksum-pinned `<channel>-<sha>-api` variant
for rollback, and deliberately no bare tag, so a deploy always states which
channel it follows and a work-in-progress branch push can never overwrite what
production pulls. The models are NOT in these images: a registry deploy still
populates its own volume with the fetch service.

There is a dev-host override that shrinks the api to fit an 8 GiB box:

```bash
docker compose -f compose.yml -f compose.dev.yml up -d
```

### One dialect only

Install just that dialect and nothing else; the api serves exactly what the
volume holds:

```bash
docker compose up -d --build api
docker compose run --rm fetch add arz
docker compose restart api
```

### Locally for development

```bash
# Create and activate a virtual environment.
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# Install the runtime dependencies.
pip install -r requirements.txt

# Install a model into ./models (the local-dev default for MODELS_DIR; the
# directory is created on first use).
python scripts/fetch_models.py add arz

# Run it. The startup scan loads whatever ./models holds.
uvicorn app.main:app --reload
```

Repeat the fetch for `acm` or `ckb` to serve more dialects locally. Each model
directory holds the exported ONNX graph, its tokenizer, and the installed
`dialect.json`, which is all the runtime reads.

### Tests

The tests mock the ML imports, so you do not need the models present to run
them. The suite hardcodes no dialect values: it discovers `app/dialects/*.json`
and builds throwaway model volumes from those files, so the real startup scan,
schema validation, and routing run against whatever dialects exist, whether
that directory holds one file or fifty. One run covers everything; there is no
per-dialect matrix and no selection variable.

```bash
pip install -r requirements-test.txt
pytest
```

The suite also verifies, for every dialect, that its `hf_repo` actually exists,
is public, and ships an ONNX graph (`tests/test_dialect_models.py`). That is
the one part that reaches the network; it runs by default and is a real gate (a
missing/private/non-ONNX repo fails), and it skips *only* if HuggingFace is
unreachable, so a transient outage never turns a build red. A skip otherwise
means a test's precondition doesn't apply to that dialect (e.g. rejecting a
language no other dialect serves either). CI runs this suite before the image
is built, so a failing test blocks the image push.

There is also a live smoke script that proves the frozen contract AND the
runtime add/remove story against a real stack (`scripts/e2e_smoke.sh`); it
asserts status codes only, so it is deterministic regardless of what the
models predict.

## The API

| Method | Path                        | What it does                          |
|--------|-----------------------------|---------------------------------------|
| `GET`  | `/health`                   | Liveness plus the loaded dialect codes |
| `GET`  | `/ready`                    | Readiness (200 once the startup scan ran) |
| `POST` | `/{dialect}/classify`       | Classify one text (dialect in path)   |
| `POST` | `/{dialect}/classify/batch` | Classify a list of texts (dialect in path) |

Interactive docs (`/docs`, `/redoc`, `/openapi.json`) are served unless
`DISABLE_DOCS` is set. The shipped [`.sample.env`](.sample.env) sets
`DISABLE_DOCS=1`, so a deployment that copies it (the documented first step)
runs docs-off; the docs are the one unauthenticated path that isn't
classification traffic, so keep that line in production. A bare container with
no env file serves them, which is convenient in development.

### Choosing the dialect

Every request names a dialect in the **path** (`/acm/classify`,
`/acm/classify/batch`). The valid set is exactly the dialects loaded from the
models volume; an unknown code is a 400 whose message names the loaded codes.
The dialect is always in the path and `lang` always in the request body.

### The `lang` field

- **`lang`** goes in the **request body** (`{"text": "...", "lang": "en"}`) and
  sets the language of the labels in the response, not which model runs. It
  accepts a canonical ISO 639-3 code or its two-letter alias (e.g. `ara`/`ar`,
  `eng`/`en`, `ckb`/`ku`). `ckb` is valid on the SAFA dialects (Iraqi and Kurdish);
  ask for it on Egyptian and you get a 422. If you omit it, the default is the
  first language the dialect declares alphabetically, which is Arabic for every
  shipped dialect.

### Request and response shapes

`/classify` takes one text and an optional `lang` (omit it for the dialect's
default language):

```json
{ "text": "نص للتصنيف", "lang": "ar" }
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

These run against the api on port 8000. I verified each one against a running
stack.

```bash
# Egyptian Arabic, Arabic labels (lang defaults to ar). Dialect in the path.
curl -X POST "http://localhost:8000/arz/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا كيف حالك"}'
# {"is_valid":true,"sub_class":"محايد","main_class":"محايد","confidence":0.9984}

# Iraqi Arabic, English labels (lang in the body).
curl -X POST "http://localhost:8000/acm/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "شلونك", "lang": "en"}'

# Iraqi Arabic classified, labels returned in Kurdish (both are SAFA dialects).
curl -X POST "http://localhost:8000/acm/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "شلونك", "lang": "ckb"}'

# Sorani Kurdish, Kurdish labels.
curl -X POST "http://localhost:8000/ckb/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "چۆنی", "lang": "ckb"}'

# A batch, Egyptian, English labels.
curl -X POST "http://localhost:8000/arz/classify/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["نص اول", "نص تاني", "نص تالت"], "lang": "en"}'
```

### Validation and status codes

- `text` on `/classify` must be 1 to 50000 characters. Empty text is a 422.
- `texts` on `/classify/batch` must be a non-empty list, up to `MAX_BATCH_SIZE`
  items (1000 by default). Each text is capped at 50000 characters, but there is
  no minimum: an empty string in a batch comes back with `is_valid=false` rather
  than failing the whole request.
- 422 bodies keep their `detail` list shape but never echo the rejected input
  value back.

The status codes you can get:

| Code | When                                                                       |
|------|----------------------------------------------------------------------------|
| 200  | Success                                                                     |
| 400  | Unknown `dialect` in the path (the message names the loaded codes)          |
| 404  | A classify path without a dialect segment (`/classify` is not a route)      |
| 405  | A wrong method on a classify route                                          |
| 413  | Request body over the size cap (`MAX_BODY_SIZE`, declared or chunked)       |
| 422  | Bad input or an invalid `lang` (a malformed body is a 422, never a 400)     |
| 429  | Reserved for the platform edge's rate limiter; the app itself never sends it |
| 500  | An unexpected error (the body is generic, no stack trace leaks)            |
| 503  | Overloaded: every slot is busy, the short queue is full, or a queued request waited too long |
| 504  | An inference ran past `INFERENCE_TIMEOUT`, which means something is wrong   |

The set is closed: nothing outside this table, and never a 502.

## Configuration

Configuration is by environment variable; `.env` holds the overrides and every
variable has a sensible default in the code. See [`.sample.env`](.sample.env)
for the full list with comments. Anything dialect-specific lives in the dialect
file on the volume, not here, so shared config stays shared.

| Variable             | Default                | What it does                                                                 |
|----------------------|------------------------|------------------------------------------------------------------------------|
| `MODELS_DIR`         | `./models` (compose: `/models`) | Where the model directories live; scanned once at startup.           |
| `CLASSIFIER_WORKERS` | `2`                    | How many inferences run at once, process-wide across all dialects.            |
| `INFERENCE_QUEUE_SIZE` | `32`                 | Requests that may wait for a slot before shedding 503. 0 = shed immediately.  |
| `INFERENCE_QUEUE_TIMEOUT` | `30`              | Seconds a queued request waits for a slot before a 503.                       |
| `ORT_INTRA_OP_THREADS`| `1`                   | ONNX Runtime threads per inference. Keep at 1 (see Deployment notes).         |
| `INFERENCE_TIMEOUT`  | `120`                  | Per-request inference timeout in seconds. A safety backstop, not a tuning knob.|
| `MAX_BATCH_SIZE`     | `1000`                 | Most texts allowed in one batch request.                                      |
| `MAX_BODY_SIZE`      | `10485760`             | Request body cap in bytes (10 MiB); oversized bodies get a 413.               |
| `CACHE_SIZE`         | `1024`                 | LRU capacity of the prediction cache, PER DIALECT. 0 disables it.             |
| `EXPOSE_CACHE_STATS` | *(off)*                | If set, `/health` includes per-dialect cache stats. Off so it leaks nothing.  |
| `DISABLE_DOCS`       | *(set in `.env`)*      | If set, turns off `/docs`, `/redoc`, and `/openapi.json`. The shipped `.env` sets it; remove the line to serve docs. |
| `LOG_LEVEL`          | `INFO`                 | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`.                           |
| `LOG_FORMAT`         | `text`                 | `text` for humans, `json` for log aggregation.                               |
| `PORT`               | `8000`                 | Host port compose publishes.                                                  |
| `TIMEOUT`            | `120`                  | Uvicorn keep-alive timeout in seconds. Not a request timeout.                 |

Compose-level variables: `NUHA_API_IMAGE` (default `nuha-api`) and
`NUHA_API_TAG` (default `local`) select the image; `API_CPU_LIMIT` (default
`3.0`) and `API_MEM_LIMIT` (default `10g`) cap the api container.

## Deployment notes

### The models volume is the control surface

The volume holds one directory per dialect (`/models/<code>/`), containing the
installed `dialect.json` and the model snapshot. The api mounts it READ-ONLY
and scans it once at startup; the fetch service is the only writer and the
only piece that needs network egress to HuggingFace. The startup scan ignores
anything that does not look like a dialect directory (the fetch command stages
downloads in dot-prefixed directories and activates them with an atomic
rename), validates each `dialect.json` against the shared schema, checks the
model's declared output width against its labels, and runs one tiny
self-inference before a dialect is allowed to serve, so a broken or
mis-packaged directory stays out while its siblings load.

The runbook:

```bash
docker compose run --rm fetch add <code>        # install or upgrade (re-fetch)
docker compose run --rm fetch add xyz --file xyz.json   # trial a new dialect
docker compose run --rm fetch add <code> --revision <r> # pin a model revision
docker compose run --rm fetch remove <code>
docker compose run --rm fetch list
docker compose restart api                      # pick any of the above up
```

Model revisions are deliberately unpinned by default, so a re-fetch picks up
the HF repo's head without a code change (the same trade-off the old
build-time bake had, now without the rebuild). `restart` drains: the 160s
`stop_grace_period` sits above the engine's worst-case admitted request
(30s queue wait + 120s inference), so in-flight work finishes before the fresh
scan. A single instance is briefly down while it reloads; with replicas behind
the platform edge, restart them serially.

### The edge

There is no reverse proxy in this stack; the api is the single public service.
TLS termination, per-IP rate limiting (the 429 in the status table), and
slow-client handling belong to the platform edge in front of it. The app
carries its own body-size cap, security headers, and overload shedding, so a
directly exposed container still bounds itself.

### Inference, concurrency, and overload

Inference is CPU bound, so the tuning rule is one in-flight inference per CPU
the container is allowed. Keep `CLASSIFIER_WORKERS` about one below
`API_CPU_LIMIT` and keep `ORT_INTRA_OP_THREADS` at 1. The product of the two
should stay near the CPU limit: that runs `CLASSIFIER_WORKERS` single-threaded
inferences, one per core. To favour fewer but faster (multi-threaded) batches
over concurrency, raise `ORT_INTRA_OP_THREADS` and lower `CLASSIFIER_WORKERS`
to keep that product the same. The headroom above the slots keeps the event
loop, tokenization, and health checks responsive while every slot is busy.

The reason for pinning the per-inference threads is a real trap, and it is the
same one the old torch build had. Left to its default, ONNX Runtime sizes its
intra-op thread pool to the host core count and ignores the Docker CPU cap.
Under that cap the kernel throttles the container and every inference slows
down. Pinning `ORT_INTRA_OP_THREADS` to 1 keeps each inference on a single
core, so the slots run one per core with no oversubscription.

The admission gate is process-wide and shared by every loaded dialect: at most
`CLASSIFIER_WORKERS` inferences run at once; a request that finds every slot
busy waits up to `INFERENCE_QUEUE_TIMEOUT` seconds for one to free instead of
being shed immediately, so a short burst is served (200) rather than rejected
the instant both workers are busy. Once `CLASSIFIER_WORKERS +
INFERENCE_QUEUE_SIZE` requests are in flight, or a queued request waits past
the timeout, the api sheds a fast 503. One consequence of the shared gate: a
burst on one dialect sheds siblings' requests too, because the capacity being
protected is the process's CPU, not a per-dialect budget. The queue smooths
bursts within capacity; it does not add throughput, so scale with replicas to
raise the ceiling. There is also a per-request timeout: an inference that runs
past `INFERENCE_TIMEOUT` returns a 504. That timeout is a backstop for
something genuinely stuck, not a tuning knob; a full 1000-text batch is a
single inference that can take tens of seconds on a 2-CPU cap, and the default
120s leaves wide margin.

Uvicorn worker processes are fixed at 1 in the image entrypoint (there is no
`WORKERS` variable). The model forward pass releases the GIL, so one process
already saturates the CPU the container has; a second worker would be a full
copy of every model in memory for no throughput gain. Raise
`CLASSIFIER_WORKERS` (with `API_CPU_LIMIT`) for more simultaneity per
container, and scale with replicas beyond that.

### Memory

One process holds every loaded model, so the container's footprint is the SUM
of the models on the volume: under a heavy stress load the real peaks are
around 1.5 GiB for the BERT dialects (`arz`, `acm`) and a bit over 2 GiB for
`ckb` (XLM-RoBERTa has a larger multilingual embedding matrix), so the shipped
three peak near 5-6 GiB together plus the web stack and in-flight bodies. The
`API_MEM_LIMIT` default (10g) is a comfortable ceiling, not a reservation;
`compose.dev.yml` caps it at 7g for an 8 GiB host. Installing more dialects
raises the real footprint: raise the limit with the volume.

### Scaling

The api is stateless (the volume is read-only data), so capacity scales by
running more replicas of the one service. On a single compose host the
published port pins one instance; real horizontal scaling runs replicas behind
the platform edge or an orchestrator, where each replica mounts the same
models and serves the same dialect set. Note the unit of scaling is the whole
service: every replica loads every model, so one hot dialect cannot be scaled
alone. That is the deliberate trade of this design; the win is that replicas
are interchangeable and the tech team can add or remove containers freely
without any per-dialect wiring.

True load-based autoscaling needs an orchestrator (Kubernetes with an HPA, or
KEDA). The pieces map directly: the image is the Deployment, the models volume
is a PVC (or an init container running the fetch command), and `/ready` is the
readiness probe.

## Adding a dialect

The whole design points at this being easy. To add a dialect you install one
directory on the volume and restart. No application code changes, no rebuild,
no compose or CI edit.

1. Write the dialect's config file: a `name`, an `hf_repo`, a `languages`
   block, a `preprocessing` block, and a `labels` block (sub and main labels
   per language, plus the sub-to-main mapping). Copy an existing
   `app/dialects/<code>.json` as a starting point. For a dialect the repo
   should ship, commit it under `app/dialects/` (the `validate-dialects`
   pre-commit hook checks it structurally); for a trial, any local file works.
2. Install it and restart:

   ```bash
   docker compose run --rm fetch add <code>              # a committed dialect
   docker compose run --rm fetch add <code> --file x.json  # a trial one
   docker compose restart api
   ```

The fetch command validates the config against the shared schema BEFORE
downloading, and the startup scan revalidates on load, so a broken file never
serves. The only time you touch Python is if the dialect needs a brand new
preprocessing family. In that case add a function to `_PREPROCESS_REGISTRY` in
`app/classifier.py` and reference it by name in the dialect file's
`preprocessing.type`. The existing `nuha` and `safa` preprocessors cover the
current dialects.

## Build

The Dockerfile builds ONE image with no model in it, in two stages:

1. **Dependencies.** Install the Python packages into a virtual environment
   from `requirements.lock` with `--require-hashes`.
2. **Runtime.** A slim image with the virtual environment, the app, the repo's
   dialect configs (the fetch command's install source), and the fetch script,
   running as a non-root user. uvicorn runs as PID 1 for clean signal handling.
   `/models` is created owned by the app user so the volume's first-use
   initialization lets the fetch service write to it.

```bash
docker build -t nuha-api:local .
docker run -p 8000:8000 -v models:/models:ro nuha-api:local
```

The image is CPU-only and model-free; the same image runs the api and the
fetch service, so there is exactly one artifact to build, scan, and ship.

### Dependencies and the lockfile

`requirements.txt` is the human-edited list of direct dependencies. Inference
runs on ONNX Runtime, whose `onnxruntime` wheel is a normal PyPI package, so the
file needs no custom wheel index (unlike the old PyTorch `+cpu` build, which had
to declare PyTorch's CPU wheel index). `transformers` is still a dependency, but
only for its `AutoTokenizer`; the models are exported to ONNX and loaded through
`onnxruntime`, not through transformers' model classes. `huggingface_hub` (the
fetch command's downloader) is already in the lock as a transitive dependency
of transformers.

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
  main.py              FastAPI app, endpoints, validation, exception handlers
  classifier.py        The engine: model loading, preprocessing, cache, gate
  registry.py          The startup scan of the models volume
  common/              Shared plumbing: config parsing, HTTP middleware and
                       handlers, schemas, logging, and the dialect-file schema
                       (dialect_schema.py, the single definition)
  dialects/            One reviewed file per dialect (arz/acm/ckb).json: name,
                       hf_repo, languages (each with a display name and
                       aliases), preprocessing, and labels. The fetch command's
                       install source; the runtime reads the volume's copies.
tests/                 pytest suite (the ML imports are mocked; one run)
scripts/
  fetch_models.py      Install/remove/list models on the volume (the compose
                       fetch service's entrypoint)
  validate_dialects.py Validate app/dialects/ (the pre-commit hook)
  check_lock.py        Lockfile drift gate
  e2e_smoke.sh         Live smoke: contract + the runtime add/remove story
Dockerfile             Two-stage, model-free build (one image for everything)
compose.yml            The api service + the fetch service + the models volume
compose.dev.yml        8 GiB dev-host override
.woodpecker/           CI pipelines (one image per channel)
requirements.txt       Direct dependencies (onnxruntime, transformers, ...)
requirements.lock      Generated, hashed lock installed by the Dockerfile
requirements-test.txt  Test dependencies
.env                   Runtime config overrides
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
Commit messages. The `validate-dialects` hook checks every `app/dialects/*.json`
against the shared schema (required fields and types, a well-formed `hf_repo`
id, label/language consistency, a sound `sub_to_main` mapping) and rejects the
commit with a clear message if one is broken, so a bad dialect file is caught
at commit time rather than at install or load time. `run-samplr` keeps
`.sample.env` in step with `.env`. (The tests remain the full gate and run in
CI; the hook's check is the fast structural subset that needs no dependencies.)

## Response contract

- The response schema is `is_valid`, `sub_class`, `main_class`, `confidence`.
- Status codes: 200 for a served request and 422 for invalid input; 400
  (unknown dialect), 404 (no dialect segment), 405 (wrong method), 413 (body
  too large), 429 (platform edge only), 500, 503, and 504 cover the error
  cases. The set is closed and a 502 never occurs. 422 bodies keep their
  `detail` list shape but do not echo the rejected input value back.

## Next steps

- Explore container orchestration and autoscaling. The service is already the
  right unit for it: stateless replicas over a read-only models volume, with
  `/ready` as the probe. Kubernetes with an HPA (or KEDA) could scale on load,
  with the fetch command running as a Job or init container against a PVC.
- Hot reload of the models volume (a periodic rescan instead of the restart)
  is a contained follow-up if the restart step ever becomes a burden; the
  registry's scan is already the seam it would build on.
