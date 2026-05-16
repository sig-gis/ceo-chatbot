# CEO Chatbot
Project setup helper (LLM chatbot) for CEO

A chatbot that answers questions about Collect Earth Online by retrieving
relevant sections from the official documentation, then having a language
model summarize them.

## What this repo does

The system has three jobs that run independently:

1. **Extract** - copies the latest CEO docs into a Google Cloud Storage bucket.

2. **Pipeline** - reads the docs from GCS, splits them into chunks, computes
   numerical embeddings, and saves a searchable index back to GCS. Runs after
   extract, or whenever you want to rebuild the index.
3. **Chatbot** - a web service that downloads the index from GCS and answers
   user questions by searching the index and asking Gemini to summarize the
   matching chunks.

# 🧱 Repository layout

This repo is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
one git repo holding multiple Python projects that share a single lockfile but
each have their own `pyproject.toml` and dependency list. The three jobs above
map to three workspace members, plus a small shared library:

- `packages/ceo_chatbot_core/` - shared library. Holds `GCSStorage` (the
  Google Cloud Storage sync helper used by every service) and a small YAML
  loader. Anything used by more than one service lives here.
- `services/ceo_ingest_docs/` - the **extract** service. Clones the CEO docs
  repo from GitHub and uploads it to GCS. Installs an `ingest-docs` command.
- `services/ceo_build_index/` - the **pipeline** service. Reads docs from GCS
  (or from a local copy), builds a FAISS index, saves it locally, and uploads
  the index files to GCS. Installs a `build-index` command. Pinned to CPU
  PyTorch so the install stays small.
- `services/ceo_rag_chatbot/` - the **chatbot** service. Loads the RAG db, embeds queries, runs retrieval, and queries Gemini for inference. Installs a `chatbot` command. Pinned to CPU PyTorch so the install stays small.
- The root project (`pyproject.toml`, `app/`, `src/ceo_chatbot/`) - currently
  holds the FastAPI **chatbot** service and the RAG glue code. A future
  refactor will move it into its own workspace member too.

The `demo/chainlit/chainlit-demo/` directory is a small standalone Chainlit
app that provides a browser chat UI. It is not a workspace member - instructions on running it are available below. It expects the FastAPI server to be running
service to be running at `http://localhost:8080` and proxies questions to it.

Why this shape: each service declares only the deps it actually needs, so a
Docker image for one service does not pull in the other services' libraries
(no torch-CUDA wheels in the chatbot image, no FastAPI in the pipeline image).
The shared lockfile keeps every service on consistent versions of common
libraries (pydantic, google-cloud-storage, etc.).

# 🗺️ Navigating the repo

```{bash}
ceo-chatbot/
├── .venv/                       # uv virtual environment (created on first `uv sync`)
├── conf/
│   ├── base/                    # shared config tracked in git
│   │   ├── prompts.yml          # chatbot prompt templates
│   │   └── rag_config.yml       # embedding model, chunk size, GCS paths, GitHub repo
│   └── local/                   # personal/local configs (gitignored)
│
├── data/                        # working copies of docs and FAISS index (not tracked)
│
├── packages/
│   └── ceo_chatbot_core/        # shared library: GCSStorage, yaml utils
│       ├── pyproject.toml
│       └── src/ceo_chatbot_core/
│
├── services/
│   ├── ceo_ingest_docs/         # extract service: GitHub repo → GCS
│   │   ├── pyproject.toml
│   │   └── src/ceo_ingest_docs/
│   ├── ceo_build_index/         # pipeline service: docs → FAISS index → GCS
│   │   ├── pyproject.toml
│   │   └── src/ceo_build_index/
│   └── ceo_rag_chatbot/         # Chatbot and FastAPI service (TODO)
│       ├── pyproject.toml
│       └── src/ceo_rag_chatbot/
|
├── app/                         # chatbot FastAPI app (root project, for now)
├── src/
│   └── ceo_chatbot/             # chatbot RAG glue code (root project, for now)
│
├── demo/
│   └── chainlit/
│       └── chainlit-demo/       # Chainlit browser UI (talks to the FastAPI chatbot)
├── notebooks/                   # development notebooks
├── scripts/
│   └── test_rag.py              # local RAG smoke test
├── tests/                       # unit/integration tests
│
├── pyproject.toml               # workspace root + chatbot project
├── uv.lock                      # single lockfile for the whole workspace
├── Dockerfile.chatbot           # currently the only Dockerfile
└── README.md
```


# 🚀 Getting Started

## What you'll need

- A computer with Python 3.12 or newer
- A Google Cloud account with access to the project (ask your team lead)
- A Gemini API key (free at https://aistudio.google.com/apikey)
- A HuggingFace access token (the embedding model is gated - see step 5)
- About 15 minutes for first-time setup

Docker is optional - only needed if you want to test the deployed-image
version of the chatbot service on your laptop before pushing to the cloud.

## First-time setup

### 1. Install the Google Cloud command-line tool

This is for authentication for the GCS buckets.

Follow the instructions for your OS here:
https://cloud.google.com/sdk/docs/install

To check it installed correctly:

`gcloud --version`

You should see output starting with "Google Cloud SDK".

### 2. Authenticate with Google Cloud and set your Project ID

Run this in your terminal:

`gcloud auth application-default login`

Sign in with your Google account which has access to the project in the browser window that opens.

> **Important:** the words `application-default` matter. Plain
> `gcloud auth login` only signs the command-line tool in. The
> `application-default` version writes a credentials file that Python
> programs can find. If you skip this you'll get a confusing
> "DefaultCredentialsError" later.

Then tell gcloud which project to use:

`gcloud config set project YOUR-PROJECT-ID`

### 3. Get a Gemini API key

Follow your SOP or go to https://aistudio.google.com/apikey. Save this value. You'll need it for inference over the documents.

### 4. Install `uv`

macOS/Linux
```{bash}
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation docs for Windows installation instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

This project uses uv for dependency management. uv is workspace-aware - you
run a single `uv sync` and it installs every workspace member's dependencies
into one shared `.venv/`. [Read more about uv in the docs.](https://docs.astral.sh/uv/getting-started/)

Also see [CONTRIBUTING.md](CONTRIBUTING.md) for more detail on developing with `uv` in this project.

### 5. Get a HuggingFace token (for the embedding model) and authenticate with HuggingFace

The default embedding model - [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) - is gated, meaning you have to accept its license before you can download it.

1. Visit the model page above and click **Acknowledge license**. Access is granted instantly.
2. Generate an access token: Profile → Settings → Access Tokens → **+ Create new token** → Token type **Read** → **Create token**. Copy the value.
3. Save the token in your `.env` (see step 7), under `HF_TOKEN=…`.

```{bash}
uv run hf auth login
```

### 6. Install the project

#### 6.1. Clone the repo

SSH:
```{bash}
git clone git@github.com:sig-gis/ceo-chatbot.git
cd ceo-chatbot
```

HTTPS:
```{bash}
git clone https://github.com/sig-gis/ceo-chatbot.git
cd ceo-chatbot
```

#### 6.2. Install all workspace members

Run this from the project root:

```{bash}
uv sync --all-packages
```

This creates `.venv/` and installs every workspace member's dependencies in
one go. The first run takes a couple of minutes (it has to download PyTorch,
sentence-transformers, etc.). Subsequent runs are quick.

You should see something like:

```
Resolved N packages in …
Installed M packages in …
 + ceo-build-index==0.1.0 (from file:///.../services/ceo_build_index)
 + ceo-chatbot-core==0.1.0 (from file:///.../packages/ceo_chatbot_core)
 + ceo-ingest-docs==0.1.0 (from file:///.../services/ceo_ingest_docs)
 + …
```

The three lines starting with `ceo-…` confirm each workspace member was
installed in editable mode.

Alternatively, you can only install dependencies for the specific service you want:

```{bash}
uv sync --package ceo_ingest_docs # Installs deps for ingesting and uploading documents
uv sync --package ceo_build_index # Similar for building the index
<!-- uv sync --package ceo_rag_chatbot -->
```

### 7. Configure your environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

Open `.env` in any text editor and set each variable. Each line has a comment
explaining what it's for. The values you'll need:

- `GOOGLE_APPLICATION_CREDENTIALS` - path to your ADC file (Linux:
  `~/.config/gcloud/application_default_credentials.json`). Step 2 created it.
- `GOOGLE_CLOUD_PROJECT` - the project you set in step 2.
- `DOCS_BUCKET` and `PREFIX` - where the source docs live in GCS.
- `DB_BUCKET` - where the FAISS index gets uploaded.
- `GEMINI_API_KEY` - from step 3.
- `HF_TOKEN` - from step 5.

> **Never commit `.env`** and never copy it into Docker images.

## Running on your laptop

The two pipeline jobs are exposed as commands by the workspace:

| Command | What it does | Implemented in |
|---|---|---|
| `uv run ingest-docs` | Clones the CEO docs repo and syncs it to GCS | `services/ceo_ingest_docs` |
| `uv run build-index` | Builds the FAISS index from docs in GCS | `services/ceo_build_index` |

Run them in order - extract first, build-index second.

> **Chat UI:** once you have the chatbot FastAPI container running (see
> [Run the chatbot container](#run-the-chatbot-container) below), you can
> also launch the Chainlit UI to talk to it from a browser.

### 1. Sync the source docs with GCS

```bash
uv run ingest-docs
```

This clones the CEO documentation repository (configured under `github:` in
`conf/base/rag_config.yml`) into a temp directory and uploads any new or
changed files to the GCS docs bucket.

If it works, you will see a summary line like this at the end (42 is an
arbitrary number):

```
{'uploaded': 42, 'skipped': 0, 'total': 42}
```

`uploaded` is files sent to GCS, `skipped` is files already up to date,
`total` is the sum.  `total` is the sum. The CEO docs are pulled from openforis' CEO documentation on github, which is specified in `conf/base/rag_config.yml`

### 2. Build the search index

```bash
uv run build-index
```

This converts the source docs into a FAISS vector index that the chatbot
searches at query time. It works as follows:

1. **Docs**: checks `data/ceo-docs/` (configurable via `docs_path` in
   `conf/base/rag_config.yml`) for existing RST files. If found, uses them.
   If not, downloads them from GCS first.
2. **Index**: loads the docs, splits them into chunks, embeds each chunk
   using the HuggingFace model, and builds a FAISS index. Saves the result to
   `data/vectorstores/ceo_docs_faiss/` (configurable via `vectorstore_path`)
   and uploads it to GCS.

Run this any time you want to rebuild the index - for example, after step 1
has synced new docs.

If the source docs have changed in GCS and you want to pull the new copy,
delete your local copy first so step 1 of the script will re-download:

```bash
rm -rf data/ceo-docs/
```

Replace `data/ceo-docs/` with your configured `docs_path` if you changed it.

<!-- #### 3.1. Install dependencies

The pipeline step requires additional packages (PyTorch, LangChain, sentence-transformers, etc.):

```bash
uv sync --extra pipeline
```

If you have an NVIDIA GPU and want GPU-accelerated FAISS, use `pipeline-gpu` instead:

```bash
uv sync --extra pipeline-gpu
```

> **Note on GPU PyTorch:** `pipeline-gpu` installs GPU-accelerated FAISS, but getting a CUDA-enabled PyTorch requires one extra step after `uv sync`. The command depends on your CUDA driver version - check with `nvidia-smi` (the "CUDA Version" field is the maximum your driver supports). See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the selector.
>
> **CUDA 13.0 or newer** - PyTorch ships CUDA-enabled wheels directly to PyPI, so no custom index is needed:
> ```bash
> uv pip install torch
> ```
>
> **CUDA 12.8 or older** - wheels come from PyTorch's own index. Use the `cu` suffix matching your CUDA version (e.g. `cu128` for 12.8, `cu126` for 12.6, `cu118` for 11.8):
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
>
> Without this step, `torch.cuda.is_available()` returns `False` even with a GPU present, and `--device auto` selects CPU.

#### 3.2. Run

To build the pipeline, it will run on your CPU by default, but it accepts args for other device types (See below). Run this command to build the index and upload it to GCS. The code block below **includes all commands required so far**

```bash
uv sync --extra pipeline # or pipeline-gpu instead
# uv pip install torch --index-url https://download.pytorch.org/whl/cu128
# uv pip install torch # For CUDA >= 13
uv run --extra pipeline python scripts/build_index.py #--device mps|cpu|cuda
```

If there are local data stores, it will not be resynced. Remove the existing corpus before running the above command. -->

#### Device options

The build-index command takes a `--device` flag that controls which hardware
runs the embedding model (the slow part):

| Flag | When to use |
|------|-------------|
| `--device auto` | Default. Picks `cuda` if available, then `mps`, then `cpu`. |
| `--device cpu` | Force CPU. Safe on any machine. Slowest. |
| `--device cuda` | Force NVIDIA GPU. Requires CUDA-enabled torch - see the note below. |
| `--device mps` | Force Apple Silicon GPU (M1/M2/M3 Mac). Requires macOS 12.3+. |

Example forcing CPU even on a GPU machine:

```bash
uv run build-index --device cpu
```

> **Note on GPU:** `services/ceo_build_index` pins PyTorch to the CPU wheel
> on PyPI to keep installs small. If you want CUDA acceleration locally, you
> can override the source pin by editing `services/ceo_build_index/pyproject.toml`
> or installing a CUDA wheel into the venv manually:
> ```bash
> uv pip install torch --index-url https://download.pytorch.org/whl/cu128
> ```
> (Replace `cu128` with the CUDA version reported by `nvidia-smi` - see
> [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).)
>
> For the chatbot service which only needs to run inference on the embedded
> query and offloads generation to Gemini, CPU is the right default. For
> indexing the ~300 RST docs, CPU runs in 10–30 minutes, GPU in under 5.

#### What to expect

The script logs each phase. On a first run (no local docs) you may see
warnings about 'reference not found' (upstream doc cross-refs), translation
warnings for missing locale files, and 'no avx2' if FAISS isn't using
AVX2-optimized binaries:

```
2026-04-30 12:00:01 - INFO - Using device: cpu
2026-04-30 12:00:02 - INFO - No local docs at data/ceo-docs, downloading from GCS...
2026-04-30 12:00:15 - INFO - Downloaded 312 files from gs://my-docs-bucket/collect-earth-online-doc/docs/source
2026-04-30 12:04:30 - INFO - Index saved to data/vectorstores/ceo_docs_faiss
2026-04-30 12:04:31 - INFO - Index uploaded to gs://my-db-bucket/ceo-docs-faiss/
gs://my-db-bucket/ceo-docs-faiss/
```

On subsequent runs the docs download is skipped:

```
2026-04-30 12:00:01 - INFO - Using device: cpu
2026-04-30 12:00:01 - INFO - Local docs found at data/ceo-docs, skipping GCS download
…
```

The last line printed is the GCS path of the uploaded index. The index is
also kept on disk at `data/vectorstores/ceo_docs_faiss/` (not tracked by
git).

## Running with Docker

Each service has its own Dockerfile under `services/<name>/Dockerfile`.
They are always built from the repo root so the workspace `uv.lock` is in the build context - every image installs only its own workspace member's dependency closure via `uv sync --package <name>`.

### Build the extract image

The extract service (`ceo-ingest-docs`) clones the upstream CEO docs repository and syncs it to GCS. From the project root:

```bash
docker build -f services/ceo_ingest_docs/Dockerfile -t ceo-ingest-docs:latest .
```

The first build takes a minute or two while uv resolves the workspace and installs the few dependencies (`pydantic`, `google-cloud-storage`, `pyyaml`, the shared `ceo_chatbot_core`) along with `git` - needed at runtime because the loader shells out to `git clone`. Subsequent builds reuse cached layers.

When it finishes you should see something like:

```
=> exporting to image
=> => writing image sha256:...
=> => naming to docker.io/library/ceo-ingest-docs:latest
```

Check the final image size:

```bash
docker image ls ceo-ingest-docs:latest
```

### Run the extract container

The container needs the same `.env` you use for local development - specifically `GOOGLE_CLOUD_PROJECT`, `DOCS_BUCKET`, and `PREFIX` - plus access to your Google Cloud credentials so it can write to the bucket.

The easiest way to supply credentials locally is the same pattern used for the chatbot below: bind-mount your ADC file (created by `gcloud auth application-default login`) and point the container at it with `GOOGLE_APPLICATION_CREDENTIALS`.

**Linux / macOS:**

```bash
docker run --rm \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  ceo-ingest-docs:latest
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  --env-file .env `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/gcp/adc.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json `
  ceo-ingest-docs:latest
```

The container clones the upstream docs repo into a temp directory and uploads any new or changed files to your `DOCS_BUCKET`. When it finishes you'll see the same summary line as `uv run ingest-docs`:

```
{'uploaded': 42, 'skipped': 0, 'total': 42}
```

`uploaded` is files sent to GCS, `skipped` is files already up to date,
`total` is their sum.

> **Note on Cloud Run Jobs:** when deploying this image to Cloud Run
> Jobs, you do not use `docker run`. Environment variables are set in
> the job configuration, and the attached service account supplies
> credentials automatically (no ADC bind-mount needed).

### Build the pipeline image

The pipeline service (`ceo-build-index`) reads the source docs from GCS (or a local copy if one exists), splits them into chunks, embeds each chunk with a HuggingFace model, builds a FAISS index, and uploads the index back to GCS. The chatbot service reads from that same path on startup. From the project root:

```bash
docker build -f services/ceo_build_index/Dockerfile -t ceo-build-index:latest .
```

The first build takes several minutes - this image carries the heavyweight machine-learning stack: PyTorch (CPU-only, ~200 MB), sentence-transformers, transformers, faiss-cpu, langchain-*, and `unstructured` (for RST parsing). The pyproject pins torch to the CPU wheel index, so this image never pulls CUDA wheels and stays under the 4 GB target. Subsequent builds reuse cached layers - only the changed layers rebuild.

When it finishes you should see something like:

```
=> exporting to image
=> => writing image sha256:...
=> => naming to docker.io/library/ceo-build-index:latest
```

Check the final image size:

```bash
docker image ls ceo-build-index:latest
```

### Run the pipeline container

The container needs the same `.env` you use for local development - specifically `GOOGLE_CLOUD_PROJECT`, `DOCS_BUCKET`, `DB_BUCKET`, `PREFIX`, and `HF_TOKEN` (the default embedding model is gated; see step 5 of first-time setup if you haven't generated a token yet). It also needs your Google Cloud credentials so it can read from `DOCS_BUCKET` and write to `DB_BUCKET`.

The easiest way to supply credentials locally is the same pattern used for the extract image above: bind-mount your ADC file (created by `gcloud auth application-default login`) and point the container at it with `GOOGLE_APPLICATION_CREDENTIALS`. If you have already downloaded the document store locally,

**Linux / macOS:**

```bash
docker run --rm \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  ceo-build-index:latest
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  --env-file .env `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/gcp/adc.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json `
  ceo-build-index:latest
```

The default entry point is `build-index` with `--device auto`, which falls through to CPU inside this image because the container has no GPU runtime. On CPU the embed step takes about 5-30 minutes for the ~300 RST files in the upstream CEO docs.

You will see log lines like:

```
2026-04-30 12:00:01 - INFO - Using device: cpu
2026-04-30 12:00:01 - INFO - Local docs found at data/ceo-docs, ...
2026-04-30 12:04:30 - INFO - Index saved to data/vectorstores/ceo_docs_faiss
2026-04-30 12:04:31 - INFO - Index uploaded to gs://my-db-bucket/ceo-docs-faiss/
gs://my-db-bucket/ceo-docs-faiss/
```

The last line printed is the GCS path of the uploaded index - this is exactly what `uv run build-index` prints, and the chatbot service reads from this same path on startup.

By default the container is stateless: it has no source docs baked in (the repo's `data/` directory is excluded from the build context via `.dockerignore`), so every run downloads the docs from `DOCS_BUCKET` before chunking. If you are iterating locally and want repeated runs to reuse the same working copy, bind-mount your host `data/` onto `/app/data/`. The pipeline checks `data/ceo-docs/` first and only falls back to GCS when it finds no RST files there:

```bash
docker run --rm \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$PWD/data:/app/data" \
  ceo-build-index:latest
```

> **Note on Cloud Run Jobs:** as with the extract image, environment variables are set on the job configuration and the attached service account supplies credentials automatically. `HF_TOKEN` should live in Secret Manager and be referenced from there rather than passed as a plain env var.

If you have downloaded the model locally already, you can bind-mount it to the docker container as well:

```bash
docker run --rm \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$PWD/data:/app/data" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ceo-build-index:latest
```

### Build the chatbot image

The chatbot service (`ceo-rag-chatbot`) downloads the FAISS index from GCS on startup, loads the embedding model from HuggingFace, and serves a FastAPI app at `/chat`. From the project root:

```bash
docker build -f services/ceo_rag_chatbot/Dockerfile -t ceo-rag-chatbot:latest .
```

The first build takes several minutes - it installs the same heavyweight machine-learning stack as the pipeline image (PyTorch CPU, sentence-transformers, transformers, faiss-cpu, langchain-*) plus FastAPI/uvicorn and google-genai. The pyproject pins torch to the CPU wheel index, so this image never pulls CUDA wheels. Subsequent builds reuse cached layers - only the changed layers rebuild.

When it finishes you should see something like:

```
=> exporting to image
=> => writing image sha256:...
=> => naming to docker.io/library/ceo-rag-chatbot:latest
```

Check the final image size:

```bash
docker image ls ceo-rag-chatbot:latest
```


### Run the chatbot container

The container reads its configuration from your `.env` file - the same one you use for local development. Make sure it contains these keys:

```
GEMINI_API_KEY=...
GOOGLE_CLOUD_PROJECT=...
DB_BUCKET=...
HF_TOKEN=...
```

It also needs your Google Cloud credentials so it can download the FAISS index from GCS. An earlier step - `gcloud auth application-default login` - creates an ADC file at `~/.config/gcloud/application_default_credentials.json`. Bind-mount this file to `/gcp/adc.json` and point the container at it with `GOOGLE_APPLICATION_CREDENTIALS`.

**Linux / macOS:**

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  ceo-rag-chatbot:latest
```
> **Note on run options**: If you ran the index building pipeline, the document store and embedding model will already exist locally. In that case, there are instructions below to use the existing versions instead of downloading the index and model once more.

**Windows (PowerShell):**

```powershell
docker run --rm `
  -p 8080:8080 `
  --env-file .env `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/gcp/adc.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json `
  ceo-rag-chatbot:latest
```

`--env-file` injects every variable from `.env` without exposing secrets in your shell history or in `docker inspect` output. Note that `.env` itself is **not** copied into the image - it's read at runtime from your host.

`-p 8080:8080` publishes the container's port 8080 to your host. Without it, uvicorn listens inside the container but your browser cannot reach it.

The container downloads the FAISS index from GCS (a few seconds) and the embedding model from HuggingFace (the first run fetches roughly 600 MB - 1.2 GB of weights for `google/embeddinggemma-300m`; tqdm progress is silent in `docker logs`). Once ready, you will see:

```
INFO:ceo_rag_chatbot.lifespan:chatbot ready in 6.1s
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

> **Note on Cloud Deployment:** when deploying to the cloud you do not use `docker run`. Environment variables are set in the service configuration, and secrets (`GEMINI_API_KEY`, `HF_TOKEN`) should be stored in GCP Secret Manager/following your SOP and referenced from there rather than passed as plain env vars.

#### Faster local iteration: bind-mounted caches

By default each container is stateless - every run re-downloads the FAISS index from GCS and the embedding model from HuggingFace. For repeated local runs you can bind-mount the host paths the container would otherwise rebuild:

- `$PWD/data:/app/data` - the chatbot writes the FAISS index to `data/vectorstores/ceo_docs_faiss/` inside the container. Reusing the host's `data/` skips the GCS download on subsequent starts.
- `$HOME/.cache/huggingface:/root/.cache/huggingface` - the embedding model is cached here. Reusing the host's HuggingFace cache skips the ~600 MB - 1.2 GB model download. If you have already run `uv run build-index` on your host, the model is already there.

**Reuse a local FAISS index only (skip the GCS index download):**

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$PWD/data:/app/data" \
  ceo-rag-chatbot:latest
```

**Reuse the HuggingFace model cache only (skip the embedding-model download):**

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ceo-rag-chatbot:latest
```

**Reuse both (fastest cold start):**

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  -v "$PWD/data:/app/data" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ceo-rag-chatbot:latest
```

These are purely local-dev optimizations - Cloud Run runs fresh containers and always re-downloads from GCS and HuggingFace.

### Check it is running

```bash
curl http://localhost:8080/healthz
```

You should see:

```json
{"status": "ready"}
```

If you see `{"status": "loading"}` with HTTP 503, the index is still
downloading - wait a moment and try again.

You can also visit http://localhost:8080/docs to see the current API and
schemas.

### Sample query

Ask the RAG system "What is CEO?". Below is a curl request with appropriate
headers hitting the port exposed locally in the `docker run` command.

```{bash}
curl -X 'POST' \
  'http://localhost:8080/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is CEO?",
  "history": []
}'
```

Sample response, which returns the answer as well as all cited sources:

```{bash}
{
  "answer": "[CEO](https://collect-earth-online-doc.readthedocs.io/en/latest/index.html) enables users to efficiently collect up-to-date information about their environment and observe changes over time. Users can learn how to [collect data in CEO](https://collect-earth-online-doc.readthedocs.io/en/latest/index.html), [manage an institution](https://collect-earth-online-doc.readthedocs.io/en/latest/index.html), or [create a project](https://collect-earth-online-doc.readthedocs.io/en/latest/index.html). To get started, users can [register for CEO](https://collect-earth-online-doc.readthedocs.io/en/latest/index.html) and set up their Collect Earth Online account.",
  "sources": [
    "https://collect-earth-online-doc.readthedocs.io/en/latest/index.html",
    "https://collect-earth-online-doc.readthedocs.io/en/latest/index.html",
    "https://collect-earth-online-doc.readthedocs.io/en/latest/setup/register.html",
    "https://collect-earth-online-doc.readthedocs.io/en/latest/index.html",
    "https://collect-earth-online-doc.readthedocs.io/en/latest/institution/imagery.html"
  ]
}
```

### Running the Chainlit UI

Once the chatbot container is up and responding at `http://localhost:8080`,
you can launch a browser-based chat interface:

```bash
uv run chainlit run demo/chainlit/chainlit-demo/src/chainlit_demo/demo.py -w
```

The `-w` flag enables auto-reload so Chainlit restarts automatically if you
edit the file.

When Chainlit starts you should see:

```
Your app is available at http://localhost:8000
```

Open that URL in your browser. On first load the UI checks the chatbot's
`/healthz` endpoint to confirm it is ready. If the chatbot is still starting
up, the UI retries every 15 seconds for up to 3 minutes before giving up.

> **Adjusting the timeout:** the per-request timeout (how long the UI waits
> for each `/chat` response) is the `TIMEOUT` constant at the top of
> `demo/chainlit/chainlit-demo/src/chainlit_demo/demo.py`. The default is
> 15 seconds. Raise it if the chatbot is slow to respond on your machine.

### Common errors

**`DefaultCredentialsError` in container logs**

The container cannot find Google Cloud credentials. Make sure you included
the `-v` and `-e GOOGLE_APPLICATION_CREDENTIALS` flags shown above. On Linux
the ADC file is at `~/.config/gcloud/application_default_credentials.json`;
on macOS it may be at the same path or under `~/Library/Application Support`.
Run `gcloud auth application-default login` on your laptop first if the file
does not exist.

**`403 Forbidden` when downloading the index**

Your credentials are valid but the service account or user does not have
read access to the GCS bucket. Ask your team lead to grant the
**Storage Object Viewer** role on the bucket.

**`/healthz` returns 503 indefinitely**

The lifespan failed to finish. Check the container logs with
`docker logs <container-id>` for the underlying error (usually a credentials
or bucket-name problem).

**`google-api-core ConnectionError`**

Your machine can't reach Google Cloud. Check your internet connection, VPN,
or firewall.

## Tests

Run the unit tests with:

```bash
uv run pytest tests/
```

You should see all tests pass with no errors.

The unit tests check that the sync logic - the rules that decide whether a
file gets uploaded to Google Cloud Storage or skipped - behaves correctly for
every case: a file that doesn't exist yet in GCS, a file that was touched
locally but didn't actually change, a file with real changes, and a file
where the remote copy is newer. These tests run without any GCS credentials
or internet connection by substituting a fake GCS client, so they are safe to
run anywhere and fast to run often.
