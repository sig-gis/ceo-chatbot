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

# 🗺️ Navigating the repo

```{bash}
ceo-chatbot/
├── .venv/                    # uv python virtual environment directory (not tracked; created on first `uv sync` or `uv run`)
├── conf/                     # Configuration files for script arguments (YAML, JSON)
│   ├── base/                 # Global/shared configuration files tracked in repo
│   │   ├── prompts.yml       # Prompt configuration
│   │   └── rag_config.yml    # RAG configuration: models, vectorstore, prompts...
│   └── local/                # Personal/local configs (excluded from version control)
│
├── data/                     # Project data directory (not tracked in repo)
│
├── demo/                     # Try ceo-chatbot in a streamlit app
│
├── notebooks/                # Jupyter notebooks for development, prototyping, demos
│
├── src/                      # Source code (modules and utilities imported by scripts) and scripts
│   └── ceo_chatbot
│       ├── __init__.py
│       ├── config.py               # Utilities for loading confs in conf/base/
│       │
│       ├── ingest                  # Data ingestion pipeline: loaders.py, chunking.py, embeddings.py -> index_builder.py
│       │   ├── __init__.py
│       │   ├── chunking.py         # Utilities for splitting documents into chunks
│       │   ├── embeddings.py       # Utilities for defining embedding model
│       │   ├── index_builder.py    # Define data ingestion pipeline
│       │   └── loaders.py          # Utilities for loading dataset
│       │
│       └── rag                     # RAG pipeline: llm.py, retriever.py -> pipeline.py
│           ├── __init__.py
│           ├── llm.py              # Utilities for defining reader model
│           ├── pipeline.py         # Define RAG pipeline
│           └── retriever.py        # Utilities for doc retrieval and similarity search
│
├── scripts/
│   ├── build_index.py        # Data ingestion pipeline runner (offline); chunk, embed, build vector store
│   └── ...             
│
├── tests/                    # Unit and integration tests
│
├── .gitignore                
├── .python-version           # Python version used by uv environment manager
├── pyproject.toml            # Metadata about the project (for uv)
├── uv.lock                   # Locked dependency versions 
└── README.md                 

```


# 🚀 Getting Started

## What you'll need

- A computer with Python 3.10 or newer
- A Google Cloud account with access to the project (ask your team lead)
- A Gemini API key (free at https://aistudio.google.com/apikey)
- About 15 minutes for first-time setup

Docker is optional - only needed if you want to test the deployed-image
version of the system on your laptop before pushing to the cloud.

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

### 4. Install `uv`:

macOS/Linux
```{bash}
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the [uv installation docs for Windows installation instructions](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2)

This app uses uv for dependency managment. 
[Read more about uv in the docs.](https://docs.astral.sh/uv/getting-started/) 


Also see [CONTRIBUTING.md](CONTRIBUTING.md) for more detail on developing with `uv` in this project.

### 5. Install the project

#### 5.1. Clone the repo

SSH: 
```{bash}
git clone git@github.com:sig-gis/ceo-chatbot.git
```

HTTPS:
```{bash}
git clone https://github.com/sig-gis/ceo-chatbot.git
```

#### 5.2. Install base dependencies

Run this command to use the `uv` package manager to install dependencies common to all deployments

```{bash}
uv sync
```

## Running on your laptop

### 1. Configure your environment

Copy `.env.example` to `.env`:
Open `.env` in any text editor and fill in the values. Each line has a comment explaining what it's for.

### 2. Sync the source docs with GCS

This command clones the CEO documentation repository from GitHub and uploads any new or changed files to the Google Cloud Storage bucket.

```bash
uv run python scripts/extract_docs.py
```

If it works, you will see a summary line like this at the end (42 is an arbitrary number):

```
{'uploaded': 42, 'skipped': 0, 'total': 42}
```

`uploaded` is files sent to GCS, `skipped` is files already up to date, `total` is the sum. The CEO docs are pulled from openforis' CEO documentation on github, which is specified in `conf/base/rag_config.yml`

### 3. Build the search index

This command converts the source docs into a FAISS vector index that the chatbot searches at query time. It works as follows:

1. **Docs**: checks `data/ceo-docs/` (configurable via `docs_path` in `conf/base/rag_config.yml`) for existing RST files. If found, uses them. If not, downloads them from GCS first.
2. **Index**: loads the docs, splits them into chunks, embeds each chunk using a HuggingFace model, and builds a FAISS index. Saves the result to `data/vectorstores/ceo_docs_faiss/` (configurable via `vectorstore_path` in `conf/base/rag_config.yml`) and uploads it to GCS.

Run this step any time you want to rebuild the index - for example, after step 2 has synced new docs to GCS.

**Getting updated docs from GCS:** If the source docs have been updated in GCS (e.g. after running `extract_docs.py`), the script will not re-download them as long as local docs exist. To force a fresh download, delete the local docs directory before building the index again (instructions on how to do that below):

```bash
rm -rf data/ceo-docs/
```

Replace `data/ceo-docs/` with your configured `docs_path` if you changed it in `conf/base/rag_config.yml`.

#### 3.1. Install dependencies

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

If there are local data stores, it will not be resynced. Remove the existing corpus before running the above command.

#### Device options

The `--device` flag controls which hardware runs the embedding model (the slow part).

| Flag | When to use |
|------|-------------|
| `--device auto` | Default. Picks `cuda` if available, then `mps`, then `cpu`. |
| `--device cpu` | Force CPU. Safe on any machine. Slowest. |
| `--device cuda` | Force NVIDIA GPU. Requires CUDA-enabled torch (see note above). |
| `--device mps` | Force Apple Silicon GPU (M1/M2/M3 Mac). Requires macOS 12.3+. |

Example forcing CPU even on a GPU machine:

```bash
uv run --extra pipeline python scripts/build_index.py --device cpu
```

#### What to expect

The script logs each phase. On a first run (no local docs) - you may see some warnings related to 'reference not found', related to upstream documents; translation wornings for missing locale files, and 'no avx2' if FAISS is not using AVX2-optimized binaries:

```
2026-04-30 12:00:01 - INFO - Using device: cuda
2026-04-30 12:00:02 - INFO - No local docs at data/ceo-docs, downloading from GCS...
2026-04-30 12:00:15 - INFO - Downloaded 312 files from gs://my-docs-bucket/collect-earth-online-doc/docs/source
2026-04-30 12:04:30 - INFO - Index saved to data/vectorstores/ceo_docs_faiss
2026-04-30 12:04:31 - INFO - Index uploaded to gs://my-db-bucket/ceo-docs-faiss/
gs://my-db-bucket/ceo-docs-faiss/
```

On subsequent runs the docs download is skipped:

```
2026-04-30 12:00:01 - INFO - Using device: cuda
2026-04-30 12:00:01 - INFO - Local docs found at data/ceo-docs, skipping GCS download
...
```

The last line printed is the GCS path of the uploaded index. The index is also kept on disk at `data/vectorstores/ceo_docs_faiss/` (not tracked by git). Embedding on CPU takes 10–30 minutes; on a GPU usually under 5 minutes.

<!-- Instructions below are outdated -->

### 2. Install ceo-chatbot 

Install the ceo-chatbot package locally in editable mode.

From the project root: 

```{bash}
uv pip install -e .
```

### 3. (offline) Build vector DB

For local development, the knowledge corpus must be set up one time.

If using a gated model such as [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m), 

1. Request access to the model on its Hugging Face model page (access is granted instantly)
2. Generate an access token: Profile > Settings > Access Tokens > + Create new token > Token type Read > Create token
3. Run the following `hf` cli command in your terminal and paste your HF access token when prompted

```{bash}
uv run hf auth login
```

Authenticate GCloud for access to the GCS bucket in our cloud project

```{bash}
gcloud auth login
```

To initially set up the CEO docs corpus or to manually re-upload updates to the GCS bucket specified, run the CEO docs extraction pipeline.
NOTE: this step is not required to reproduce the chatbot once the corpus exists in GCS. If the corpus already exists in GCS, skip to *Build the vector DB* below. 
 

```{bash}
uv run scripts/extract_docs.py
```

Build the vector DB:

```{bash}
uv run scripts/build_index.py
```

Stay tuned! A planned future version will automate uploading the corpus to GCS and building the vector DB.

## Running with Docker

### Build the image

From the project root:

```bash
docker build -f Dockerfile.chatbot -t ceo-chatbot:latest .
```

The build takes a few minutes the first time while it downloads and installs PyTorch, sentence-transformers, and the other dependencies. Subsequent builds reuse cached layers - only the changed layers rebuild.

### Run the container

The container reads its configuration from your existing `.env` file - the same one you use for local development. Make sure it contains these keys:

```
GEMINI_API_KEY=...
DB_BUCKET=...
GCP_PROJECT_ID=...
HF_TOKEN=...
```

In addition, the container needs your Google Cloud credentials so it can download the FAISS index from GCS. An earlier step - `gcloud auth application-default login` - creates some application credentials in `~/.config/gcloud/application_default_credentials.json`. The easiest way to supply your credentials is to bind-mount this file to `/gcp/adc.json`, and then point the container at it with `GOOGLE_APPLICATION_CREDENTIALS`.

**Linux / macOS:**

```bash
docker run --rm \
  -p 8080:8080 \
  --env-file .env \
  -v "$HOME/.config/gcloud/application_default_credentials.json:/gcp/adc.json:ro" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json \
  ceo-chatbot:latest
```

**Windows (PowerShell):**

```powershell
docker run --rm `
  -p 8080:8080 `
  --env-file .env `
  -v "$env:APPDATA\gcloud\application_default_credentials.json:/gcp/adc.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/gcp/adc.json `
  ceo-chatbot:latest
```

`--env-file` injects every variable from `.env` without exposing secrets in your shell history or in `docker inspect` output.

> **Note on Cloud Run:** when deploying to Cloud Run you do not use > `docker run`. Environment variables are set in the service configuration, and secrets (`GEMINI_API_KEY`, `HF_TOKEN`) should be stored in GCP Secret Manager and referenced from there rather than passed as plain env vars.

The container downloads the FAISS index from GCS on startup (this takes a few seconds). Once it is ready, you will see a log line like:

```
INFO:app.lifespan:chatbot ready in 8.3s
```

### Check it is running

```bash
curl http://localhost:8080/healthz
```

You should see:

```json
{"status": "ready"}
```

If you see `{"status": "loading"}` with HTTP 503, the index is still downloading - wait a moment and try again.

You can also visit http://localhost:8080/docs to see the current API and schemas.

### Sample Query:
Ask the RAG system "What is CEO?". Below is a curl request with appropriate headers hitting the port exposed locally in the `docker run` command.
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


### Common errors

**`DefaultCredentialsError` in container logs**

The container cannot find Google Cloud credentials. Make sure you included the
`-v` and `-e GOOGLE_APPLICATION_CREDENTIALS` flags shown above. On Linux the
ADC file is at `~/.config/gcloud/application_default_credentials.json`; on
macOS it may be at the same path or under `~/Library/Application Support`.
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
Your machine can't reach Google Cloud. Check your internet connection, VPN or firewall.

## Tests

Run the unit tests with:

```bash
uv run pytest tests/
```

You should see all tests pass with no errors.

The unit tests check that the sync logic - the rules that decide whether a file gets uploaded to Google Cloud Storage or skipped - behaves correctly for every case: a file that doesn't exist yet in GCS, a file that was touched locally but didn't actually change, a file with real changes, and a file where the remote copy is newer. These tests run without any GCS credentials or internet connection by substituting a fake GCS client, so they are safe to run anywhere and fast to run often.
