# syntax=docker/dockerfile:1
#
# services/ceo_rag_chatbot/Dockerfile — chatbot service (Cloud Run Service)
#
# Build from the repo root so the workspace uv.lock is in the build context:
#   docker build -f services/ceo_rag_chatbot/Dockerfile -t ceo-rag-chatbot:latest .
#
# Required env at runtime:
#   GEMINI_API_KEY, GOOGLE_CLOUD_PROJECT, DB_BUCKET
#   HF_TOKEN  (read directly by huggingface_hub for the gated embedding model)
# Optional:
#   PORT       (Cloud Run injects this; the `chatbot` script defaults to 8080)
#   LOG_LEVEL  (default INFO)
# Plus Google Cloud credentials:
#   - local: bind-mount your ADC file and set GOOGLE_APPLICATION_CREDENTIALS
#   - Cloud Run: credentials come from the attached service account
#
# The pyproject pins torch to the CPU-only wheel via the pytorch-cpu index,
# so this image never pulls CUDA wheels.

ARG UV_VERSION=0.11.6
ARG PYTHON_VERSION=3.12

# ── Stage 1: builder ─────────────────────────────────────────────────────────

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# UV_COMPILE_BYTECODE: compile .py to .pyc at install time so cold starts
#   skip recompilation (important for Cloud Run cold-start latency).
# UV_LINK_MODE=copy: the cache mount lives on a different filesystem from
#   the install target; hard links don't cross filesystem boundaries.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Layer 1 — install only the external dep closure for ceo-rag-chatbot.
#
# Bind-mounting pyproject files (instead of COPYing them) keeps them out
# of this layer's hash, so unrelated edits don't invalidate the slow
# dep-install step (torch + sentence-transformers + fastapi[standard] are
# large). Sibling member pyproject.tomls are bind-mounted because uv reads
# every workspace member's pyproject.toml during workspace discovery, even
# when --package narrows the install set.
#
# --no-install-workspace skips building any workspace member (including
# ceo-chatbot-core, which ceo-rag-chatbot depends on). Workspace members
# need their src/ trees on disk to build; deferring them to Layer 2 lets
# this layer cache against only the pyproject + lockfile.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/ceo_chatbot_core/pyproject.toml,target=packages/ceo_chatbot_core/pyproject.toml \
    --mount=type=bind,source=services/ceo_ingest_docs/pyproject.toml,target=services/ceo_ingest_docs/pyproject.toml \
    --mount=type=bind,source=services/ceo_build_index/pyproject.toml,target=services/ceo_build_index/pyproject.toml \
    --mount=type=bind,source=services/ceo_rag_chatbot/pyproject.toml,target=services/ceo_rag_chatbot/pyproject.toml \
    uv sync --package ceo-rag-chatbot --frozen --no-dev --no-install-workspace --no-editable

# Layer 2 — copy the source for ceo-rag-chatbot and its workspace dep
# (ceo_chatbot_core), then install the project itself into the venv.
# Sibling members' pyproject.tomls are still required on disk for the
# workspace to resolve, but their source trees are not.
COPY pyproject.toml uv.lock ./
COPY packages/ceo_chatbot_core/ packages/ceo_chatbot_core/
COPY services/ceo_rag_chatbot/ services/ceo_rag_chatbot/
COPY services/ceo_ingest_docs/pyproject.toml services/ceo_ingest_docs/pyproject.toml
COPY services/ceo_build_index/pyproject.toml services/ceo_build_index/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package ceo-rag-chatbot --frozen --no-dev --no-editable

# ── Stage 2: runtime ─────────────────────────────────────────────────────────

FROM python:${PYTHON_VERSION}-slim AS runtime

WORKDIR /app

# Self-contained venv from the builder. --no-editable above means the
# installed packages live inside site-packages, so we do NOT need to
# COPY src/ into this stage.
COPY --from=builder /app/.venv /app/.venv

# rag_config.yml and prompts.yml are loaded via relative paths
# ("conf/base/rag_config.yml", "conf/base/prompts.yml") from
# services/ceo_rag_chatbot/src/ceo_rag_chatbot/config.py. WORKDIR=/app
# means those resolve to /app/conf/...
COPY conf/ conf/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# /healthz returns 200 once the lifespan has finished downloading the
# FAISS index from GCS and loading the embedding model. start-period
# covers that download + the first-time HuggingFace model fetch;
# without it Docker would mark the container unhealthy during normal
# boot and may restart it.
#
# Uses Python's urllib because python:3.12-slim does not include curl.
# Cloud Run ignores this HEALTHCHECK and uses its own readiness probe;
# this is purely a local-dev signal for `docker ps`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=6 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" \
        || exit 1

# `chatbot` is the project script exposed by services/ceo_rag_chatbot/pyproject.toml
# It calls uvicorn.run(...) with host=0.0.0.0 and port=int(os.environ.get("PORT", 8080)),
# so $PORT injection from Cloud Run is honored automatically.
CMD ["chatbot"]
