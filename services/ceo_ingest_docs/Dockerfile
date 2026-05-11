# syntax=docker/dockerfile:1
#
# services/ceo_ingest_docs/Dockerfile — extract service (Cloud Run Job)
#
# Build from the repo root so the workspace uv.lock is in the build context:
#   docker build -f services/ceo_ingest_docs/Dockerfile -t ceo-ingest-docs:latest .
#
# Required env at runtime:
#   GCP_PROJECT_ID, DOCS_BUCKET, PREFIX
# Plus Google Cloud credentials:
#   - local: bind-mount your ADC file and set GOOGLE_APPLICATION_CREDENTIALS
#   - Cloud Run Job: credentials come from the attached service account

ARG UV_VERSION=0.11.6
ARG PYTHON_VERSION=3.12

# ── Stage 1: builder ─────────────────────────────────────────────────────────

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# UV_COMPILE_BYTECODE: compile .py to .pyc at install time so cold starts
#   skip recompilation.
# UV_LINK_MODE=copy: the cache mount lives on a different filesystem from
#   the install target; hard links don't cross filesystem boundaries.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Layer 1 — install only the external dep closure for ceo-ingest-docs.
#
# Bind-mounting pyproject files (instead of COPYing them) keeps them out
# of this layer's hash, so unrelated edits don't invalidate the slow
# dep-install step. The sibling member pyproject.tomls are bind-mounted
# because uv reads every workspace member's pyproject.toml during
# workspace discovery, even when --package narrows the install set.
#
# --no-install-workspace skips building any workspace member (including
# ceo-chatbot-core, which ceo-ingest-docs depends on). Workspace members
# need their src/ trees on disk to build; deferring them to Layer 2 lets
# this layer cache against only the pyproject + lockfile.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/ceo_chatbot_core/pyproject.toml,target=packages/ceo_chatbot_core/pyproject.toml \
    --mount=type=bind,source=services/ceo_ingest_docs/pyproject.toml,target=services/ceo_ingest_docs/pyproject.toml \
    --mount=type=bind,source=services/ceo_build_index/pyproject.toml,target=services/ceo_build_index/pyproject.toml \
    --mount=type=bind,source=services/ceo_rag_chatbot/pyproject.toml,target=services/ceo_rag_chatbot/pyproject.toml \
    uv sync --package ceo-ingest-docs --frozen --no-dev --no-install-workspace --no-editable

# Layer 2 — copy the source for ceo-ingest-docs and its workspace dep
# (ceo_chatbot_core), then install the project itself into the venv.
# Sibling members' pyproject.tomls are still required on disk for the
# workspace to resolve, but their source trees are not.
COPY pyproject.toml uv.lock ./
COPY packages/ceo_chatbot_core/ packages/ceo_chatbot_core/
COPY services/ceo_ingest_docs/ services/ceo_ingest_docs/
COPY services/ceo_build_index/pyproject.toml services/ceo_build_index/pyproject.toml
COPY services/ceo_rag_chatbot/pyproject.toml services/ceo_rag_chatbot/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package ceo-ingest-docs --frozen --no-dev --no-editable

# ── Stage 2: runtime ─────────────────────────────────────────────────────────

FROM python:${PYTHON_VERSION}-slim AS runtime

# git is required at runtime: ceo_ingest_docs.github_loader.GitHubLoader.clone()
# shells out to `git clone --depth 1 ...`.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Self-contained venv from the builder. --no-editable above means the
# installed packages live inside site-packages, so we do NOT need to
# COPY src/ into this stage.
COPY --from=builder /app/.venv /app/.venv

# rag_config.yml is loaded via the relative path "conf/base/rag_config.yml"
# (see services/ceo_ingest_docs/src/ceo_ingest_docs/main.py). WORKDIR=/app
# means it resolves to /app/conf/base/rag_config.yml.
COPY conf/ conf/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

CMD ["ingest-docs"]
