# syntax=docker/dockerfile:1
#
# services/ceo_build_index_gpu/Dockerfile -- GPU pipeline service (Cloud Run Job)
#
# Build from the repo root so the workspace uv.lock is in the build context:
#   docker build -f services/ceo_build_index_gpu/Dockerfile -t ceo-build-index-gpu:latest .
#
# Required env at runtime:
#   GOOGLE_CLOUD_PROJECT, DOCS_BUCKET, DB_BUCKET, PREFIX
#   HF_TOKEN  (read directly by huggingface_hub for the gated embedding model)
# Plus Google Cloud credentials:
#   - local: bind-mount your ADC file and set GOOGLE_APPLICATION_CREDENTIALS
#   - Cloud Run Job: credentials come from the attached service account
#
# torch CUDA build: the whole repo is one uv workspace with a single uv.lock,
# and ceo-build-index pins torch to the CPU wheel index. uv resolves one torch
# build for the entire workspace, so `uv sync` below installs +cpu torch even
# here. A later step reinstalls the CUDA build from an explicit PyTorch index.
# To target a different CUDA version, override TORCH_CUDA_INDEX at build time:
#   docker build --build-arg TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 ...
#
# Run with GPU access locally (native Docker Engine + NVIDIA Container Toolkit;
# Docker Desktop on Linux cannot access the host GPU):
#   docker run --rm --device nvidia.com/gpu=all --env-file .env ... ceo-build-index-gpu:latest

ARG UV_VERSION=0.11.6
ARG PYTHON_VERSION=3.12

# -- Stage 1: builder ---------------------------------------------------------

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

# Layer 1 -- install only the external dep closure for ceo-build-index-gpu.
#
# Bind-mounting pyproject files keeps them out of this layer's hash, so
# unrelated edits don't invalidate the slow dep-install step. All sibling
# member pyproject.tomls are bind-mounted because uv reads every workspace
# member's pyproject.toml during workspace discovery, even when --package
# narrows the install set.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/ceo_chatbot_core/pyproject.toml,target=packages/ceo_chatbot_core/pyproject.toml \
    --mount=type=bind,source=services/ceo_ingest_docs/pyproject.toml,target=services/ceo_ingest_docs/pyproject.toml \
    --mount=type=bind,source=services/ceo_build_index/pyproject.toml,target=services/ceo_build_index/pyproject.toml \
    --mount=type=bind,source=services/ceo_build_index_gpu/pyproject.toml,target=services/ceo_build_index_gpu/pyproject.toml \
    --mount=type=bind,source=services/ceo_rag_chatbot/pyproject.toml,target=services/ceo_rag_chatbot/pyproject.toml \
    uv sync --package ceo-build-index-gpu --frozen --no-dev --no-install-workspace --no-editable

# Layer 2 -- copy source for ceo-build-index-gpu and its workspace dep
# (ceo_chatbot_core), then install the project itself into the venv.
COPY pyproject.toml uv.lock ./
COPY packages/ceo_chatbot_core/ packages/ceo_chatbot_core/
COPY services/ceo_build_index_gpu/ services/ceo_build_index_gpu/
COPY services/ceo_build_index/pyproject.toml services/ceo_build_index/pyproject.toml
COPY services/ceo_ingest_docs/pyproject.toml services/ceo_ingest_docs/pyproject.toml
COPY services/ceo_rag_chatbot/pyproject.toml services/ceo_rag_chatbot/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package ceo-build-index-gpu --frozen --no-dev --no-editable

# Swap the CPU torch (installed from the lock above) for the CUDA build. 
# --reinstall-package torch replaces only torch, pulling its
# CUDA runtime deps (nvidia-*) fresh from the PyTorch index.
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv/bin/python --reinstall-package torch \
    torch==2.11.0 --index-url ${TORCH_CUDA_INDEX}

# -- Stage 2: runtime ---------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim AS runtime

# libmagic1 is required at runtime: ceo_build_index_gpu imports
# langchain_community.document_loaders.rst, which in turn pulls in
# `unstructured`. `unstructured` calls libmagic for content-type sniffing
# at import time, so the package fails to load without it even though
# the RST parser itself is pure Python.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libmagic1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY conf/ conf/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# --device auto detects CUDA at runtime. Force cuda explicitly here because
# this image exists specifically to run on GPU -- if CUDA is not available
# something is wrong with the container runtime, and we want a loud failure
# rather than a silent CPU fallback.
CMD ["build-index-gpu", "--device", "cuda"]
