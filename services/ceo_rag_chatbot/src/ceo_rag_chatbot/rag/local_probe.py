"""
 summary:
    Readiness check for a local Ollama inference server.

 details:
    Asks Ollama's own API how far along the model is:
      /api/version responds  -> server is up
      model in /api/tags     -> model is downloaded
      model in /api/ps       -> model is loaded in VRAM, ready to answer
"""

from enum import Enum

import httpx


class ModelState(str, Enum):
    """How ready the local model is, from least to most ready."""
    UNREACHABLE = "unreachable"
    SERVER_UP = "server_up"
    DOWNLOADED = "downloaded"
    LOADED = "loaded"


def _model_names(client: httpx.Client, url: str) -> set[str]:
    """Return the model names listed by an Ollama endpoint (/api/tags or /api/ps)."""
    response = client.get(url)
    response.raise_for_status()
    return {m["name"] for m in response.json().get("models", [])}


def probe_model(base_url: str, model: str, client: httpx.Client) -> ModelState:
    """Return how ready `model` is on the Ollama server at `base_url`.

    `base_url` is the OpenAI-compatible URL from rag_config.yml (ending in /v1);
    Ollama's own API lives at the server root, so /v1 is stripped.
    Any connection or HTTP error counts as UNREACHABLE.
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        client.get(f"{root}/api/version").raise_for_status()
        if model not in _model_names(client, f"{root}/api/tags"):
            return ModelState.SERVER_UP
        if model not in _model_names(client, f"{root}/api/ps"):
            return ModelState.DOWNLOADED
        return ModelState.LOADED
    except httpx.HTTPError:
        return ModelState.UNREACHABLE
