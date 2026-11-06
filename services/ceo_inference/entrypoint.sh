#!/bin/bash
# Starts the Ollama server, makes sure OLLAMA_MODEL is downloaded, and loads
# it into VRAM so the chatbot's first request doesn't have to wait.
set -euo pipefail

: "${OLLAMA_MODEL:?Set OLLAMA_MODEL to the model name in rag_config.yml (models.local.model)}"

ollama serve &
server_pid=$!

# The image has no curl, so use the CLI to wait: `ollama list` fails until the server answers.
until ollama list >/dev/null 2>&1; do sleep 1; done

# Use the cached copy in /root/.ollama/models if it's there; otherwise download it.
# The list is stored first because piping straight into `grep -q` can fail under pipefail.
cached_models=$(ollama list | awk 'NR > 1 {print $1}')
if grep -Fxq "$OLLAMA_MODEL" <<< "$cached_models"; then
  echo "Model $OLLAMA_MODEL found in cache, skipping download"
else
  echo "Model $OLLAMA_MODEL not in cache, downloading from the Ollama registry"
  ollama pull "$OLLAMA_MODEL"
fi

# An empty prompt loads the model without generating anything (Ollama FAQ: "preload a model").
# OLLAMA_KEEP_ALIVE=-1 (set in the Dockerfile) keeps it loaded afterwards.
ollama run "$OLLAMA_MODEL" ""
echo "Model $OLLAMA_MODEL loaded into memory"

# Keep the container running for as long as the server runs.
wait "$server_pid"
