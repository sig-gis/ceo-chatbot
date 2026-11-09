# Ollama inference server for local models.
# Build from the repo root:
#   docker build -f services/ceo_inference/Dockerfile -t ceo-inference:latest .
FROM ollama/ollama:0.40.15

# Keep models in memory forever instead of unloading them after 5 idle minutes
# (Ollama's default). The chatbot only answers while the model is loaded.
ENV OLLAMA_KEEP_ALIVE=-1

COPY services/ceo_inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Replaces the image's default `ollama serve` with our script, which starts the
# server itself and then downloads and loads the model.
ENTRYPOINT ["/entrypoint.sh"]
