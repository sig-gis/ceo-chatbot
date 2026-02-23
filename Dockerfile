
# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-trixie

# python 3.10 astral images:
# ghcr.io/astral-sh/uv:python3.10-alpine
# ghcr.io/astral-sh/uv:python3.10-trixie
# ghcr.io/astral-sh/uv:python3.10-trixie-slim

# For Docker Trixie base image - shadowutils install
# RUN apt-get update && apt-get install -y shadow && rm -rf /var/lib/apt/lists/* # didn't work

# # For Docker Alpine base image - shadowutils install
# # RUN apk add --no-cache shadow \

# # Setup a non-root user - see if installing shadowutils resolves groupadd cmd not found
# RUN groupadd --system --gid 1001 nonroot \
#  && useradd --system --gid 1001 --uid 1001 --create-home nonroot

# INSTALL GCLOUD SDK ##############
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH="$PATH:/usr/local/gcloud/google-cloud-sdk/bin"

###### GCLOUD end ####### 
# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# install ceo-chatbot as a package
RUN uv pip install .

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Use the non-root user to run our application
# USER nonroot

# Run the FastAPI application by default
# Uses `uv run` to sync dependencies on startup, respecting UV_NO_DEV
# Uses `fastapi dev` to enable hot-reloading when the `watch` sync occurs
# Uses `--host 0.0.0.0` to allow access from outside the container
# Note in production, you should use `fastapi run` instead
# CMD ["uv", "run", "fastapi", "dev", "--host", "0.0.0.0", "src/uv_docker_example"]

# Command to run the Streamlit app
# CMD ["uv", "run", "streamlit", "run", "demo/chat_app.py", "--server.port", "8080", "--server.enableCORS", "true", "--server.enableXsrfProtection", "false"]
# run gcloud auth first
CMD ["/bin/bash", "-c", "gcloud auth activate-service-account --key-file=/app/keyfile.json && uv run streamlit run demo/chat_app.py --server.port 8080 --server.enableCORS true --server.enableXsrfProtection false"]

# # Expose the Streamlit port
EXPOSE 8080

