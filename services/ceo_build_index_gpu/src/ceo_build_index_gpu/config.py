from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Static configuration describing what to ingest.

    Loaded from the `github:` section of conf/base/rag_config.yml.
    """

    embedding_model_name: str
    chunk_size: int = Field(gt=0, description="Chunk size must be positive")
    docs_path: Path
    vectorstore_path: Path
    vectorstore_gcs: Path


def load_rag_config(config_path="conf/base/rag_config.yml") -> RAGConfig:
    """Parse the `embeddings:` section of a YAML file into a RAGConfig."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    return RAGConfig(**raw.get("embeddings", {}))
