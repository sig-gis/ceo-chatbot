from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestionConfig(BaseModel):
    """Static configuration describing what to ingest.

    Loaded from the `github:` section of conf/base/rag_config.yml.
    """

    repo_url: str = Field(pattern=r"^https?://", description="HTTP/HTTPS clone URL")
    ref: str = "main"
    path: str = ""


def load_ingestion_config(config_path: str | Path) -> IngestionConfig:
    """Parse the `github:` section of a YAML file into an IngestionConfig."""
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    # Since the field names match YAML keys, simply giving it the github dict
    # is sufficient as its nested keys correspond to each other
    return IngestionConfig(**raw.get("github", {}))
