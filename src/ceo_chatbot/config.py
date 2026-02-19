import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}
class RAGConfig(BaseModel):    
    embedding_model_name: str
    chunk_size: int = Field(gt=0, description="Chunk size must be positive")
    llm_framework: str
    model_choices: List[str]
    vectorstore_path: Path
    prompt_file: Path
class DocumentExtractionConfig(BaseModel):
    github_repo_url: str = Field(pattern=r'^https?://', description="Must be a valid HTTP/HTTPS URL")
    gcs_project_id: str
    gcs_bucket_name: str
    github_ref: str = "main"
    github_path: str = ""
    gcs_prefix: str = ""

def load_rag_config(
    base_path: str | Path = PROJECT_ROOT / "conf/base/rag_config.yml",
) -> RAGConfig:
    """
    Load the RAG configuration from YAML to dataclass RAGConfig to configure pipeline.

    - base_path: Path(conf/base/rag_config.yml)
    """
    base_path = Path(base_path)
    base_cfg = _load_yaml(base_path)

    return RAGConfig(**base_cfg)

def load_document_extraction_config(
    config_path: str | Path = PROJECT_ROOT / "conf/base/extract_docs_config.yml",
) -> DocumentExtractionConfig:
    """
    Load the document extraction configuration from YAML to dataclass DocumentExtractionConfig.

    - config_path: Path to the document extraction configuration file
    """
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)

    # Extract nested values with defaults
    github = cfg.get("github", {})
    gcs = cfg.get("gcs", {})

    return DocumentExtractionConfig(
        github_repo_url=github["repo_url"],
        github_ref=github.get("ref", "main"),
        github_path=github.get("path", ""),
        gcs_project_id=gcs["project_id"],
        gcs_bucket_name=gcs["bucket_name"],
        gcs_prefix=gcs.get("prefix", ""),
    )

def load_prompt_template(
    prompt_file: str | Path,
    prompt_key: str = "default_prompt",
) -> List[Dict[str, Any]]:
    """
    Load a chat template (list of {role, content} dicts) from a YAML file.

    Example YAML structure:

    default_prompt:
      - role: system
        content: "..."
      - role: user
        content: "..."
    """
    prompt_file = PROJECT_ROOT / prompt_file
    path = Path(prompt_file)
    data = _load_yaml(path)

    if prompt_key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Prompt key '{prompt_key}' not found in {prompt_file}. "
            f"Available keys: {available}"
        )

    template = data[prompt_key]
    if not isinstance(template, list):
        raise ValueError(
            f"Prompt '{prompt_key}' in {prompt_file} is not a list of messages."
        )

    return template
class AppSettings(BaseSettings):
    """Loads all required environment variables"""
    # Define fields at the top level, using aliases to map to env vars.
    # This is the most robust way to load them.
    google_api_key: str = Field(..., alias='GOOGLE_API_KEY')

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )
