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
    docs_path: Path
    vectorstore_path: Path
    vectorstore_gcs: Path
    prompt_file: Path
    max_output_tokens: int = Field(gt=1024, description="this can also be configured downstream")
    github_repo_url: str = Field(pattern=r'^https?://', description="Must be a valid HTTP/HTTPS URL")
    github_ref: str = "main"
    github_path: str = ""

class AppSettings(BaseSettings):
    """Loads all required environment variables"""
    # Define fields at the top level, using aliases to map to env vars.

    google_application_credentials: str = Field(..., alias='GOOGLE_APPLICATION_CREDENTIALS')
    gcp_project_id: str = Field(...,alias='GCP_PROJECT_ID')
    db_bucket_name: str = Field(...,alias='DB_BUCKET')
    docs_bucket_name: str = Field(...,alias='DOCS_BUCKET')
    folder_prefix: str = Field(...,alias='PREFIX')
    gemini_api_key: str = Field(..., alias='GEMINI_API_KEY')
    huggingface_token: str = Field(...,alias='HF_TOKEN')

    # This tells BaseSettings where to find the above fields.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

class GeminiInferenceSettings(BaseSettings):
    """Env vars required for inference only (chatbot service).

    Kept separate from AppSettings so the chatbot container does not need
    pipeline-specific variables (DB_BUCKET, DOCS_BUCKET, PREFIX, HF_TOKEN).
    """
    gemini_api_key: str = Field(..., alias='GEMINI_API_KEY')

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

def load_rag_config(
    config_path: str | Path = PROJECT_ROOT / "conf/base/rag_config.yml",
) -> RAGConfig:
    """
    Load the RAG configuration from YAML to dataclass RAGConfig to configure pipeline.

    - base_path: Path(conf/base/rag_config.yml)
    """
    config_path = Path(config_path)
    cfg = _load_yaml(config_path)

    # Extract nested values with defaults
    embeddings = cfg.get("embeddings", {})
    llm = cfg.get("llm", {})
    github = cfg.get("github",{})

    return RAGConfig(
        embedding_model_name=embeddings["embedding_model_name"],
        chunk_size=embeddings.get("chunk_size", 512),
        docs_path=embeddings.get("docs_path", "data/ceo-docs"),
        vectorstore_path=embeddings.get("vectorstore_path", "data/vectorstores/ceo_docs_faiss"),
        vectorstore_gcs=embeddings.get("vectorstore_gcs", "ceo-docs-faiss"),
        llm_framework=llm.get("llm_framework","gemini"),
        model_choices=llm.get("model_choices",["gemini-2.5-flash"]),
        max_output_tokens=llm.get("max_output_tokens", 8192),
        prompt_file=llm.get("prompt_file","conf/base/prompts.yml"),
        github_repo_url=github["repo_url"],
        github_ref=github["ref"],
        github_path=github["path"]
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
