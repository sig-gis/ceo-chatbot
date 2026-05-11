import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from ceo_chatbot_core.yaml_utils import load_yaml

class ChatbotConfig(BaseModel):
    """
    Class that stores all config fields necessary to run the chatbot.
    Reads from the YAML file.
    """
    embedding_model_name: str
    vectorstore_path: Path
    vectorstore_gcs: Path
    llm_framework: str
    model_choices: List[str]
    max_output_tokens: int = Field(gt=1024, description="this can also be configured downstream")
    prompt_file: Path

@lru_cache
def load_chatbot_config(
    config_path: str | Path = "conf/base/rag_config.yml",
) -> ChatbotConfig:
    """
    Load the RAG configuration from YAML to dataclass ChatbotConfigfig to configure pipeline.

    - base_path: Path(conf/base/rag_config.yml)
    """
    config_path = Path(config_path)
    cfg = load_yaml(config_path)

    # Extract nested values with defaults
    embeddings = cfg.get("embeddings", {})
    llm = cfg.get("llm", {})

    return ChatbotConfig(
        embedding_model_name=embeddings["embedding_model_name"],
        vectorstore_path=embeddings.get("vectorstore_path", "data/vectorstores/ceo_docs_faiss"),
        vectorstore_gcs=embeddings.get("vectorstore_gcs", "ceo-docs-faiss"),
        llm_framework=llm.get("llm_framework","gemini"),
        model_choices=llm.get("model_choices",["gemini-2.5-flash"]),
        max_output_tokens=llm.get("max_output_tokens", 8192),
        prompt_file=llm.get("prompt_file","conf/base/prompts.yml"),
    )

class ChatbotSettings(BaseSettings):
    """Env vars required for inference only (chatbot service).

    Kept separate from AppSettings so the chatbot container does not need
    pipeline-specific variables (DB_BUCKET, DOCS_BUCKET, PREFIX, HF_TOKEN).
    """
    gemini_api_key: str = Field(..., alias='GEMINI_API_KEY')
    # GCP project ID. storage.Client() requires this; it cannot be inferred
    # from user ADC credentials (only service-account keys embed a project).
    google_cloud_project: str = Field(..., alias="GOOGLE_CLOUD_PROJECT")

    #  GCS bucket that holds the FAISS index uploaded by the pipeline job
    db_bucket_name: str = Field(..., alias="DB_BUCKET")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> ChatbotSettings:
    """Return a cached ChatbotSettings instance.

    lru_cache ensures ChatbotSettings() is only constructed once — env vars and the
    .env file are read a single time, not on every request.
    """
    return ChatbotSettings()

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
    prompt_file = prompt_file
    path = Path(prompt_file)
    data = load_yaml(path)

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