from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from ceo_chatbot_core.yaml_utils import load_yaml

class ModelConfig(BaseModel):
    """One entry under `llm.models` in rag_config.yml: an OpenAI-compatible endpoint and model."""
    base_url: str
    model: str
    api_key_env: Optional[str] = None  # name of the env var holding the key; None for keyless servers (Ollama)
    context_window: int
    max_output_tokens: int = 8192
    temperature: float = 0.2
    timeout: float = 60
    fallback: List[str] = []  # extra model names tried in order on the same base_url and key
    probe: bool = False  # check local server readiness at startup
    json_mode: bool = False  # server supports response_format json_object; check the provider's docs first

class ChatbotConfig(BaseModel):
    """
    Class that stores all config fields necessary to run the chatbot.
    Reads from the YAML file.
    """
    embedding_model_name: str
    vectorstore_path: Path
    vectorstore_gcs: Path
    models: Dict[str, ModelConfig]
    active_model: str
    prompt_file: Path

    @model_validator(mode="after")
    def _active_model_exists(self):
        """
        Check that active_model names one of the entries under models.

        Runs automatically when the config is created, so a typo in
        rag_config.yml fails at startup instead of on the first chat request.

        Returns:
            ChatbotConfig: The config itself, unchanged.

        Raises:
            ValueError: If active_model is not a key in models. The message lists the valid names.
        """
        if self.active_model not in self.models:
            raise ValueError(
                f"active_model '{self.active_model}' is not one of the configured models: {list(self.models)}"
            )
        return self

    @property
    def llm(self) -> ModelConfig:
        """
        The config entry for the model currently in use.

        Returns:
            ModelConfig: The entry under models named by active_model.
        """
        return self.models[self.active_model]

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
        models=llm["models"],
        active_model=llm["active_model"],
        prompt_file=llm.get("prompt_file","conf/base/prompts.yml"),
    )

class ChatbotSettings(BaseSettings):
    """Env vars the chatbot service reads through pydantic.

    Holds only what the chatbot needs, so the chatbot container does not need
    pipeline-only variables (DOCS_BUCKET, PREFIX). HF_TOKEN and the LLM API key
    are not fields here: they are read from os.environ directly (see
    get_reader_llm), because pydantic reading .env does not populate os.environ.
    """
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