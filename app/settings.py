from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment variables required by the chatbot service at runtime.

    Kept separate from AppSettings (pipeline) and GeminiInferenceSettings so
    the chatbot container only needs the vars it actually uses.
    """
    # GEMINI_API_KEY is also read directly by GeminiInferenceSettings inside
    # RagService — this field documents it as a chatbot requirement.
    google_api_key: str = Field(..., alias="GEMINI_API_KEY")

    # GCS bucket that holds the FAISS index uploaded by the pipeline job
    # (same bucket as DB_BUCKET used by the pipeline)
    gcs_bucket_index: str = Field(..., alias="DB_BUCKET")

    # GCP project ID. storage.Client() requires this; it cannot be inferred
    # from user ADC credentials (only service-account keys embed a project).
    google_cloud_project: str = Field(..., alias="GCP_PROJECT_ID")

    # HuggingFace token. Required for gated models (e.g. google/embeddinggemma-300m
    # requires accepting Google's terms on HuggingFace). The huggingface_hub
    # library reads HF_TOKEN from the environment automatically — this field
    # exists to document the requirement and surface a clear error if it is missing.
    hf_token: str = Field(..., alias="HF_TOKEN")

    # Local directory to download the index into on startup
    index_local_dir: Path = Field(default=Path("/tmp/ceo-chatbot-index"), alias="INDEX_LOCAL_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    lru_cache ensures Settings() is only constructed once — env vars and the
    .env file are read a single time, not on every request.
    """
    return Settings()
