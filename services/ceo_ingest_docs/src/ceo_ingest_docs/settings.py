from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Environment variables required by the ingestion service.

    Populated automatically from process env / .env at construction time.
    """

    docs_bucket_name: str = Field(alias="DOCS_BUCKET")
    google_cloud_project: str = Field(..., alias="GOOGLE_CLOUD_PROJECT")
    folder_prefix: str = Field(alias="PREFIX")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_ingestion_settings() -> IngestionSettings:
    """Return a cached IngestionSettings instance so env / .env is read once."""
    return IngestionSettings()
