from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGSettings(BaseSettings):
    """Environment variables required by the index building service.

    Populated automatically from process env / .env at construction time.
    """

    docs_bucket_name: str = Field(...,alias='DOCS_BUCKET')
    folder_prefix: str = Field(...,alias='PREFIX')
    google_cloud_project: str = Field(..., alias="GCP_PROJECT_ID")
    db_bucket_name: str = Field(...,alias='DB_BUCKET')

    # This tells BaseSettings where to find the above fields.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

@lru_cache
def get_index_settings() -> RAGSettings:
    """Return a cached IngestionSettings instance so env / .env is read once."""
    return RAGSettings()
