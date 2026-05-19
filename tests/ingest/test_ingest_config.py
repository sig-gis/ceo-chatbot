import pytest
from pydantic import ValidationError

from ceo_ingest_docs.config import IngestionConfig, load_ingestion_config


def test_valid_config():
    """A valid HTTPS URL is accepted; ref defaults to 'main' and path to ''."""
    config = IngestionConfig(repo_url="https://github.com/org/repo.git")
    assert config.ref == "main"
    assert config.path == ""


def test_url_must_be_http():
    """An SSH-style URL (git@github.com:…) raises ValidationError before any subprocess call."""
    with pytest.raises(ValidationError):
        IngestionConfig(repo_url="git@github.com:org/repo.git")


def test_load_ingestion_config_reads_github_section(tmp_path):
    """Reads repo_url, ref, and path from the github: section of a YAML file."""
    yml = tmp_path / "rag_config.yml"
    yml.write_text(
        "github:\n"
        "  repo_url: https://github.com/org/repo.git\n"
        "  ref: v2\n"
        "  path: docs/source\n"
    )
    config = load_ingestion_config(yml)
    assert config.repo_url == "https://github.com/org/repo.git"
    assert config.ref == "v2"
    assert config.path == "docs/source"


def test_load_ingestion_config_defaults(tmp_path):
    """Omitted ref and path in YAML produce the correct defaults."""
    yml = tmp_path / "rag_config.yml"
    yml.write_text("github:\n  repo_url: https://github.com/org/repo.git\n")
    config = load_ingestion_config(yml)
    assert config.ref == "main"
    assert config.path == ""
