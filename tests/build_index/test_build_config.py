import pytest
from pydantic import ValidationError

from ceo_build_index.config import RAGConfig, load_rag_config


def test_valid_config():
    """A well-formed config is accepted and fields are stored correctly."""
    config = RAGConfig(
        embedding_model_name="mymodel",
        chunk_size=512,
        docs_path="data/docs",
        vectorstore_path="data/vs",
        vectorstore_gcs="gcs/path",
    )
    assert config.chunk_size == 512


def test_chunk_size_must_be_positive():
    """chunk_size=0 raises ValidationError; a zero value would silently produce an empty index."""
    with pytest.raises(ValidationError):
        RAGConfig(
            embedding_model_name="mymodel",
            chunk_size=0,
            docs_path="data/docs",
            vectorstore_path="data/vs",
            vectorstore_gcs="gcs/path",
        )


def test_load_rag_config_reads_embeddings_section(tmp_path):
    """Values are read from the embeddings: YAML section with the correct field mapping."""
    yml = tmp_path / "rag_config.yml"
    yml.write_text(
        "embeddings:\n"
        "  embedding_model_name: mymodel\n"
        "  chunk_size: 256\n"
        "  docs_path: data/docs\n"
        "  vectorstore_path: data/vs\n"
        "  vectorstore_gcs: bucket/path\n"
    )
    config = load_rag_config(yml)
    assert config.embedding_model_name == "mymodel"
    assert config.chunk_size == 256
