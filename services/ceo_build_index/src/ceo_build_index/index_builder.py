from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument

from ceo_chatbot.ingest.loaders import load_rst_docs
from ceo_chatbot.ingest.chunking import semantic_recursive_chunks
from ceo_chatbot.embeddings import get_embedding_model
from ceo_chatbot.config import load_rag_config, AppSettings


def build_faiss_index(
    docs: List[LangchainDocument],
    config_path: str | Path = "conf/base/rag_config.yml",
    device: str | None = None,
) -> FAISS:
    """
    Build a FAISS index from documents using the specified embedding model.
    """
    config = load_rag_config(config_path)
    embedding_model = get_embedding_model(embedding_model_name=config.embedding_model_name, device=device)
    index = FAISS.from_documents(
        docs,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return index