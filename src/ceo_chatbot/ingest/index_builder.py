from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument

from ceo_chatbot.ingest.loaders import load_rst_docs
from ceo_chatbot.ingest.chunking import semantic_recursive_chunks
from ceo_chatbot.ingest.embeddings import get_embedding_model
from ceo_chatbot.config import load_rag_config


def build_faiss_index(
    docs: List[LangchainDocument],
    config_path: str | Path = "conf/base/rag_config.yml"
) -> FAISS:
    """
    Build a FAISS index from documents using the specified embedding model.
    """
    config = load_rag_config(config_path)
    embedding_model = get_embedding_model(model_name=config.embedding_model_name)
    index = FAISS.from_documents(
        docs,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return index

def build_and_save_index(
    output_dir: Path,
    config_path: str | Path = "conf/base/rag_config.yml"
) -> None:
    """
    Full ingest pipeline:
      - load docs
      - chunk
      - embed and build FAISS
      - save index to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_rag_config(config_path)
    
    # load docs
    raw_docs = load_rst_docs()

    # chunk
    docs_processed = semantic_recursive_chunks(
        knowledge_base=raw_docs,
        chunk_size=config.chunk_size
    )

    # build index
    index = build_faiss_index(
        docs=docs_processed
    )

    # save index
    index.save_local(folder_path=str(output_dir))