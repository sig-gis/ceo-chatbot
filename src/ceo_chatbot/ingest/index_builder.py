from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument

from ceo_chatbot.ingest.loaders import load_rst_docs
from ceo_chatbot.ingest.chunking import semantic_recursive_chunks
from ceo_chatbot.embeddings import get_embedding_model
from ceo_chatbot.ingest.gcs_uploader import GCSHandler
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

def build_and_save_index(
    output_dir: Path,
    rag_config: str | Path = "conf/base/rag_config.yml",
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
    rag_cfg = load_rag_config(rag_config)
    
    # load docs
    raw_docs = load_rst_docs()

    # chunk
    docs_processed = semantic_recursive_chunks(
        knowledge_base=raw_docs,
        chunk_size=rag_cfg.chunk_size
    )

    # build index
    index = build_faiss_index(
        docs=docs_processed
    )

    # save index
    index.save_local(folder_path=str(output_dir))
    print(f"FAISS index saved to {output_dir}")

    # upload the built faiss db's files to gcs
    app_settings = AppSettings()
    gcs_handler = GCSHandler(rag_cfg)
    gcs_handler.upload_db(Path("data/vectorstores/ceo_docs_faiss"))