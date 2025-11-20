from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from ceo_chatbot.ingest.loaders import load_hf_docs
from ceo_chatbot.ingest.chunking import split_documents
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

# Huggingface docs will need to be replaced with CEO docs
# and build_and_save index will need to be updated:
# raw_docs = <replace load_hf_docs() with load CEO docs logic>
def build_and_save_index(
    output_dir: Path,
) -> None:
    """
    Full ingest pipeline:
      - load HF docs
      - chunk
      - embed and build FAISS
      - save index to disk
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load docs
    raw_docs = load_hf_docs(dataset_name="m-ric/huggingface_doc", split="train")

    # chunk
    docs_processed = split_documents(
        knowledge_base=raw_docs,
        chunk_size=512
    )

    # build index
    index = build_faiss_index(
        docs=docs_processed
    )

    # save index
    index.save_local(folder_path=str(output_dir))
