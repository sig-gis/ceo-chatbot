from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from ceo_chatbot.ingest.loaders import load_hf_docs
from ceo_chatbot.ingest.chunking import split_documents, DEFAULT_EMBEDDING_MODEL_NAME
from ceo_chatbot.ingest.embeddings import get_embedding_model


def build_faiss_index(
    docs: list[LangchainDocument],
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> FAISS:
    """
    Build a FAISS index from documents using the specified embedding model.
    """
    embedding_model = get_embedding_model(model_name=embedding_model_name)
    index = FAISS.from_documents(
        docs,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )
    return index


def build_and_save_index(
    output_dir: Path,
    dataset_name: str = "m-ric/huggingface_doc",
    split: str = "train",
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
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
    raw_docs = load_hf_docs(dataset_name=dataset_name, split=split)

    # chunk
    docs_processed = split_documents(
        knowledge_base=raw_docs,
        chunk_size=512,
        tokenizer_name=embedding_model_name,
    )

    # build index
    index = build_faiss_index(
        docs=docs_processed,
        embedding_model_name=embedding_model_name,
    )

    # save index
    index.save_local(folder_path=str(output_dir))
