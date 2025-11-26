from pathlib import Path
from typing import Callable, List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangchainDocument
from ceo_chatbot.ingest.embeddings import get_embedding_model


def load_faiss_index(
    index_dir: Path,
    embedding_model_name: str = "thenlper/gte-small",
) -> FAISS:
    """
    Load a FAISS index from disk with a compatible embedding model.
    """
    embedding_model = get_embedding_model(model_name=embedding_model_name)
    index = FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,  # necessary if running locally / trusted
    )
    return index


def get_retriever(index: FAISS) -> Callable[[str, int], List[LangchainDocument]]:
    """
    Wrap FAISS index into a simple callable retriever(question, k) -> docs.
    """
    def _retrieve(query: str, k: int = 5) -> List[LangchainDocument]:
        return index.similarity_search(query=query, k=k)

    return _retrieve
