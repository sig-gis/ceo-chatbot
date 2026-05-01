from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding_model(
    embedding_model_name: str = "thenlper/gte-small",
    multi_process: bool = False,
    device: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFaceEmbeddings instance configured for cosine similarity.
    If device is None, auto-detects cuda or falls back to cpu.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=multi_process,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
