from langchain_huggingface import HuggingFaceEmbeddings
import torch


def get_embedding_model(
    model_name: str = "thenlper/gte-small",
    multi_process: bool = False,  # WARNING: disabled for testing purposes ... 
) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFaceEmbeddings instance configured for cosine similarity on a GPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        multi_process=multi_process,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
