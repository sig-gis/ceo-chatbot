"""FastAPI dependency functions.
"""
from fastapi import HTTPException, Request

from ceo_chatbot.rag.pipeline import RagService


def get_rag(request: Request) -> RagService:
    """Return the loaded RagService. Raises 503 if lifespan hasn't finished yet."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service still loading")
    return request.app.state.rag
