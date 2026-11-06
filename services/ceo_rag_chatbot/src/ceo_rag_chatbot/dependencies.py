"""FastAPI dependency functions.
"""
from fastapi import HTTPException, Request
from ceo_rag_chatbot.rag.local_probe import ModelState
from ceo_rag_chatbot.rag.pipeline import RagService

def get_rag(request: Request) -> RagService:
    """
    Return the loaded RagService for a route that needs it.

    details:
        Failing fast here gives the user a clear message instead of a request
        that hangs until it times out. Hosted models (model_state is None) only
        need startup to have finished.

    Args:
        request (Request): The incoming request; used to read app.state.

    Returns:
        RagService: The service created at startup.

    Raises:
        HTTPException: 503 if startup hasn't finished, or if a local model is
            being probed and isn't loaded in VRAM yet (the message includes its state).
    """
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="service still loading")
    # model_state is None for hosted models (no probe), so they skip this check.
    model_state = getattr(request.app.state, "model_state", None)
    if model_state is not None and model_state != ModelState.LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"The local model is not ready yet (state: {model_state.value}). Try again shortly.",
        )
    return request.app.state.rag
