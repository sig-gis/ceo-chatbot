from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ceo_rag_chatbot.schemas import HealthResponse

router = APIRouter()


# exclude_none keeps the response as {"status": "ready"} for hosted models.
@router.get("/healthz", response_model=HealthResponse, response_model_exclude_none=True)
def healthz(request: Request):
    """
    Readiness probe.

    details:
        Returns 200 once lifespan has finished loading the index. Returns 503
        before that so load balancers and Cloud Run don't route traffic too early.
        For a local model it also reports the model's state (e.g. "downloaded").
        This route reads app.state directly and must not use Depends(get_rag),
        because its purpose is to report readiness before the service is ready.

    Args:
        request (Request): The incoming request; used to read app.state.

    Returns:
        HealthResponse | JSONResponse: {"status": "ready"} (plus model_state for a
        local model) with 200, or {"status": "loading"} with 503.
    """
    if getattr(request.app.state, "ready", False):
        model_state = getattr(request.app.state, "model_state", None)
        return HealthResponse(status="ready", model_state=model_state.value if model_state else None)
    return JSONResponse(status_code=503, content={"status": "loading"})
