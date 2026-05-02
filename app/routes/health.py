from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request):
    """Readiness probe.

    Returns 200 once lifespan has finished loading the index. Returns 503
    before that so load balancers and Cloud Run don't route traffic too early.
    This route reads app.state directly and must not use Depends(get_rag),
    because its purpose is to report readiness before the service is ready.
    """
    if getattr(request.app.state, "ready", False):
        return HealthResponse(status="ready")
    return JSONResponse(status_code=503, content={"status": "loading"})
