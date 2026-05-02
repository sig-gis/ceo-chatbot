from fastapi import APIRouter, Depends

from app.dependencies import get_rag
from app.schemas import ChatRequest, ChatResponse
from ceo_chatbot.rag.pipeline import RagService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, rag: RagService = Depends(get_rag)):
    """Answer a question about Collect Earth Online using RAG.

    The client must send the full conversation history with each request —
    the server is stateless and stores nothing between calls.
    """
    # We call answer() (non-streaming) rather than stream_answer() because this
    # endpoint returns a single JSON response. Streaming can be done separately
    # in the future.
    #
    # history is converted from ChatMessage objects to plain dicts because
    # RagService expects List[Dict[str, str]].
    answer, docs = rag.answer(
        question=req.query,
        history=[m.model_dump() for m in req.history],
        num_retrieved_docs=30,
        num_docs_final=5,
    )

    # doc.metadata["url"] is the ReadTheDocs URL set by loaders.py for each chunk
    sources = [doc.metadata.get("url", "unknown") for doc in docs]

    return ChatResponse(answer=answer, sources=sources)
