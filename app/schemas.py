from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single turn in a conversation."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    The chatbot is stateless: the server keeps no conversation history between
    requests. The client must send the full conversation so far in `history`
    with every request. This is required for correct operation across multiple
    Cloud Run instances, which share no memory.
    """
    query: str
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str
    sources: list[str]


class HealthResponse(BaseModel):
    """Response body for GET /healthz."""
    status: Literal["ready", "loading"]
