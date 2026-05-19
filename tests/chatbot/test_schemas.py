import pytest
from pydantic import ValidationError

from ceo_rag_chatbot.schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse


def test_chat_message_valid_roles():
    """'user' and 'assistant' are the only accepted role values."""
    ChatMessage(role="user", content="hello")
    ChatMessage(role="assistant", content="hi")


def test_chat_message_rejects_invalid_role():
    """Any role outside the Literal raises ValidationError, guarding the API contract."""
    with pytest.raises(ValidationError):
        ChatMessage(role="system", content="ignored")


def test_chat_request_defaults_history_to_empty():
    """Omitting history produces [] not None, so callers never need a None check."""
    req = ChatRequest(query="What is CEO?")
    assert req.history == []


def test_chat_request_accepts_history():
    """A populated history list is stored correctly for multi-turn conversations."""
    req = ChatRequest(
        query="follow up",
        history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
    )
    assert len(req.history) == 2


def test_chat_response_fields():
    """answer and sources fields round-trip correctly through the response model."""
    resp = ChatResponse(answer="CEO is a tool.", sources=["https://example.com"])
    assert resp.answer == "CEO is a tool."
    assert resp.sources == ["https://example.com"]


def test_health_response_valid_statuses():
    """'ready' and 'loading' are the only valid status values."""
    HealthResponse(status="ready")
    HealthResponse(status="loading")


def test_health_response_rejects_invalid_status():
    """Any status string outside the Literal raises ValidationError."""
    with pytest.raises(ValidationError):
        HealthResponse(status="error")
