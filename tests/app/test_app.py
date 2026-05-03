import pytest
from unittest.mock import MagicMock, call
from fastapi import HTTPException
from starlette.datastructures import State
from starlette.testclient import TestClient

from app.main import app
from app.dependencies import get_rag


# ---------------------------------------------------------------------------
# Group 1: GET /healthz
# ---------------------------------------------------------------------------

def test_healthz_loading():
    # reset_app_state autouse fixture ensures ready=False
    c = TestClient(app)
    response = c.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "loading"}


def test_healthz_ready(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


# ---------------------------------------------------------------------------
# Group 2: POST /chat — readiness gate
# ---------------------------------------------------------------------------

def test_chat_503_when_not_ready():
    # reset_app_state ensures ready=False and no dependency_overrides
    c = TestClient(app)
    response = c.post("/chat", json={"query": "What is CEO?"})
    assert response.status_code == 503
    assert "loading" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Group 3: POST /chat — happy path
# ---------------------------------------------------------------------------

def test_chat_returns_answer_and_sources(client):
    response = client.post("/chat", json={"query": "What is CEO?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer."
    assert data["sources"] == ["https://example.com/page.html"]


def test_chat_no_history_defaults_to_empty(client, mock_rag):
    response = client.post("/chat", json={"query": "What is CEO?"})
    assert response.status_code == 200
    _, kwargs = mock_rag.answer.call_args
    assert kwargs["history"] == []


def test_chat_passes_history_as_dicts(client, mock_rag):
    history = [{"role": "user", "content": "Hi"}]
    client.post("/chat", json={"query": "follow up", "history": history})
    _, kwargs = mock_rag.answer.call_args
    assert kwargs["history"] == [{"role": "user", "content": "Hi"}]


def test_chat_empty_sources_when_no_docs(client, mock_rag):
    mock_rag.answer.return_value = ("Answer.", [])
    response = client.post("/chat", json={"query": "What is CEO?"})
    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_chat_source_falls_back_to_unknown(client, mock_rag):
    mock_doc = MagicMock()
    mock_doc.metadata = {}  # no "url" key
    mock_rag.answer.return_value = ("Answer.", [mock_doc])
    response = client.post("/chat", json={"query": "What is CEO?"})
    assert response.status_code == 200
    assert response.json()["sources"] == ["unknown"]


# ---------------------------------------------------------------------------
# Group 4: Schema validation (FastAPI returns 422 automatically)
# ---------------------------------------------------------------------------

def test_chat_missing_query(client):
    response = client.post("/chat", json={})
    assert response.status_code == 422


def test_chat_invalid_role(client):
    response = client.post("/chat", json={
        "query": "x",
        "history": [{"role": "admin", "content": "hi"}],
    })
    assert response.status_code == 422


def test_chat_invalid_history_type(client):
    response = client.post("/chat", json={"query": "x", "history": "not a list"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Group 5: get_rag dependency — unit tested directly, no HTTP
# ---------------------------------------------------------------------------

def test_get_rag_raises_503_when_not_ready():
    request = MagicMock()
    request.app.state = State()  # real State: getattr returns default when attr missing

    with pytest.raises(HTTPException) as exc_info:
        get_rag(request)
    assert exc_info.value.status_code == 503


def test_get_rag_returns_rag_when_ready(mock_rag):
    request = MagicMock()
    request.app.state = State()
    request.app.state.ready = True
    request.app.state.rag = mock_rag

    result = get_rag(request)
    assert result is mock_rag
