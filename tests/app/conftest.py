"""Shared pytest fixtures for the FastAPI app test suite.

Architecture note
-----------------
The FastAPI app (app/main.py) has a *lifespan* — an async startup/shutdown hook
that downloads a FAISS vector index from Google Cloud Storage and loads an
embedding model. That is fine in production but would make every test require
live GCS credentials and several seconds of model loading. The fixtures here
prevent that from happening.

The app is also a module-level singleton (`app = FastAPI(...)` in app/main.py).
That means all tests share the same object. The autouse fixtures below ensure
each test starts with a clean, predictable state.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from starlette.testclient import TestClient

from app.main import app
from app.dependencies import get_rag


# autouse=True means this fixture runs automatically for every single test,
# even tests that don't explicitly request it.
@pytest.fixture(autouse=True)
def no_op_lifespan():
    """Replace the app lifespan with a no-op to prevent GCS/model loading in tests.

    How it works
    ------------
    FastAPI stores the lifespan coroutine on app.router.lifespan_context.
    Swap it out for a coroutine that does nothing (_no_op), run the test,
    then restore the original so other tests / the real server are unaffected.

    Without this, TestClient would trigger the real lifespan on every `with`
    block, which would try to connect to GCS and load a multi-GB model.
    """
    # no-op just means it doesn't do anything — it yields immediately so
    # FastAPI's startup/shutdown hooks run but with no side effects.
    @asynccontextmanager
    async def _no_op(a):
        yield

    original = app.router.lifespan_context
    app.router.lifespan_context = _no_op
    yield                                    # test runs here
    app.router.lifespan_context = original   # restore for next test


@pytest.fixture(autouse=True)
def reset_app_state():
    """Ensure clean app state before each test."""
    # ready=False simulates the app before startup has finished.
    # dependency_overrides is a dict FastAPI checks when resolving Depends();
    # clearing it ensures no mock from a previous test bleeds through.
    app.state.ready = False
    app.dependency_overrides.clear()


@pytest.fixture
def mock_rag():
    """Create a fake RagService for use in tests.

    RagService is the class that wraps the LLM + retriever pipeline. In
    production it loads a FAISS index and calls the Gemini API. Here we
    replace it with a MagicMock so tests run instantly with no network calls.

    Pre-configured return value
    ---------------------------
    mock_rag.answer() returns ("Test answer.", [mock_doc]) where mock_doc
    has metadata["url"] set. Tests that need different behaviour override
    mock_rag.answer.return_value inline before making the HTTP call.
    """
    mock_doc = MagicMock()
    mock_doc.metadata = {"url": "https://example.com/page.html"}
    rag = MagicMock()
    rag.answer.return_value = ("Test answer.", [mock_doc])
    return rag


@pytest.fixture
def client(mock_rag):
    """Return a TestClient configured as if the app has finished startup.

    What this sets up
    -----------------
    1. app.state.ready = True  — tells /healthz and get_rag the app is ready.
    2. app.dependency_overrides[get_rag] — FastAPI checks this dict before
       calling the real get_rag(). We point it at a lambda that returns the
       mock, so no real RagService is ever instantiated.
    3. `with TestClient(app)` — the `with` block triggers the lifespan
       (now the no-op from no_op_lifespan) so Starlette's request lifecycle
       is properly initialised.

    Tests that need the *not-ready* state (e.g. testing 503 responses) should
    NOT use this fixture — they create their own TestClient inline so that
    app.state.ready stays False.
    """
    app.state.ready = True
    app.state.rag = mock_rag
    # dependency_overrides is FastAPI's built-in testing escape hatch:
    # any Depends(get_rag) in a route will call this lambda instead.
    app.dependency_overrides[get_rag] = lambda: mock_rag
    with TestClient(app) as c:
        yield c
    # Teardown: clear overrides and reset ready flag for the next test.
    app.dependency_overrides.clear()
    app.state.ready = False
