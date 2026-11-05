from unittest.mock import MagicMock, patch

import pytest

from ceo_rag_chatbot.rag.llm import OpenAIProvider


def _completion(text):
    """Build a fake non-streaming chat completion response."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _chunk(text):
    """Build a fake streaming chunk."""
    return MagicMock(choices=[MagicMock(delta=MagicMock(content=text))])


@pytest.fixture
def client():
    # Patch the OpenAI class so no network client is created.
    with patch("ceo_rag_chatbot.rag.llm.OpenAI") as openai_cls:
        yield openai_cls.return_value


def _provider(models=("primary", "backup")):
    return OpenAIProvider(
        base_url="http://example/v1", api_key="k", models=list(models),
        context_window=1000, max_output_tokens=100,
    )


def test_generate_returns_message_content(client):
    client.chat.completions.create.return_value = _completion("hello")
    assert _provider().generate("hi") == "hello"


def test_generate_falls_back_to_next_model(client):
    client.chat.completions.create.side_effect = [RuntimeError("down"), _completion("ok")]
    assert _provider().generate("hi") == "ok"
    assert client.chat.completions.create.call_args.kwargs["model"] == "backup"


def test_generate_raises_last_error_when_all_models_fail(client):
    client.chat.completions.create.side_effect = [RuntimeError("a"), RuntimeError("b")]
    with pytest.raises(RuntimeError, match="b"):
        _provider().generate("hi")


def test_stream_yields_chunks_in_order(client):
    client.chat.completions.create.return_value = iter([_chunk("a"), _chunk("b"), _chunk(None)])
    assert list(_provider().stream("hi")) == ["a", "b"]
