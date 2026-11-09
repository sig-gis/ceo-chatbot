from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ceo_rag_chatbot.rag.pipeline import RagService

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def rag():
    """RagService with a fake LLM (no network) and a retriever that finds nothing."""
    llm = MagicMock()
    llm.max_context_tokens = 100_000
    llm.count_tokens.side_effect = lambda text: len(text) // 4
    with patch("ceo_rag_chatbot.rag.pipeline.get_reader_llm", return_value=llm):
        yield RagService(
            config_path=REPO_ROOT / "conf/base/rag_config.yml",
            retriever=lambda query, k: [],
        )


def test_prompt_keeps_separate_chat_roles(rag):
    history = [
        {"role": "user", "content": "What is CEO?"},
        {"role": "assistant", "content": "A data collection tool."},
    ]
    interpretation = MagicMock(interpreted_question="Who made it?")

    messages = rag._build_prompt_from_history(interpretation, history)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1:3] == history


def test_count_tokens_sums_message_contents(rag):
    messages = [{"role": "system", "content": "a" * 40}, {"role": "user", "content": "b" * 80}]
    assert rag._count_tokens(messages) == 10 + 20
