from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ceo_rag_chatbot.config import ChatbotConfig, load_chatbot_config
from ceo_rag_chatbot.rag.llm import get_reader_llm

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "gemini": {
        "base_url": "https://gemini.example/openai/",
        "model": "gemini-2.5-flash",
        "api_key_env": "TEST_GEMINI_KEY",
        "context_window": 1000000,
        "fallback": ["gemini-2.0-flash"],
    },
    "local": {
        "base_url": "http://inference:11434/v1",
        "model": "llama3.1:8b-instruct-q4_K_M",
        "context_window": 131072,
        "probe": True,
    },
}


def _config(active_model):
    return ChatbotConfig(
        embedding_model_name="m", vectorstore_path="p", vectorstore_gcs="g",
        prompt_file="conf/base/prompts.yml", models=MODELS, active_model=active_model,
    )


def test_unknown_active_model_is_rejected():
    with pytest.raises(ValidationError, match="nope"):
        _config("nope")


def test_fallback_models_reuse_base_url_and_key(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret")
    with patch("ceo_rag_chatbot.rag.llm.OpenAI") as openai_cls:
        provider, _ = get_reader_llm(_config("gemini"))
    assert provider.models == ["gemini-2.5-flash", "gemini-2.0-flash"]
    kwargs = openai_cls.call_args.kwargs
    assert kwargs["base_url"] == "https://gemini.example/openai/"
    assert kwargs["api_key"] == "secret"


def test_missing_api_key_env_names_the_variable(monkeypatch):
    monkeypatch.delenv("TEST_GEMINI_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_GEMINI_KEY"):
        get_reader_llm(_config("gemini"))


def test_keyless_local_model_builds_without_a_key():
    with patch("ceo_rag_chatbot.rag.llm.OpenAI"):
        provider, _ = get_reader_llm(_config("local"))
    assert provider.models == ["llama3.1:8b-instruct-q4_K_M"]


def test_repo_rag_config_loads():
    config = load_chatbot_config(REPO_ROOT / "conf/base/rag_config.yml")
    assert config.active_model in config.models
