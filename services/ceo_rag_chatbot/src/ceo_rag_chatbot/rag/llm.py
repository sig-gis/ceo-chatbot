
"""
 #! src/ceo_chatbot/rag/llm.py
 summary:
    LLM provider abstraction for local and API-based models.

 details:
    This module provides a unified interface for LLM backends. All models,
    local (Ollama) or hosted (Gemini, DeepSeek, ...), are reached through
    their OpenAI-compatible chat API.
    It handles initialization, text generation, and streaming.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Iterable, List
from openai import OpenAI

from ceo_rag_chatbot.config import ChatbotConfig



class LLMProvider(ABC):
    """Base class for LLM providers."""
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        pass


class OpenAIProvider(LLMProvider):
    """Provider for any server that speaks the OpenAI chat API (Gemini, Ollama, DeepSeek, ...).

    `models` is the primary model followed by fallbacks. Each call tries them
    in order and re-raises the last error if all of them fail.
    """
    def __init__(self, base_url: str, api_key: str, models: List[str], context_window: int,
                 max_output_tokens: int = 8192, temperature: float = 0.2, timeout: float = 60):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.models = models
        self._max_context = context_window
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def _inference_with_fallback(self, prompt: str, stream: bool, **kwargs):
        """Run one request per model in turn until one succeeds.

        For streaming, the whole answer is collected before returning, so a model
        that fails halfway falls back cleanly instead of yielding a partial answer.
        Unknown kwargs (e.g. Gemini's response_mime_type) are ignored on purpose.
        """
        last_exception = None
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_output_tokens", self.max_output_tokens),
                    temperature=kwargs.get("temperature", self.temperature),
                    stream=stream,
                )
                if not stream:
                    return response.choices[0].message.content
                # Return the text pieces of the streamed response, skipping empty chunks
                return [c.choices[0].delta.content for c in response
                        if c.choices and c.choices[0].delta.content]
            except Exception as e:
                logging.warning(f"Model {model} failed: {e}")
                last_exception = e
        raise last_exception

    def generate(self, prompt: str, **kwargs) -> str:
        return self._inference_with_fallback(prompt, stream=False, **kwargs)

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        yield from self._inference_with_fallback(prompt, stream=True, **kwargs)

    def count_tokens(self, text: str) -> int:
        # Rough estimate (~4 characters per token); there is no tokenizer for remote models.
        return len(text) // 4

    @property
    def max_context_tokens(self) -> int:
        return self._max_context


def get_reader_llm(config: ChatbotConfig) -> LLMProvider:
    """
    Build an OpenAIProvider for the active model in rag_config.yml.

    The API key is read from the env var named by `api_key_env`. Entries
    without `api_key_env` (e.g. Ollama) get a placeholder key, because the
    OpenAI client requires a value even when the server ignores it.
    """
    llm = config.llm
    if llm.api_key_env:
        # Read os.environ directly: pydantic reading .env doesn't populate os.environ,
        # so the key must come from the real environment (docker --env-file, Cloud Run, etc.).
        api_key = os.environ.get(llm.api_key_env)
        if not api_key:
            raise ValueError(
                f"{llm.api_key_env} is not set, but model '{config.active_model}' needs it"
            )
    else:
        api_key = "ollama"

    return OpenAIProvider(
        base_url=llm.base_url,
        api_key=api_key,
        models=[llm.model, *llm.fallback],
        context_window=llm.context_window,
        max_output_tokens=llm.max_output_tokens,
        temperature=llm.temperature,
        timeout=llm.timeout,
    )