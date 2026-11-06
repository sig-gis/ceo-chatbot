
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
    """Base class for LLM providers. Every provider must implement all four members."""
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Send a prompt to the model and return the full answer.

        Args:
            prompt (str): The complete prompt text to send.
            **kwargs: Optional per-call overrides, such as max_output_tokens or temperature.

        Returns:
            str: The model's answer.
        """
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Send a prompt to the model and return the answer in pieces.

        Args:
            prompt (str): The complete prompt text to send.
            **kwargs: Optional per-call overrides, such as max_output_tokens or temperature.

        Returns:
            Iterable[str]: Text pieces that join together into the full answer.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count (or estimate) how many tokens the model would see in a text.

        Used to trim conversation history so the prompt fits the model's context window.

        Args:
            text (str): The text to measure.

        Returns:
            int: Number of tokens.
        """
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """
        The largest number of tokens the model accepts in one request (prompt plus answer).

        Returns:
            int: The model's context window size.
        """
        pass


class OpenAIProvider(LLMProvider):
    """Provider for any server that speaks the OpenAI chat API (Gemini, Ollama, DeepSeek, ...).

    `models` is the primary model followed by fallbacks. Each call tries them
    in order and re-raises the last error if all of them fail.
    """
    def __init__(self, base_url: str, api_key: str, models: List[str], context_window: int,
                 max_output_tokens: int = 8192, temperature: float = 0.2, timeout: float = 60):
        """
        Create an OpenAI SDK client pointed at the given server.

        No network call is made here; the first request happens on generate() or stream().

        Args:
            base_url (str): The server's OpenAI-compatible address, e.g. http://inference:11434/v1.
            api_key (str): API key for the server. Keyless servers (Ollama) accept any value.
            models (List[str]): Model names to try, primary first, then fallbacks.
            context_window (int): Largest prompt plus answer size the model accepts, in tokens.
            max_output_tokens (int): Default cap on answer length, in tokens.
            temperature (float): Default randomness of answers (lower is more predictable).
            timeout (float): Seconds to wait for a response before giving up.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.models = models
        self._max_context = context_window
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def _inference_with_fallback(self, prompt: str, stream: bool, **kwargs):
        """
        Send the prompt to each model in turn until one succeeds.

        details:
            The prompt is sent as a single user message. For streaming, the whole
            answer is collected before returning, so a model that fails halfway
            falls back cleanly instead of returning a partial answer.
            Unknown kwargs (e.g. Gemini's response_mime_type) are ignored on purpose.

        Args:
            prompt (str): The complete prompt text to send.
            stream (bool): True to use the streaming API, False for a single response.
            **kwargs: Optional overrides: max_output_tokens, temperature.

        Returns:
            str | List[str]: The full answer when stream is False, or the list of
            answer pieces when stream is True.

        Raises:
            Exception: The last model's error, if every model fails.
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
        """
        Send a prompt and return the full answer, trying fallback models on failure.

        Args:
            prompt (str): The complete prompt text to send.
            **kwargs: Optional overrides: max_output_tokens, temperature.

        Returns:
            str: The model's answer.

        Raises:
            Exception: The last model's error, if every model fails.
        """
        return self._inference_with_fallback(prompt, stream=False, **kwargs)

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        """
        Send a prompt and yield the answer in pieces, trying fallback models on failure.

        The whole answer is received before the first piece is yielded (see
        _inference_with_fallback), so callers never get a partial answer.

        Args:
            prompt (str): The complete prompt text to send.
            **kwargs: Optional overrides: max_output_tokens, temperature.

        Yields:
            str: Pieces of the answer, in order.

        Raises:
            Exception: The last model's error, if every model fails.
        """
        yield from self._inference_with_fallback(prompt, stream=True, **kwargs)

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        There is no tokenizer for models behind an API, so this uses the common
        rule of thumb of about 4 characters per token. Keep some headroom in
        context_window because the estimate can be off.

        Args:
            text (str): The text to measure.

        Returns:
            int: Estimated number of tokens.
        """
        return len(text) // 4

    @property
    def max_context_tokens(self) -> int:
        """
        The model's context window, as set in rag_config.yml.

        Returns:
            int: Largest prompt plus answer size the model accepts, in tokens.
        """
        return self._max_context


def get_reader_llm(config: ChatbotConfig) -> LLMProvider:
    """
    Build an OpenAIProvider for the active model in rag_config.yml.

    details:
        The API key is read from the env var named by `api_key_env`. Entries
        without `api_key_env` (e.g. Ollama) get a placeholder key, because the
        OpenAI client requires a value even when the server ignores it.
        The entry's fallback models are tried after its main model.

    Args:
        config (ChatbotConfig): Loaded chatbot config; its active_model entry is used.

    Returns:
        LLMProvider: A provider ready to send prompts to the active model.

    Raises:
        ValueError: If the active model needs an API key and its env var is not set.
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