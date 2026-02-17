"""
 #! src/ceo_chatbot/rag/llm.py
 summary:
    LLM provider abstraction for local and API-based models.

 details:
    This module provides a unified interface for different LLM backends,
    supporting Hugging Face local models and Google Gemini API.
    It handles initialization, text generation, and streaming.
"""

import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    PreTrainedTokenizerBase,
)
import google.genai as genai

from ceo_chatbot.config import AppSettings


class LLMProvider:
    """Base class for LLM providers."""
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    @property
    def max_context_tokens(self) -> int:
        raise NotImplementedError


class HuggingFaceProvider(LLMProvider):
    """Provider for Hugging Face local models."""
    def __init__(self, model_name: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
        
        # Try to get max position embeddings
        if hasattr(self.model.config, "max_position_embeddings"):
            self._max_context = self.model.config.max_position_embeddings
        else:
            self._max_context = 4096 # Fallback

    def generate(self, prompt: str, **kwargs) -> str:
        outputs = self.pipeline(prompt, **kwargs)
        return outputs[0]["generated_text"]

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        from threading import Thread
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_new_tokens", 500),
            do_sample=True,
            temperature=kwargs.get("temperature", 0.2),
            repetition_penalty=1.1,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def max_context_tokens(self) -> int:
        return self._max_context


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini API using the new google-genai SDK."""
    def __init__(self, model_name: str):
        # Instantiate AppSettings to load from .env or environment
        settings = AppSettings()
        api_key = settings.google_api_key
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or .env file")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        # Gemini 1.5 Flash/Pro have huge context, but we'll set a reasonable limit for RAG
        self._max_context = 1000000 if "gemini" in model_name else 32768

    def generate(self, prompt: str, **kwargs) -> str:
        config = {
            "temperature": kwargs.get("temperature", 0.2),
            "max_output_tokens": kwargs.get("max_new_tokens", 500),
        }
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text

    def stream(self, prompt: str, **kwargs) -> Iterable[str]:
        config = {
            "temperature": kwargs.get("temperature", 0.2),
            "max_output_tokens": kwargs.get("max_new_tokens", 500),
        }
        response = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        for chunk in response:
            yield chunk.text

    def count_tokens(self, text: str) -> int:
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=text
        )
        return response.total_tokens

    @property
    def max_context_tokens(self) -> int:
        return self._max_context


def get_reader_llm(
    model_name: str = "HuggingFaceH4/zephyr-7b-beta",
) -> Tuple[LLMProvider, Optional[PreTrainedTokenizerBase]]:
    """
    Factory function to get the appropriate LLM provider.
    """
    if "gemini" in model_name.lower():
        provider = GeminiProvider(model_name)
        return provider, None
    else:
        provider = HuggingFaceProvider(model_name)
        return provider, provider.tokenizer
