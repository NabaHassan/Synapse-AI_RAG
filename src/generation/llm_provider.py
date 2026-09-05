"""Main LLM provider abstraction.

The first provider wraps the existing local LLM generator. Future local HTTP,
OpenAI-compatible, or custom providers should implement the same interface.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMProviderConfig:
    provider: str = "local_vllm"
    model_name: str = ""
    endpoint_url: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: float = 60.0
    max_concurrency: int = 2
    supports_streaming: bool = True
    supports_json_mode: bool = False
    supports_citations: bool = False
    supports_logprobs: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model_name: str
    latency_seconds: float
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    structured: Optional[Dict[str, Any]] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMProvider:
    """Interface for main LLM generation."""

    config: LLMProviderConfig

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError


class LocalLLMGeneratorProvider(LLMProvider):
    """Provider adapter for the existing `LLMGenerator` implementation."""

    def __init__(self, generator: Any, config: Optional[LLMProviderConfig] = None):
        self.generator = generator
        generator_config = getattr(generator, "config", None)
        model_name = getattr(generator_config, "model_name", "") or ""
        self.config = config or LLMProviderConfig(
            provider="local_vllm",
            model_name=model_name,
            timeout_seconds=float(getattr(generator_config, "timeout", 60.0) or 60.0),
            max_concurrency=int(getattr(generator_config, "max_concurrency", 2) or 2),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
        purpose: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        started = time.time()
        text = self.generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop,
            purpose=purpose,
            **kwargs,
        )
        latency = time.time() - started
        return LLMResponse(
            text=text,
            provider=self.config.provider,
            model_name=self.config.model_name,
            latency_seconds=latency,
            usage={
                "approx_output_words": len((text or "").split()),
                "prompt_chars": len(prompt or ""),
            },
            raw_metadata=metadata or {},
        )

