"""Model adapter interface and shared call-result type."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    text: str
    latency_ms: int = 0
    tokens: int | None = None              # total (fallback when no split)
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    raw: Any = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


@dataclass
class ModelConfig:
    name: str                       # registry key / display name
    provider: str                   # openai | anthropic | gemini | ollama | mock
    model: str                      # provider-side model id
    base_url: str | None = None
    base_url_env: str | None = None  # read base_url from this env var (overrides base_url)
    api_key_env: str | None = None
    api_key_default: str = "EMPTY"   # used when the env var is unset (vLLM ignores keys)
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 120.0
    concurrency: int = 4
    price_input: float | None = None   # 元 / 每百万输入 tokens（按官网价核对）
    price_output: float | None = None  # 元 / 每百万输出 tokens
    extra: dict[str, Any] = field(default_factory=dict)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float | None:
        """估算单次调用成本（元）；未配置单价返回 None。"""
        if self.price_input is None or self.price_output is None:
            return None
        return (input_tokens * self.price_input + output_tokens * self.price_output) / 1_000_000

    def resolve_base_url(self) -> str | None:
        if self.base_url_env:
            return os.environ.get(self.base_url_env) or self.base_url
        return self.base_url

    def resolve_api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or self.api_key_default
        return self.api_key_default


class BaseVLModel:
    """A vision-language chat model. Implementations must be safe to share
    across asyncio tasks (create per-call clients or reuse an httpx client)."""

    provider_name: str = "base"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    async def chat(self, image_path: str | None, prompt: str) -> ModelResponse:
        raise NotImplementedError
