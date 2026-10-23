"""OpenAI-compatible adapter: OpenAI, DeepSeek, Zhipu GLM, DashScope Qwen,
SiliconFlow, and locally served vLLM/sglang/LM Studio (all expose /v1/chat/completions)."""
from __future__ import annotations

import time

from openai import AsyncOpenAI

from ..images import data_url
from .base import BaseVLModel, ModelConfig, ModelResponse


class OpenAICompatModel(BaseVLModel):
    provider_name = "openai"

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.client = AsyncOpenAI(
            api_key=cfg.resolve_api_key() or "EMPTY",
            base_url=cfg.resolve_base_url(),
            timeout=cfg.timeout,
            max_retries=0,  # retries handled by the runner
        )

    async def chat(self, image_path: str, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        content: list[dict] = []
        if image_path:
            content.append({"type": "image_url", "image_url": {"url": data_url(image_path)}})
        content.append({"type": "text", "text": prompt})
        resp = await self.client.chat.completions.create(
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        latency = int((time.perf_counter() - start) * 1000)
        choice = resp.choices[0] if resp.choices else None
        text = ((choice.message.content or "").strip()) if choice else ""
        u = resp.usage
        tokens = getattr(u, "total_tokens", None) if u else None
        in_tok = getattr(u, "prompt_tokens", None) if u else None
        out_tok = getattr(u, "completion_tokens", None) if u else None
        finish = getattr(choice, "finish_reason", None) if choice else None
        return ModelResponse(
            text=text, latency_ms=latency, tokens=tokens,
            input_tokens=in_tok, output_tokens=out_tok, finish_reason=finish,
        )
