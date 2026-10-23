"""Ollama native /api/chat adapter (llava, qwen2.5vl, minicpm-v, ...)."""
from __future__ import annotations

import time

import httpx

from ..images import b64_image
from .base import BaseVLModel, ModelResponse


class OllamaModel(BaseVLModel):
    provider_name = "ollama"

    async def chat(self, image_path: str | None, prompt: str) -> ModelResponse:
        images: list[str] = []
        if image_path:
            b64, _mime = b64_image(image_path)
            images.append(b64)
        base = (self.cfg.resolve_base_url() or "http://localhost:11434").rstrip("/")
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            resp = await client.post(
                f"{base}/api/chat",
                json={
                    "model": self.cfg.model,
                    "stream": False,
                    "options": {"temperature": self.cfg.temperature},
                    "messages": [{"role": "user", "content": prompt, "images": images}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        latency = int((time.perf_counter() - start) * 1000)
        text = ((data.get("message") or {}).get("content") or "").strip()
        in_tok = data.get("prompt_eval_count")
        out_tok = data.get("eval_count")
        tokens = out_tok
        return ModelResponse(
            text=text, latency_ms=latency, tokens=tokens,
            input_tokens=in_tok, output_tokens=out_tok,
        )
