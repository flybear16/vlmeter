"""Anthropic Messages API adapter (Claude models), called via httpx."""
from __future__ import annotations

import time

import httpx

from ..images import b64_image
from .base import BaseVLModel, ModelResponse


class AnthropicModel(BaseVLModel):
    provider_name = "anthropic"

    async def chat(self, image_path: str | None, prompt: str) -> ModelResponse:
        blocks: list[dict] = []
        if image_path:
            b64, mime = b64_image(image_path)
            blocks.append(
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}
            )
        blocks.append({"type": "text", "text": prompt})
        base = (self.cfg.resolve_base_url() or "https://api.anthropic.com").rstrip("/")
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            resp = await client.post(
                f"{base}/v1/messages",
                headers={
                    "x-api-key": self.cfg.resolve_api_key() or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.cfg.model,
                    "max_tokens": self.cfg.max_tokens,
                    "temperature": self.cfg.temperature,
                    "messages": [{"role": "user", "content": blocks}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
        latency = int((time.perf_counter() - start) * 1000)
        text = "".join(b.get("text", "") for b in data.get("content", [])).strip()
        usage = data.get("usage") or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        tokens = (in_tok or 0) + (out_tok or 0) or None
        return ModelResponse(
            text=text, latency_ms=latency, tokens=tokens,
            input_tokens=in_tok, output_tokens=out_tok,
        )
