"""Google Gemini generateContent adapter, called via httpx."""
from __future__ import annotations

import time

import httpx

from ..images import b64_image
from .base import BaseVLModel, ModelResponse


class GeminiModel(BaseVLModel):
    provider_name = "gemini"

    async def chat(self, image_path: str | None, prompt: str) -> ModelResponse:
        parts: list[dict] = []
        if image_path:
            b64, mime = b64_image(image_path)
            parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        parts.append({"text": prompt})
        base = (self.cfg.resolve_base_url() or "https://generativelanguage.googleapis.com").rstrip("/")
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            resp = await client.post(
                f"{base}/v1beta/models/{self.cfg.model}:generateContent",
                headers={
                    "x-goog-api-key": self.cfg.resolve_api_key() or "",
                    "content-type": "application/json",
                },
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "temperature": self.cfg.temperature,
                        "maxOutputTokens": self.cfg.max_tokens,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()
        latency = int((time.perf_counter() - start) * 1000)
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()
        usage = data.get("usageMetadata") or {}
        in_tok = usage.get("promptTokenCount")
        out_tok = usage.get("candidatesTokenCount")
        tokens = usage.get("totalTokenCount")
        return ModelResponse(
            text=text, latency_ms=latency, tokens=tokens,
            input_tokens=in_tok, output_tokens=out_tok,
        )
