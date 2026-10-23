"""Offline mock model: deterministic hash-based answers parsed from the prompt.

Used for pipeline self-tests without spending API quota. It never sees ground
truth, so its accuracy is pseudo-random — good enough to exercise scoring,
resume and reporting end to end.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re

from .base import BaseVLModel, ModelResponse

_LETTER_LINE = re.compile(r"([A-H])[.、:：)\s]\s*(.+)")


class MockModel(BaseVLModel):
    provider_name = "mock"

    async def chat(self, image_path: str | None, prompt: str) -> ModelResponse:
        await asyncio.sleep(0.01)
        key = f"{image_path}|{prompt[:64]}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        if "检查项" in prompt and "JSON" in prompt:
            n = len(re.findall(r"^\s*\d+\.", prompt, re.M))
            verdict = {
                str(i + 1): {"verdict": "是" if (h >> i) & 1 else "否",
                             "reason": f"mock 依据 {i + 1}"}
                for i in range(n)
            }
            return ModelResponse(text=json.dumps(verdict, ensure_ascii=False), latency_ms=30, tokens=20)
        options = _LETTER_LINE.findall(prompt)
        if options:
            letter, _text = options[h % len(options)]
            return ModelResponse(text=letter, latency_ms=30, tokens=8)
        if "几个" in prompt or "多少" in prompt:  # counting-style question
            return ModelResponse(text=str(2 + h % 4), latency_ms=30, tokens=6)
        return ModelResponse(text="未知", latency_ms=30, tokens=4)
