"""Adapter smoke tests against canned local HTTP endpoints.

Validates request shape and response parsing for the anthropic / gemini /
ollama adapters without needing real keys or a GPU. Run:
python tests/test_adapters.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmeter.models.anthropic_client import AnthropicModel
from vlmeter.models.base import ModelConfig
from vlmeter.models.gemini_client import GeminiModel
from vlmeter.models.ollama_client import OllamaModel

RECEIVED: dict[str, dict] = {}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        route = (
            "anthropic" if self.path.startswith("/v1/messages")
            else "gemini" if ":generateContent" in self.path
            else "ollama"
        )
        RECEIVED[route] = {"path": self.path, "body": body, "headers": dict(self.headers)}
        if route == "anthropic":
            payload = {"content": [{"type": "text", "text": "A"}],
                       "usage": {"input_tokens": 10, "output_tokens": 1}}
        elif route == "gemini":
            payload = {"candidates": [{"content": {"parts": [{"text": "B"}]}}],
                       "usageMetadata": {"totalTokenCount": 11}}
        else:
            payload = {"message": {"content": "C"}, "eval_count": 9}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # silence
        pass


IMG = Path(__file__).resolve().parents[1] / "data" / "images" / "cls_shape_00.png"


def main() -> None:
    if not IMG.exists():
        sys.exit("run scripts/make_sample_data.py first")
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    cases = [
        (AnthropicModel, "anthropic"),
        (GeminiModel, "gemini"),
        (OllamaModel, "ollama"),
    ]
    for cls, route in cases:
        cfg = ModelConfig(name=f"t-{route}", provider=route, model="m",
                          base_url=f"http://127.0.0.1:{port}")
        resp = asyncio.run(cls(cfg).chat(str(IMG), "test"))
        assert resp.text in ("A", "B", "C"), f"{route}: bad text {resp.text!r}"
        body = RECEIVED[route]["body"]
        if route == "anthropic":
            assert body["messages"][0]["content"][0]["type"] == "image"
            assert body["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
            base64.b64decode(body["messages"][0]["content"][0]["source"]["data"])
        elif route == "gemini":
            part = body["contents"][0]["parts"][0]["inline_data"]
            base64.b64decode(part["data"])
            assert part["mime_type"] == "image/png"
        else:
            base64.b64decode(body["messages"][0]["images"][0])
            assert body["stream"] is False
        print(f"{route} adapter OK (latency {resp.latency_ms}ms, tokens {resp.tokens})")

    # text-only call (judge path) must not include any image block
    cfg = ModelConfig(name="t-an2", provider="anthropic", model="m",
                      base_url=f"http://127.0.0.1:{port}")
    asyncio.run(AnthropicModel(cfg).chat(None, "judge"))
    assert all(b["type"] == "text"
               for b in RECEIVED["anthropic"]["body"]["messages"][0]["content"])
    print("text-only (judge) call OK")
    srv.shutdown()
    print("all adapter tests passed")


if __name__ == "__main__":
    main()
