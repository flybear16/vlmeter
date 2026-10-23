"""Model registry: load the packaged config/models.yaml and build adapter instances."""
from __future__ import annotations

from pathlib import Path

import yaml

from .base import BaseVLModel, ModelConfig
from .mock import MockModel

PROVIDERS: dict[str, type[BaseVLModel]] = {}


def _register(cls: type[BaseVLModel]) -> type[BaseVLModel]:
    PROVIDERS[cls.provider_name] = cls
    return cls


def _build_providers() -> None:
    from .anthropic_client import AnthropicModel
    from .gemini_client import GeminiModel
    from .ollama_client import OllamaModel
    from .openai_client import OpenAICompatModel

    for cls in (OpenAICompatModel, AnthropicModel, GeminiModel, OllamaModel, MockModel):
        if cls.provider_name not in PROVIDERS:
            PROVIDERS[cls.provider_name] = cls


def load_registry(path: str | Path | None = None) -> dict[str, ModelConfig]:
    _build_providers()
    if path is None:
        # Packaged default registry — works both in the source tree and after
        # `pip install` (site-packages); override with --registry for your own.
        path = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model registry not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    registry: dict[str, ModelConfig] = {}
    for name, spec in raw.get("models", {}).items():
        if spec is None:
            continue
        if spec.get("enabled") is False:
            continue
        spec = {k: v for k, v in spec.items() if k != "enabled"}
        provider = spec.get("provider")
        if provider not in PROVIDERS:
            raise ValueError(
                f"model '{name}': unknown provider '{provider}', "
                f"expected one of {sorted(PROVIDERS)}"
            )
        cfg = ModelConfig(name=name, **spec)
        registry[name] = cfg
    return registry


def build_model(name: str, registry: dict[str, ModelConfig]) -> BaseVLModel:
    _build_providers()
    if name not in registry:
        raise KeyError(f"model '{name}' not in registry; available: {sorted(registry)}")
    cfg = registry[name]
    return PROVIDERS[cfg.provider](cfg)
