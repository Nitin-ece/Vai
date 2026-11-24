"""
ModelRegistry — maps config.yaml model names to ModelProvider classes.

Usage
-----
    registry = ModelRegistry()
    registry.register("yolov8", YoloV8Provider)

    provider = registry.get("yolov8")          # instantiate
    provider.load(device="cuda", weights=...)  # load weights
"""
from __future__ import annotations

from typing import Any

from .model_provider import ModelProvider


class ModelRegistry:
    """Global registry of model-provider classes, keyed by name."""

    def __init__(self) -> None:
        self._providers: dict[str, type] = {}

    def register(self, name: str, provider_cls: type) -> None:
        """Register a ModelProvider class under *name*."""
        if not isinstance(provider_cls, type):
            raise TypeError(f"Expected a class, got {type(provider_cls).__name__}")
        self._providers[name] = provider_cls

    def get(self, name: str, **kwargs: Any) -> ModelProvider:
        """Instantiate and return a provider by name."""
        if name not in self._providers:
            available = ", ".join(sorted(self._providers)) or "(none)"
            raise KeyError(
                f"Model provider '{name}' not registered. Available: {available}"
            )
        instance = self._providers[name](name=name, **kwargs)
        if not isinstance(instance, ModelProvider):
            raise TypeError(
                f"Class '{self._providers[name].__name__}' does not satisfy the ModelProvider protocol"
            )
        return instance

    def list(self) -> list[str]:
        return sorted(self._providers.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._providers
