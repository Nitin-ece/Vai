"""
ModelProvider protocol.

Any ML model used by a stage must be wrapped in a class that
satisfies this protocol.  Implementations are registered in the
ModelRegistry and resolved by name at runtime via config.yaml.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelProvider(Protocol):
    """Protocol that every model wrapper must satisfy."""

    @property
    def name(self) -> str:
        """Unique model identifier (matches config.yaml key)."""
        ...

    def load(self, device: str = "cpu", weights: str | None = None) -> None:
        """Load model weights onto *device*.  Idempotent."""
        ...

    def preprocess(self, image: np.ndarray) -> Any:
        """Convert a BGR/RGB numpy image into the model's input format."""
        ...

    def predict(self, preprocessed: Any) -> Any:
        """Run forward pass.  Returns raw model output."""
        ...

    def postprocess(self, raw_output: Any) -> Any:
        """Convert raw model output into a stage-friendly structure."""
        ...


class BaseModelProvider:
    """
    Convenience base class (not required — the Protocol is enough).

    Provides default no-op implementations so concrete providers only
    override what they need.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._model: Any = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return self._name

    def load(self, device: str = "cpu", weights: str | None = None) -> None:
        self._device = device

    def preprocess(self, image: np.ndarray) -> Any:
        return image

    def predict(self, preprocessed: Any) -> Any:
        return preprocessed

    def postprocess(self, raw_output: Any) -> Any:
        return raw_output
