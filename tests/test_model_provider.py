"""Tests for ModelProvider and ModelRegistry."""
from __future__ import annotations

import numpy as np
import pytest

from mangavrse.core.model_provider import BaseModelProvider, ModelProvider
from mangavrse.core.registry import ModelRegistry


class DummyModel(BaseModelProvider):
    def __init__(self, name: str = "dummy", scale: float = 1.0):
        super().__init__(name)
        self.scale = scale

    def predict(self, preprocessed: np.ndarray) -> np.ndarray:
        return preprocessed * self.scale


def test_model_provider_protocol():
    model = DummyModel()
    assert isinstance(model, ModelProvider)
    assert model.name == "dummy"
    model.load(device="cpu")
    arr = np.array([1.0, 2.0, 3.0])
    pre = model.preprocess(arr)
    pred = model.predict(pre)
    post = model.postprocess(pred)
    np.testing.assert_array_equal(post, arr)


def test_model_registry():
    registry = ModelRegistry()
    registry.register("dummy", DummyModel)
    assert "dummy" in registry
    assert "dummy" in registry.list()

    instance = registry.get("dummy", scale=2.0)
    assert isinstance(instance, DummyModel)
    assert instance.scale == 2.0


def test_model_registry_missing():
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.get("nonexistent")
