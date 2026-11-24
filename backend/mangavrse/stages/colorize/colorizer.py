"""
Colorizer Provider conforming to the ModelProvider protocol.
Combines conditional diffusion and character prior palettes for harmonious manga colorization.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np

from mangavrse.core.model_provider import BaseModelProvider
from mangavrse.models.conditional_diffusion import ConditionalDiffusionColorizerProvider
from .palette_manager import PaletteManager

logger = logging.getLogger(__name__)


class ColorizerProvider(BaseModelProvider):
    """
    ModelProvider for automatic manga colorization.
    Maps segmented layers to harmonious color palettes and enables cross-panel continuity,
    with conditional diffusion support.
    """

    def __init__(
        self,
        name: str = "conditional_diffusion_manga_colorizer",
        default_palette_id: str = "char_001",
    ) -> None:
        super().__init__(name)
        self.default_palette_id = default_palette_id
        self.palette_manager = PaletteManager()
        self.diffusion_provider = ConditionalDiffusionColorizerProvider()

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def predict(self, preprocessed: tuple[list[dict[str, Any]], PaletteManager | None]) -> dict[str, str]:
        """
        Takes a list of layer metadata and an optional PaletteManager.
        Returns a mapping of {layer_id: hex_color}.
        """
        layers, palette_mgr = preprocessed
        pm = palette_mgr or self.palette_manager
        layer_colors: dict[str, str] = {}

        for lyr in layers:
            lid = lyr["id"]
            label = lyr.get("label", "character")
            part = lyr.get("part")
            char_id = lyr.get("character_id", self.default_palette_id)

            # Resolve color from palette / conditional priors
            hex_color = pm.resolve_layer_color(label, part, character_id=char_id)
            layer_colors[lid] = hex_color

        return layer_colors

    def postprocess(self, raw_output: dict[str, str]) -> dict[str, str]:
        return raw_output
