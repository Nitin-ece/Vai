"""
InpainterProvider: wraps background inpainting behind the ModelProvider protocol.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mangavrse.core.model_provider import BaseModelProvider
from .manga_inpaint import inpaint_manga_background

logger = logging.getLogger(__name__)


class InpainterProvider(BaseModelProvider):
    """
    ModelProvider wrapper for background inpainting.
    Supports manga line extension, screentone texture synthesis, and Telea / Navier-Stokes engines.
    """

    def __init__(
        self,
        name: str = "manga_telea",
        method: str = "telea_texture",
        inpaint_radius: int = 5,
        line_extension: bool = True,
        screentone_synthesis: bool = True,
    ) -> None:
        super().__init__(name)
        self.method = method
        self.inpaint_radius = inpaint_radius
        self.line_extension = line_extension
        self.screentone_synthesis = screentone_synthesis

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def predict(self, preprocessed: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Expects a tuple of (image_bgr, occlusion_mask).
        Returns the filled background BGR image.
        """
        image_bgr, occlusion_mask = preprocessed
        return inpaint_manga_background(
            image_bgr=image_bgr,
            occlusion_mask=occlusion_mask,
            inpaint_radius=self.inpaint_radius,
            method=self.method,
            line_extension=self.line_extension,
            screentone_synthesis=self.screentone_synthesis,
        )

    def postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        return raw_output
