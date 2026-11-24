"""
Tracer: wraps Potrace behind the ModelProvider protocol to trace binary masks into vector curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import potrace

from mangavrse.core.model_provider import BaseModelProvider


@dataclass
class VectorSegment:
    is_corner: bool
    end_point: tuple[float, float]
    c: tuple[float, float] | None = None  # Corner point for CornerSegment
    c1: tuple[float, float] | None = None  # First control point for BezierSegment
    c2: tuple[float, float] | None = None  # Second control point for BezierSegment


@dataclass
class VectorPath:
    start_point: tuple[float, float]
    segments: list[VectorSegment] = field(default_factory=list)
    is_closed: bool = True
    area: float = 0.0
    bounding_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x0, y0, x1, y1


class PotraceVectorProvider(BaseModelProvider):
    """
    Traces binary raster images into vector paths with Bezier curves using Potrace.
    Conforms to the ModelProvider protocol.
    """

    def __init__(
        self,
        name: str = "potrace",
        turdsize: int = 2,
        alphamax: float = 1.0,
        opttolerance: float = 0.8,
    ) -> None:
        super().__init__(name)
        self.turdsize = turdsize
        self.alphamax = alphamax
        self.opttolerance = opttolerance

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Ensures boolean 2D numpy array where True is foreground ink."""
        if len(image.shape) == 3:
            gray = image[:, :, 0]
        else:
            gray = image
        return gray > 127

    def predict(self, preprocessed: np.ndarray) -> list[VectorPath]:
        """Runs Potrace tracing algorithm on boolean bitmap."""
        # Standardize foreground ink to boolean True
        if preprocessed.dtype != bool:
            fg_bool = preprocessed > 127
        else:
            fg_bool = preprocessed

        # Potrace Bitmap inverts its data during __init__ (self.invert()).
        # Therefore, passing ~fg_bool ensures that foreground ink is True in Bitmap.data.
        bitmap = potrace.Bitmap(~fg_bool)
        path_list = bitmap.trace(
            turdsize=self.turdsize,
            alphamax=self.alphamax,
            opttolerance=self.opttolerance,
        )

        vector_paths: list[VectorPath] = []

        for curve in path_list.curves:
            start_pt = (float(curve.start_point.x), float(curve.start_point.y))
            segments: list[VectorSegment] = []

            min_x, max_x = start_pt[0], start_pt[0]
            min_y, max_y = start_pt[1], start_pt[1]

            for seg in curve.segments:
                ep = (float(seg.end_point.x), float(seg.end_point.y))
                min_x = min(min_x, ep[0])
                max_x = max(max_x, ep[0])
                min_y = min(min_y, ep[1])
                max_y = max(max_y, ep[1])

                if seg.is_corner:
                    c_pt = (float(seg.c.x), float(seg.c.y))
                    min_x = min(min_x, c_pt[0])
                    max_x = max(max_x, c_pt[0])
                    min_y = min(min_y, c_pt[1])
                    max_y = max(max_y, c_pt[1])
                    segments.append(VectorSegment(is_corner=True, end_point=ep, c=c_pt))
                else:
                    c1 = (float(seg.c1.x), float(seg.c1.y))
                    c2 = (float(seg.c2.x), float(seg.c2.y))
                    min_x = min(min_x, c1[0], c2[0])
                    max_x = max(max_x, c1[0], c2[0])
                    min_y = min(min_y, c1[1], c2[1])
                    max_y = max(max_y, c1[1], c2[1])
                    segments.append(VectorSegment(is_corner=False, end_point=ep, c1=c1, c2=c2))

            bbox_area = max(0.0, (max_x - min_x) * (max_y - min_y))

            vector_paths.append(
                VectorPath(
                    start_point=start_pt,
                    segments=segments,
                    is_closed=True,
                    area=bbox_area,
                    bounding_box=(min_x, min_y, max_x, max_y),
                )
            )

        return vector_paths

    def postprocess(self, raw_output: list[VectorPath]) -> list[VectorPath]:
        return raw_output
