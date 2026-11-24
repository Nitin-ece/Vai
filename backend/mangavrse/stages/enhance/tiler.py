"""
Seamless Tiler: splits large images into overlapping tiles to prevent OOM
and blends them using continuous cosine window feathering to eliminate seams.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


class SeamlessTiler:
    """
    Tiling orchestrator for neural upscaling and image processing.
    Eliminates tile boundaries via smooth 2D cosine-tapered alpha blending.
    """

    def __init__(self, tile_size: int = 512, tile_pad: int = 32) -> None:
        self.tile_size = tile_size
        self.tile_pad = tile_pad

    def process(
        self,
        image_bgr: np.ndarray,
        process_fn: Callable[[np.ndarray], np.ndarray],
        scale: int = 4,
    ) -> np.ndarray:
        """
        Processes *image_bgr* through *process_fn* tile-by-tile if dimensions exceed *tile_size*.
        Returns the seamlessly reconstructed upscaled image.
        """
        h, w = image_bgr.shape[:2]

        # If image fits in a single tile without exceeding tile_size
        if h <= self.tile_size and w <= self.tile_size:
            return process_fn(image_bgr)

        tile_step = self.tile_size - (2 * self.tile_pad)
        if tile_step <= 0:
            tile_step = self.tile_size // 2

        out_h = h * scale
        out_w = w * scale
        channels = image_bgr.shape[2] if len(image_bgr.shape) == 3 else 1

        out_canvas = np.zeros((out_h, out_w, channels), dtype=np.float32)
        weight_canvas = np.zeros((out_h, out_w, 1), dtype=np.float32)

        # Iterate across rows and columns
        y_starts = list(range(0, h, tile_step))
        x_starts = list(range(0, w, tile_step))

        for y_start in y_starts:
            for x_start in x_starts:
                # Calculate padded tile bounds on input
                y0 = max(0, y_start - self.tile_pad)
                x0 = max(0, x_start - self.tile_pad)
                y1 = min(h, y_start + tile_step + self.tile_pad)
                x1 = min(w, x_start + tile_step + self.tile_pad)

                input_tile = image_bgr[y0:y1, x0:x1]

                # Run process_fn on this tile
                output_tile = process_fn(input_tile).astype(np.float32)
                th, tw = output_tile.shape[:2]

                # Compute 2D cosine blend weight mask for this tile
                weight_tile = self._compute_tile_weight(
                    tile_h=th,
                    tile_w=tw,
                    pad_top=(y_start - y0) * scale,
                    pad_bottom=(y1 - (y_start + tile_step)) * scale,
                    pad_left=(x_start - x0) * scale,
                    pad_right=(x1 - (x_start + tile_step)) * scale,
                )

                # Destination coordinates on output canvas
                out_y0 = y0 * scale
                out_x0 = x0 * scale
                out_y1 = out_y0 + th
                out_x1 = out_x0 + tw

                out_canvas[out_y0:out_y1, out_x0:out_x1] += output_tile * weight_tile
                weight_canvas[out_y0:out_y1, out_x0:out_x1] += weight_tile

        # Normalize by accumulated weights
        valid_mask = weight_canvas > 1e-6
        out_canvas[valid_mask[:, :, 0]] /= weight_canvas[valid_mask[:, :, 0]]
        return np.clip(np.round(out_canvas), 0, 255).astype(np.uint8)

    def _compute_tile_weight(
        self,
        tile_h: int,
        tile_w: int,
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
    ) -> np.ndarray:
        """Constructs a smooth 2D cosine taper weighting mask across padding margins."""
        # 1D weights along Y
        wy = np.ones(tile_h, dtype=np.float32)
        if pad_top > 0:
            wy[:pad_top] = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, pad_top)))
        if pad_bottom > 0:
            wy[-pad_bottom:] = 0.5 * (1.0 + np.cos(np.linspace(0, math.pi, pad_bottom)))

        # 1D weights along X
        wx = np.ones(tile_w, dtype=np.float32)
        if pad_left > 0:
            wx[:pad_left] = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, pad_left)))
        if pad_right > 0:
            wx[-pad_right:] = 0.5 * (1.0 + np.cos(np.linspace(0, math.pi, pad_right)))

        weight_2d = np.outer(wy, wx)[:, :, np.newaxis]
        return weight_2d
