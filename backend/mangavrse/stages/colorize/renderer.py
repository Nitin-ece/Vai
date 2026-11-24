"""
Renderer: composites semantic color fills, screentone shading, and black line art
into a high-resolution raster image (colored_panel.png).
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def hex_to_bgr(hex_code: str) -> tuple[int, int, int]:
    """Converts a hex color (#RRGGBB) to OpenCV BGR tuple."""
    hex_clean = hex_code.lstrip("#")
    if len(hex_clean) != 6:
        return (200, 200, 200)
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return (b, g, r)


def render_colored_panel(
    width: int,
    height: int,
    layers: list[dict[str, Any]],
    layer_colors: dict[str, str],
    masks_dict: dict[str, np.ndarray],
    background_plate: np.ndarray | None = None,
    line_art: np.ndarray | None = None,
    screentone: np.ndarray | None = None,
) -> np.ndarray:
    """
    Renders the complete colored panel image:
    1. Base: background plate or neutral canvas.
    2. Color Fills: paint each layer mask with its assigned palette color in suggested_z_order.
    3. Screentone: multiply halftone dot patterns on top to shade the colors.
    4. Line Art: multiply pure black ink linework on top to ensure zero lines are occluded.
    """
    if background_plate is not None and background_plate.shape[:2] == (height, width):
        canvas = background_plate.copy()
    else:
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    # Sort layers by suggested_z_order
    sorted_layers = sorted(layers, key=lambda l: l.get("suggested_z_order", 0))

    color_canvas = canvas.copy()

    for lyr in sorted_layers:
        lid = lyr["id"]
        color_hex = layer_colors.get(lid, "#E0E0E0")
        bgr_color = hex_to_bgr(color_hex)

        mask = masks_dict.get(lid)
        if mask is None or np.count_nonzero(mask > 127) == 0:
            continue

        mask_bool = mask > 127
        color_canvas[mask_bool] = bgr_color

    # Modulate with screentone shading if present (screentone dots act as darker shading)
    if screentone is not None and screentone.shape[:2] == (height, width):
        st_gray = cv2.cvtColor(screentone, cv2.COLOR_BGR2GRAY) if len(screentone.shape) == 3 else screentone
        tone_dots = st_gray < 220
        # Darken color slightly under screentone dots
        color_canvas[tone_dots] = (color_canvas[tone_dots].astype(np.float32) * 0.70).astype(np.uint8)

    # Multiply black ink line art on the very top (lines are always 100% black)
    if line_art is not None and line_art.shape[:2] == (height, width):
        line_gray = cv2.cvtColor(line_art, cv2.COLOR_BGR2GRAY) if len(line_art.shape) == 3 else line_art
        ink_bool = line_gray < 150
        color_canvas[ink_bool] = (15, 15, 15)

    return color_canvas
