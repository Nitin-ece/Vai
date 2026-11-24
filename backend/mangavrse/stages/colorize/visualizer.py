"""
Visualizer: renders a side-by-side Before/After comparison image
between the grayscale input panel and the palette-colorized vector result.
"""
from __future__ import annotations

import cv2
import numpy as np


def create_colorize_comparison(
    grayscale_bgr: np.ndarray,
    colored_bgr: np.ndarray,
) -> np.ndarray:
    """
    Renders a side-by-side Before / After comparison image:
    Left: Enhanced Grayscale Panel
    Right: Vector Palette Colorized Scene
    """
    h, w = grayscale_bgr.shape[:2]

    # Resize colored if needed
    if colored_bgr.shape[:2] != (h, w):
        colored_bgr = cv2.resize(colored_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    # Top banner
    banner = np.full((36, w * 2, 3), 25, dtype=np.uint8)
    cv2.putText(banner, "BEFORE (Grayscale)", (w // 4 - 60, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(banner, "AFTER (Vector Palette Colorized)", (w + w // 4 - 100, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 240, 160), 1, cv2.LINE_AA)

    # Side-by-side image
    side_by_side = np.hstack([grayscale_bgr, colored_bgr])
    combined = np.vstack([banner, side_by_side])
    return combined
