"""
Post-processing modules for the Enhance stage:
- Line sharpening & ink-edge refinement
- Halo suppression
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_line_sharpening(
    image_bgr: np.ndarray,
    amount: float = 0.7,
    radius: float = 0.8,
) -> np.ndarray:
    """
    Sharpens dark ink linework while suppressing white overshoot halos.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr

    # Gaussian blur for high-pass extraction
    ksize = int(np.ceil(radius * 3)) * 2 + 1
    blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), radius)

    diff = image_bgr.astype(np.float32) - blurred.astype(np.float32)

    # Invert edge mask so sharpening targets darker ink pixels predominantly
    ink_weight = (255.0 - gray.astype(np.float32)) / 255.0
    if len(image_bgr.shape) == 3:
        ink_weight = ink_weight[:, :, np.newaxis]

    # Weighted line sharpening: dark ink lines get full sharpening; paper gets suppressed
    sharpened = image_bgr.astype(np.float32) + (diff * amount * (0.4 + 0.6 * ink_weight))
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)
