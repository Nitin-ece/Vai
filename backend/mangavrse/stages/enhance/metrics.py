"""
Metrics calculation for the Enhance stage:
- Resolution before and after
- Laplacian variance (focus / high frequency energy)
- Tenengrad gradient magnitude (line sharpness)
- Step timings in milliseconds
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def compute_sharpness_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """
    Computes objective sharpness metrics:
    - Laplacian variance: standard benchmark for edge contrast & crispness
    - Tenengrad gradient energy: Sobel gradient magnitude squared
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr

    # 1. Laplacian variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # 2. Tenengrad gradient energy
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx**2 + gy**2))

    return {
        "laplacian_variance": round(lap_var, 2),
        "tenengrad_energy": round(tenengrad, 2),
    }
