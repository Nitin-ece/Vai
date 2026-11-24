"""
Generates side-by-side or split before/after comparison images with titles.
"""
from __future__ import annotations

import cv2
import numpy as np


def generate_comparison_image(
    original_bgr: np.ndarray,
    enhanced_bgr: np.ndarray,
    title_original: str = "ORIGINAL",
    title_enhanced: str = "ENHANCED (4x)",
) -> np.ndarray:
    """
    Produces a clean side-by-side comparison image at equalized resolution.
    Upscales the original with nearest-neighbor to show pixel-level improvement clearly.
    """
    eh, ew = enhanced_bgr.shape[:2]
    oh, ow = original_bgr.shape[:2]

    # Resize original to match enhanced height & width for side-by-side
    orig_scaled = cv2.resize(original_bgr, (ew, eh), interpolation=cv2.INTER_NEAREST)

    # Header banner height
    banner_h = 70
    divider_w = 8

    comp_h = eh + banner_h
    comp_w = (ew * 2) + divider_w
    comparison = np.full((comp_h, comp_w, 3), 30, dtype=np.uint8)

    # Place original on left, enhanced on right
    comparison[banner_h:, :ew] = orig_scaled
    # Divider in the middle
    comparison[:, ew : ew + divider_w] = (80, 80, 80)
    comparison[banner_h:, ew + divider_w :] = enhanced_bgr

    # Draw header text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        comparison,
        f"{title_original} ({ow}x{oh})",
        (40, 48),
        font,
        1.1,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        comparison,
        f"{title_enhanced} ({ew}x{eh})",
        (ew + divider_w + 40, 48),
        font,
        1.1,
        (0, 255, 180),
        2,
        cv2.LINE_AA,
    )

    return comparison
