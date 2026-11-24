"""
Quality: checks inpainting boundary seam disparity and texture consistency.
"""
from __future__ import annotations

import cv2
import numpy as np


def evaluate_inpaint_quality(
    original_bgr: np.ndarray,
    inpainted_bgr: np.ndarray,
    occlusion_mask: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluates whether the inpainting matches the surrounding background context:
    - Seam disparity: edge gradient discontinuity across the hole boundary.
    - Texture consistency: standard deviation ratio between inpaint patch and surrounding context.
    Returns: {
      "seam_disparity": float,
      "texture_consistency": float,
      "inpaint_quality": "good" | "medium" | "low",
      "needs_manual_review": bool
    }
    """
    hole_bool = occlusion_mask > 127
    if not np.any(hole_bool):
        return {
            "seam_disparity": 0.0,
            "texture_consistency": 1.0,
            "inpaint_quality": "good",
            "needs_manual_review": False,
        }

    orig_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY) if len(original_bgr.shape) == 3 else original_bgr
    inpaint_gray = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2GRAY) if len(inpainted_bgr.shape) == 3 else inpainted_bgr

    # Boundary perimeter
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated_hole = cv2.dilate((hole_bool).astype(np.uint8) * 255, kernel)
    perimeter_band = cv2.bitwise_and(dilated_hole, cv2.bitwise_not((hole_bool).astype(np.uint8) * 255))

    # Inner perimeter (inside the inpaint region)
    eroded_hole = cv2.erode((hole_bool).astype(np.uint8) * 255, kernel)
    inner_band = cv2.bitwise_and((hole_bool).astype(np.uint8) * 255, cv2.bitwise_not(eroded_hole))

    # Gradient difference across boundary
    sobel_inpaint = cv2.Sobel(inpaint_gray, cv2.CV_32F, 1, 1, ksize=3)
    sobel_orig = cv2.Sobel(orig_gray, cv2.CV_32F, 1, 1, ksize=3)

    mean_outer_grad = float(np.mean(np.abs(sobel_orig[perimeter_band > 0]))) if np.any(perimeter_band > 0) else 0.0
    mean_inner_grad = float(np.mean(np.abs(sobel_inpaint[inner_band > 0]))) if np.any(inner_band > 0) else 0.0

    denom = max(1.0, mean_outer_grad + mean_inner_grad)
    seam_disparity = round(abs(mean_outer_grad - mean_inner_grad) / denom, 3)

    # Texture variance consistency
    std_outer = float(np.std(orig_gray[perimeter_band > 0])) if np.any(perimeter_band > 0) else 1.0
    std_inner = float(np.std(inpaint_gray[hole_bool])) if np.any(hole_bool) else 1.0
    var_ratio = round(min(std_inner, std_outer) / max(1e-3, max(std_inner, std_outer)), 3)

    # Classification
    if seam_disparity < 0.25 and var_ratio > 0.45:
        quality = "good"
        needs_review = False
    elif seam_disparity < 0.45 and var_ratio > 0.20:
        quality = "medium"
        needs_review = False
    else:
        quality = "low"
        needs_review = True

    return {
        "seam_disparity": seam_disparity,
        "texture_consistency": var_ratio,
        "inpaint_quality": quality,
        "needs_manual_review": needs_review,
    }
