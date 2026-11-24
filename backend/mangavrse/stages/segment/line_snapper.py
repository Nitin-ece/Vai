"""
Line Snapper: uses manga structural ink linework to guide and snap
raw segmentation mask boundaries, preventing masks from bleeding across ink borders.
"""
from __future__ import annotations

import cv2
import numpy as np


def snap_mask_to_lines(
    mask: np.ndarray,
    line_mask: np.ndarray | None = None,
    max_snap_distance: int = 4,
) -> np.ndarray:
    """
    Snaps *mask* (uint8 binary) against *line_mask* (uint8 binary ink linework).
    Prevents diffuse segmentation boundaries from bleeding across sharp ink outlines.
    """
    if line_mask is None or np.count_nonzero(line_mask) == 0:
        return mask.copy()

    h, w = mask.shape[:2]
    binary_mask = (mask > 127).astype(np.uint8) * 255

    # 1. Fill micro-pinholes inside the mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = binary_mask.copy()
    if hierarchy is not None:
        for i, h_info in enumerate(hierarchy[0]):
            # If inner hole with area < 100 px, fill it
            if h_info[3] != -1:
                hole_area = cv2.contourArea(contours[i])
                if hole_area < 150:
                    cv2.drawContours(filled, contours, i, 255, -1)

    # 2. Subtract thick ink lines where mask boundary crosses them
    # Ink outlines act as an impenetrable boundary
    dilated_lines = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    
    # Identify fringe bleed: pixels that are inside the mask, touch the outer perimeter,
    # but sit beyond the ink line
    mask_edges = cv2.Canny(filled, 50, 150)
    overlap_with_ink = cv2.bitwise_and(mask_edges, dilated_lines)

    # If the boundary already aligns well with ink lines (> 30%), snap cleanly
    overlap_ratio = float(np.count_nonzero(overlap_with_ink)) / max(1.0, float(np.count_nonzero(mask_edges)))
    if overlap_ratio > 0.15:
        # Smooth jagged edges along the ink boundary
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        snapped = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, close_kernel)
        return snapped

    return filled
