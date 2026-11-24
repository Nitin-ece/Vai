"""
Extractor: isolates line art contours and separates periodic screentone fields
using OpenCV morphological filtering and U-Net line extraction.
"""
from __future__ import annotations

import cv2
import numpy as np

from mangavrse.models.unet_line_extractor import UNetLineExtractorProvider

# Global cached extractor provider
_unet_line_extractor: UNetLineExtractorProvider | None = None


def get_unet_extractor() -> UNetLineExtractorProvider:
    global _unet_line_extractor
    if _unet_line_extractor is None:
        _unet_line_extractor = UNetLineExtractorProvider()
    return _unet_line_extractor


def extract_lines_and_tones(
    image_bgr: np.ndarray,
    adaptive_block_size: int = 15,
    adaptive_c: int = 8,
    screentone_separation: bool = True,
    min_tone_frequency: float = 0.12,
    use_unet: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Decomposes *image_bgr* into:
    1. line_mask: binary uint8 (255 for structural ink outlines, 0 for background).
    2. tone_mask: binary uint8 (255 inside continuous screentone regions, 0 outside).
    3. tone_density: estimated fill percentage of dots within the tone regions.
    Uses OpenCV adaptive thresholding + U-Net line extraction.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()
    h, w = gray.shape

    if use_unet:
        extractor = get_unet_extractor()
        return extractor.predict(gray)

    # Ensure block size is odd and >= 3
    if adaptive_block_size % 2 == 0:
        adaptive_block_size += 1
    adaptive_block_size = max(3, adaptive_block_size)

    # 1. High-sensitivity adaptive threshold to capture fine lines and dots
    raw_binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block_size,
        adaptive_c,
    )

    if not screentone_separation:
        return raw_binary, np.zeros((h, w), dtype=np.uint8), 0.0

    # 2. Screentone separation:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, kernel)

    # Isolated dots are the residual difference
    detached_dots = cv2.bitwise_and(raw_binary, cv2.bitwise_not(line_mask))

    # 3. Screentone region container:
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    tone_envelope = cv2.morphologyEx(detached_dots, cv2.MORPH_CLOSE, close_kernel)

    # Remove tiny isolated noise in tone envelope (< 100 pixels)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tone_envelope, connectivity=8)
    tone_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 150:
            tone_mask[labels == i] = 255

    # Compute average dot density inside tone regions
    tone_pixels = np.count_nonzero(tone_mask)
    if tone_pixels > 0:
        dot_pixels_in_tone = np.count_nonzero(cv2.bitwise_and(detached_dots, tone_mask))
        tone_density = float(dot_pixels_in_tone) / float(tone_pixels)
    else:
        tone_density = 0.0

    return line_mask, tone_mask, round(tone_density, 3)
