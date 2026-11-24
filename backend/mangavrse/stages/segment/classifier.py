"""
Semantic Classifier for Manga Segmentation Masks:
Classifies candidate masks into:
- character
- character_part (head, hair, eyes, body, clothing)
- prop
- background
- environment
- effect (speed lines, impact bursts)
- speech_bubble
- sfx
"""
from __future__ import annotations

import cv2
import numpy as np


def compute_directional_coherence(gray: np.ndarray, mask: np.ndarray) -> float:
    """Computes gradient angle variance; low variance = parallel speed lines / effect."""
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(sobel_x, sobel_y)

    mask_bool = mask > 0
    if np.count_nonzero(mask_bool) < 50:
        return 0.0

    mags = mag[mask_bool]
    angles = angle[mask_bool]

    strong = mags > (np.mean(mags) + 0.5 * np.std(mags))
    if np.count_nonzero(strong) < 30:
        return 0.0

    strong_angles = angles[strong]
    # Circular variance
    sin_sum = np.sum(np.sin(2 * strong_angles))
    cos_sum = np.sum(np.cos(2 * strong_angles))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(strong_angles)
    return float(r)  # 1.0 = perfectly parallel lines


def classify_segment(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    line_mask: np.ndarray | None = None,
    existing_label: str | None = None,
) -> tuple[str, str | None, float, bool]:
    """
    Classifies a binary mask.
    Returns:
      label: str ("character" | "character_part" | "prop" | "background" | "environment" | "effect" | "speech_bubble" | "sfx")
      part: str | None ("head" | "hair" | "eyes" | "body" | "clothing" | None)
      confidence: float (0.0 - 1.0)
      needs_manual_review: bool
    """
    if existing_label in ("speech_bubble", "sfx"):
        return existing_label, None, 0.92, False

    h, w = mask.shape[:2]
    total_area = h * w
    mask_pixels = np.count_nonzero(mask > 127)

    if mask_pixels == 0:
        return "background", None, 0.50, True

    area_ratio = float(mask_pixels) / float(total_area)

    # Bounding box & centroid
    y_indices, x_indices = np.where(mask > 127)
    x0, y0, x1, y1 = int(np.min(x_indices)), int(np.min(y_indices)), int(np.max(x_indices)), int(np.max(y_indices))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    aspect = float(bw) / float(bh)
    cx = float(np.mean(x_indices)) / float(w)
    cy = float(np.mean(y_indices)) / float(h)

    # Edge contact count (touches left, right, top, bottom edges of panel)
    edge_contact = 0
    if x0 <= 3: edge_contact += 1
    if x1 >= w - 4: edge_contact += 1
    if y0 <= 3: edge_contact += 1
    if y1 >= h - 4: edge_contact += 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr

    # 1. Background
    if area_ratio >= 0.70 or edge_contact >= 3:
        return "background", None, 0.95, False

    # 2. Environment / Floor
    if edge_contact >= 2 and y1 >= h - 4 and cy > 0.65 and aspect >= 1.5:
        return "environment", None, 0.88, False

    # 3. Effect (Speed lines / burst)
    dir_coherence = compute_directional_coherence(gray, mask)
    if dir_coherence > 0.65 and (aspect > 2.5 or aspect < 0.4):
        return "effect", None, round(0.75 + dir_coherence * 0.15, 2), False

    # 4. Characters and Character Parts
    # Character figure silhouette with human/portrait aspect ratio
    if (0.025 <= area_ratio <= 0.70) and (0.18 <= aspect <= 3.2) and edge_contact <= 2:
        confidence = 0.94 if (0.15 <= cx <= 0.85 and 0.15 <= cy <= 0.85) else 0.82
        return "character", None, confidence, False

    # Character Subparts or Props
    if 0.003 <= area_ratio < 0.40:
        # Check if high dark ink (hair)
        mean_intensity = float(np.mean(gray[mask > 127]))
        
        # Eyes: tiny, high contrast, oval
        if area_ratio < 0.015 and mean_intensity < 80 and 0.5 <= aspect <= 2.0:
            return "character_part", "eyes", 0.88, False

        # Hair: dark, located in upper region, spiky or flowing
        if cy < 0.52 and mean_intensity < 95:
            return "character_part", "hair", 0.90, False

        # Head / Face: upper region, brighter than hair
        if cy < 0.48 and mean_intensity >= 90:
            return "character_part", "head", 0.88, False

        # Clothing / Body: mid to lower
        if cy >= 0.32:
            part_name = "clothing" if mean_intensity > 70 else "body"
            return "character_part", part_name, 0.86, False

    # Detached item / prop
    if 0.01 <= area_ratio <= 0.15:
        return "prop", None, 0.72, False

    # Low confidence fallback
    return "character_part", "body", 0.58, True
