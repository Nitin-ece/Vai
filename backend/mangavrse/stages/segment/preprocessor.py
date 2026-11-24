"""
Manga-specific preprocessors for the Segment stage:
- Screentone desensitization / smoothing to eliminate halftone noise from mask proposals.
- Rule-based Speech Bubble and SFX candidate pre-detector.
"""
from __future__ import annotations

import cv2
import numpy as np


def smooth_for_segmentation(
    image_bgr: np.ndarray,
    screentone_smoothing: bool = True,
) -> np.ndarray:
    """
    Suppresses periodic halftone screentone dots while preserving sharp ink line art.
    Uses median blur + bilateral filter.
    """
    if not screentone_smoothing:
        return image_bgr.copy()

    # 1. Light median blur knocks out 1x1 to 3x3 isolated screentone frequency dots
    med = cv2.medianBlur(image_bgr, 3)

    # 2. Bilateral filter preserves sharp ink borders while flattening textured tone fields
    smoothed = cv2.bilateralFilter(med, d=7, sigmaColor=75, sigmaSpace=75)
    return smoothed


def detect_candidate_bubbles_and_sfx(
    image_bgr: np.ndarray,
) -> list[dict]:
    """
    Pre-detects candidate speech bubbles and high-contrast sound effects (SFX)
    so they are never merged into character clothing, hair, or backgrounds.

    Returns list of dicts:
    [
      {
        "label": "speech_bubble" | "sfx",
        "mask": np.ndarray (binary uint8 255),
        "confidence": float,
        "bbox": [x0, y0, x1, y1],
        "area": int
      }
    ]
    """
    h, w = image_bgr.shape[:2]
    total_area = h * w
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()

    candidates: list[dict] = []

    # 1. High paper-brightness mask for speech bubbles (bubbles are enclosed white interiors)
    # The black ink border (gray < 210) acts as an impenetrable barrier separating the bubble from outer background
    _, paper_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    # 4-connectivity prevents diagonal leaking through thin ink lines
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(paper_white, connectivity=4)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        # Bubble must be between 0.3% and 35% of total panel area
        if area < int(total_area * 0.003) or area > int(total_area * 0.35):
            continue

        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Skip if it touches the outer canvas perimeter (that is the background gutter)
        if bx <= 2 or by <= 2 or (bx + bw >= w - 2) or (by + bh >= h - 2):
            continue

        # Extract sub-mask and bridge internal text strokes
        sub_mask = (labels[by : by + bh, bx : bx + bw] == i).astype(np.uint8) * 255
        # Fill any internal text holes within the bubble
        cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        cv2.drawContours(sub_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        cnt_area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * (cnt_area / (perimeter * perimeter))
        hull = cv2.convexHull(cnt)
        solidity = float(cnt_area) / max(1.0, cv2.contourArea(hull))

        # Bubbles are highly convex and relatively smooth (solidity > 0.80)
        aspect = float(bw) / max(1.0, float(bh))
        if solidity >= 0.78 and 0.35 <= aspect <= 3.0:
            # Check internal text presence in original grayscale (text produces high edge count)
            crop_gray = gray[by : by + bh, bx : bx + bw]
            edges = cv2.Canny(crop_gray, 50, 150)
            text_edge_density = float(np.count_nonzero(edges[sub_mask > 0])) / max(1.0, float(cnt_area))

            if text_edge_density > 0.010 or circularity > 0.45:
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[by : by + bh, bx : bx + bw] = sub_mask

                candidates.append({
                    "label": "speech_bubble",
                    "mask": full_mask,
                    "confidence": round(min(0.96, 0.75 + solidity * 0.15 + (0.05 if text_edge_density > 0.02 else 0)), 3),
                    "bbox": [bx, by, bx + bw, by + bh],
                    "area": int(np.count_nonzero(full_mask)),
                })

    # 2. SFX / Action Text candidate detection
    # Strong high-contrast black contours with irregular/spiky boundaries and high variance
    # Adaptive threshold to isolate ink strokes
    ink_binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    # Mask out already detected bubbles
    for c in candidates:
        ink_binary[c["mask"] > 0] = 0

    sfx_num, sfx_labels, sfx_stats, _ = cv2.connectedComponentsWithStats(ink_binary, connectivity=8)
    for i in range(1, sfx_num):
        area = int(sfx_stats[i, cv2.CC_STAT_AREA])
        # SFX strokes are distinct clusters of area 200 - 15,000 px
        if 250 <= area <= int(total_area * 0.06):
            sx = int(sfx_stats[i, cv2.CC_STAT_LEFT])
            sy = int(sfx_stats[i, cv2.CC_STAT_TOP])
            sw = int(sfx_stats[i, cv2.CC_STAT_WIDTH])
            sh = int(sfx_stats[i, cv2.CC_STAT_HEIGHT])
            
            # Skip if touching all image borders
            if sx <= 2 and sy <= 2 and (sx + sw >= w - 2) and (sy + sh >= h - 2):
                continue

            sub_mask = (sfx_labels[sy : sy + sh, sx : sx + sw] == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * np.pi * (cv2.contourArea(cnt) / (perimeter * perimeter))
            hull = cv2.convexHull(cnt)
            solidity = float(cv2.contourArea(cnt)) / max(1.0, cv2.contourArea(hull))

            # SFX is spiky / non-convex (solidity < 0.70) with sharp corners
            if solidity < 0.65 and circularity < 0.25:
                # Group adjacent SFX characters with morphological dilation
                full_mask = np.zeros((h, w), dtype=np.uint8)
                full_mask[sy : sy + sh, sx : sx + sw] = sub_mask
                
                # Check if it has already been matched
                overlap = False
                for c in candidates:
                    if np.logical_and(c["mask"] > 0, full_mask > 0).any():
                        overlap = True
                        break
                if not overlap and area >= 400:
                    candidates.append({
                        "label": "sfx",
                        "mask": full_mask,
                        "confidence": 0.85,
                        "bbox": [sx, sy, sx + sw, sy + sh],
                        "area": int(np.count_nonzero(full_mask)),
                    })

    return candidates
