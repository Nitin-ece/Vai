"""
Visualizer: renders color-coded semantic overlays with bounding boxes,
labels, parentage markers, and confidence/review flags for human inspection.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

# Distinct BGR color palette for semantic layer types
COLOR_MAP: dict[str, tuple[int, int, int]] = {
    "background": (180, 180, 180),     # Soft Gray
    "environment": (120, 180, 140),    # Sage Green
    "prop": (230, 180, 80),            # Golden Ochre
    "character": (220, 140, 40),       # Blue / Cyan
    "clothing": (240, 100, 120),       # Violet / Purple
    "body": (200, 170, 100),           # Steel Blue
    "head": (120, 200, 240),           # Peach / Yellow
    "hair": (80, 80, 220),             # Crimson / Coral
    "eyes": (60, 220, 220),            # Cyan / Gold
    "effect": (60, 60, 240),           # Vivid Red
    "speech_bubble": (40, 220, 160),   # Neon Mint
    "sfx": (40, 140, 240),             # Bright Orange
}


def create_segment_overlay(
    image_bgr: np.ndarray,
    segments: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Renders an informative color-coded overlay image:
    1. Grayscale panel backdrop with subtle contrast.
    2. 45% alpha color fills for all detected segments.
    3. Crisp contour outlines per segment.
    4. Text badges showing segment ID, label (part), and [REVIEW] if flagged.
    5. Top banner legend.
    """
    h, w = image_bgr.shape[:2]

    # Convert backdrop to lightly dimmed grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr.copy()
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    color_layer = np.zeros_like(canvas)

    # Sort segments by suggested_z_order so foreground layers paint on top
    sorted_segs = sorted(segments, key=lambda s: s.get("suggested_z_order", 0))

    for seg in sorted_segs:
        sid = seg["id"]
        mask = masks.get(sid)
        if mask is None or np.count_nonzero(mask > 127) == 0:
            continue

        label = seg["label"]
        part = seg.get("part")
        lookup_key = part if (label == "character_part" and part in COLOR_MAP) else label
        color = COLOR_MAP.get(lookup_key, (200, 200, 200))

        # Color fill
        color_layer[mask > 127] = color

        # Draw crisp outline
        contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)

    # Alpha blend color fills
    cv2.addWeighted(color_layer, 0.42, canvas, 1.0, 0.0, canvas)

    # Add text tags and bounding boxes
    for seg in sorted_segs:
        sid = seg["id"]
        bbox = seg.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        label = seg["label"]
        part = seg.get("part")
        conf = seg.get("confidence", 0.0)
        needs_review = seg.get("needs_manual_review", False)

        lookup_key = part if (label == "character_part" and part in COLOR_MAP) else label
        color = COLOR_MAP.get(lookup_key, (200, 200, 200))

        # Draw dotted/thin bounding box
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 1)

        # Label text
        tag = f"{sid}: {part if part else label} ({conf:.2f})"
        if needs_review:
            tag += " [REVIEW]"

        # Text banner
        font_scale = 0.40
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ty0 = max(18, y0 - 4)
        cv2.rectangle(canvas, (x0, ty0 - th - 3), (x0 + tw + 6, ty0 + 3), (20, 20, 20), -1)
        text_color = (80, 80, 255) if needs_review else (255, 255, 255)
        cv2.putText(canvas, tag, (x0 + 3, ty0 - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)

    # Top header legend
    legend_bar = np.full((32, w, 3), 30, dtype=np.uint8)
    cur_x = 10
    for l_name, l_col in [("Char", (220, 140, 40)), ("Part", (80, 80, 220)), ("Bubble", (40, 220, 160)), ("SFX", (40, 140, 240)), ("Effect", (60, 60, 240)), ("Env", (120, 180, 140))]:
        cv2.rectangle(legend_bar, (cur_x, 8), (cur_x + 14, 22), l_col, -1)
        cv2.putText(legend_bar, l_name, (cur_x + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cur_x += 105

    combined = np.vstack([legend_bar, canvas])
    return combined
