"""
Interactive Refiner: handles user click point prompts (positive/negative)
and bounding box prompts to refine or generate individual masks for Studio use.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .line_snapper import snap_mask_to_lines
from .sam_provider import SAM2SegmenterProvider


def refine_mask_with_prompts(
    image_bgr: np.ndarray,
    points: list[dict[str, Any]] | None = None,
    box: list[float] | None = None,
    existing_mask: np.ndarray | None = None,
    provider: SAM2SegmenterProvider | None = None,
    line_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, list[int], int]:
    """
    Refines a mask interactively based on positive and negative click points or bounding boxes.
    Returns:
      refined_mask: np.ndarray uint8 (255 foreground, 0 background)
      confidence: float (0.0 to 1.0)
      bbox: [x0, y0, x1, y1]
      area: int (nonzero pixel count)
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if provider is None:
        provider = SAM2SegmenterProvider()

    pt_coords: list[tuple[float, float]] = []
    pt_labels: list[int] = []

    if points:
        for p in points:
            pt_coords.append((float(p["x"]), float(p["y"])))
            pt_labels.append(int(p.get("label", 1)))

    box_tuple = tuple(box) if (box and len(box) == 4) else None

    # Predict mask
    refined_raw, conf = provider.predict_prompt(
        image_rgb=image_rgb,
        points=pt_coords if pt_coords else None,
        point_labels=pt_labels if pt_labels else None,
        box=box_tuple,
        existing_mask=existing_mask,
    )

    # Snap to local line art outlines to prevent bleed
    snapped = snap_mask_to_lines(refined_raw, line_mask)

    # Compute bbox and area
    y_idxs, x_idxs = np.where(snapped > 127)
    if len(x_idxs) > 0:
        bbox = [int(np.min(x_idxs)), int(np.min(y_idxs)), int(np.max(x_idxs)), int(np.max(y_idxs))]
        area = int(len(x_idxs))
    else:
        bbox = [0, 0, 0, 0]
        area = 0

    return snapped, round(conf, 3), bbox, area
