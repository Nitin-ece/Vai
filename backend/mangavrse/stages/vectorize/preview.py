"""
Preview Generator: produces side-by-side 4x zoomed visual inspections
comparing raster source pixels vs vector path curves at high magnification.
"""
from __future__ import annotations

import cv2
import numpy as np

from .fidelity import rasterize_vector_paths
from .tracer import VectorPath


def generate_preview_4x(
    source_bgr: np.ndarray,
    paths: list[VectorPath],
    crop_size: int = 160,
    zoom_scale: int = 4,
) -> np.ndarray:
    """
    Finds a high-detail patch, extracts a crop, and renders:
    Left: Raster source nearest-neighbor scaled 4x (exposing pixels).
    Right: Vector paths rendered natively at 4x resolution (sharp anti-aliased curves).
    """
    h, w = source_bgr.shape[:2]

    # Clamp crop_size so it never exceeds source image dimensions
    crop_size = max(20, min(crop_size, h, w))

    # Find center of mass / high detail area
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY) if len(source_bgr.shape) == 3 else source_bgr
    edges = cv2.Canny(gray, 50, 150)

    # Search for crop window with maximum edge pixel count
    step = max(10, crop_size // 4)
    best_x, best_y, max_edges = 0, 0, -1

    for y in range(0, max(1, h - crop_size + 1), step):
        for x in range(0, max(1, w - crop_size + 1), step):
            edge_count = int(np.count_nonzero(edges[y : y + crop_size, x : x + crop_size]))
            if edge_count > max_edges:
                max_edges = edge_count
                best_x, best_y = x, y

    # Clamp best_x, best_y
    best_x = max(0, min(best_x, w - crop_size))
    best_y = max(0, min(best_y, h - crop_size))

    # Extract crop from source
    src_crop = source_bgr[best_y : best_y + crop_size, best_x : best_x + crop_size]

    # Render vector paths at full 4x magnification
    vector_4x_full = rasterize_vector_paths(paths, w, h, scale=zoom_scale)
    # Invert so white background with black ink
    vec_4x_inv = 255 - vector_4x_full
    vec_4x_bgr = cv2.cvtColor(vec_4x_inv, cv2.COLOR_GRAY2BGR)

    crop_y0 = best_y * zoom_scale
    crop_x0 = best_x * zoom_scale
    crop_y1 = crop_y0 + (crop_size * zoom_scale)
    crop_x1 = crop_x0 + (crop_size * zoom_scale)

    vec_crop = vec_4x_bgr[crop_y0:crop_y1, crop_x0:crop_x1]

    # 4x upscale source crop with nearest-neighbor to reveal pixel grid
    target_dim = crop_size * zoom_scale
    src_4x = cv2.resize(src_crop, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)

    # Ensure vec_crop is exactly target_dim x target_dim
    if vec_crop.shape[0] != target_dim or vec_crop.shape[1] != target_dim:
        vec_crop = cv2.resize(vec_crop, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)

    # Compose side-by-side preview with header
    header_h = 60
    divider_w = 6

    canvas = np.full((target_dim + header_h, (target_dim * 2) + divider_w, 3), 30, dtype=np.uint8)

    # Headers
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"SOURCE RASTER (4x Zoom)", (20, 38), font, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"VECTOR SVG (4x Zoom)", (target_dim + divider_w + 20, 38), font, 0.75, (0, 255, 180), 2, cv2.LINE_AA)

    canvas[header_h:, :target_dim] = src_4x
    canvas[:, target_dim : target_dim + divider_w] = (80, 80, 80)
    canvas[header_h:, target_dim + divider_w :] = vec_crop

    return canvas
