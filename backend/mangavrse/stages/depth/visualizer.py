"""
Visualizer for the Depth stage:
1. Continuous colormapped depth map (depth_map.png).
2. Stepped layer order visualization (depth_visualization.png).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .depth_extractor import load_layer_mask

logger = logging.getLogger(__name__)


def render_depth_map(depth_map: np.ndarray, output_path: Path) -> Path:
    """
    Renders continuous depth map with a perceptual colormap (TURBO).
    0.0 (Far) = Deep Violet / Navy
    1.0 (Near) = Bright Yellow / Red
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth_u8 = (np.clip(depth_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output_path), colored)
    return output_path


def render_depth_visualization(
    depth_map: np.ndarray,
    layers: list[dict[str, Any]],
    output_path: Path,
    panel_dir: Path | None = None,
    source_image: np.ndarray | None = None,
) -> Path:
    """
    Renders a clear 2-pane visual representation of layer order:
    - Left Pane: Spatial Depth Map showing layers tinted by their discrete assigned depth,
                 with bounding boxes and layer labels.
    - Right Pane: Ordered Layer Ladder showing layers sorted by zOrder (0 to N-1) with
                  depth bars, categories, and relative positions.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = depth_map.shape[:2]

    # Create base canvas for left pane
    if source_image is not None:
        left_base = cv2.resize(source_image, (w, h))
        if len(left_base.shape) == 2:
            left_base = cv2.cvtColor(left_base, cv2.COLOR_GRAY2BGR)
        elif left_base.shape[2] == 4:
            left_base = cv2.cvtColor(left_base, cv2.COLOR_BGRA2BGR)
        # Dim source image to allow overlays to pop
        left_base = cv2.addWeighted(left_base, 0.35, np.zeros_like(left_base), 0.65, 0)
    else:
        left_base = np.zeros((h, w, 3), dtype=np.uint8)

    # Sort layers by zOrder
    sorted_layers = sorted(layers, key=lambda l: l.get("zOrder", 0))

    # Color palette helper from depth (TURBO colormap LUT)
    turbo_lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_TURBO).reshape(256, 3)

    # Paint layers onto left pane
    left_overlay = left_base.copy()
    for lyr in sorted_layers:
        depth_val = float(lyr.get("depth", 0.5))
        lut_idx = int(np.clip(depth_val * 255.0, 0, 255))
        color = [int(c) for c in turbo_lut[lut_idx]]

        mask = load_layer_mask(lyr, panel_dir=panel_dir, shape=(h, w))
        if mask is not None and np.any(mask > 128):
            roi = mask > 128
            left_overlay[roi] = (0.4 * left_overlay[roi] + 0.6 * np.array(color)).astype(np.uint8)

        # Draw bbox if present
        bbox = lyr.get("bbox")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            cv2.rectangle(left_overlay, (x0, y0), (x1, y1), color, 1)

    # Annotate top 12 primary layers on left pane to avoid clutter
    step = max(1, len(sorted_layers) // 12)
    for lyr in sorted_layers[::step]:
        bbox = lyr.get("bbox")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            z_idx = lyr.get("zOrder", 0)
            d_val = lyr.get("depth", 0.0)
            tag = f"z{z_idx}:{lyr.get('id', '')} d={d_val:.2f}"
            cv2.putText(left_overlay, tag, (x0 + 4, max(y0 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(left_overlay, tag, (x0 + 4, max(y0 + 16, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Add Left Pane Title
    cv2.putText(left_overlay, "SPATIAL LAYER DEPTH", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(left_overlay, "SPATIAL LAYER DEPTH", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)

    # Create Right Pane: Layer Order Ladder
    right_w = max(500, int(w * 0.65))
    right_pane = np.full((h, right_w, 3), 26, dtype=np.uint8)  # dark slate background

    cv2.putText(right_pane, "LAYER ORDER (zOrder: 0 = Far -> Front)", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

    # Render up to 20 representative layer rows
    n_display = min(len(sorted_layers), 22)
    if n_display > 0:
        row_h = max(24, int((h - 80) / n_display))
        start_y = 60

        # Subsample if more layers exist
        indices = np.linspace(0, len(sorted_layers) - 1, n_display, dtype=int)
        for i, idx in enumerate(indices):
            lyr = sorted_layers[idx]
            y = start_y + i * row_h
            z_idx = lyr.get("zOrder", idx)
            depth_val = float(lyr.get("depth", 0.0))
            lut_idx = int(np.clip(depth_val * 255.0, 0, 255))
            color = [int(c) for c in turbo_lut[lut_idx]]

            # Depth bar gauge
            bar_x0 = 16
            bar_w = int((right_w - 220) * depth_val)
            cv2.rectangle(right_pane, (bar_x0, y + 4), (bar_x0 + max(bar_w, 6), y + row_h - 4), color, -1)

            # Text metadata
            lbl = lyr.get("label") or lyr.get("type") or "layer"
            part = lyr.get("part")
            name_str = f"z{z_idx}: {lyr.get('id', '')} [{lbl}{'/' + part if part else ''}]"
            depth_str = f"d={depth_val:.3f}"
            if lyr.get("is_override"):
                depth_str += " *"

            cv2.putText(right_pane, name_str, (bar_x0 + 8, y + row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right_pane, depth_str, (right_w - 95, y + row_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)

    # Combine side-by-side
    combined = np.hstack([left_overlay, right_pane])
    cv2.imwrite(str(output_path), combined)
    return output_path
