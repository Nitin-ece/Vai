"""
Parallax Tester: renders multi-plane parallax displacement test composites
to immediately expose any holes, missing background, or edge halos.
"""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def render_parallax_test(
    background_bgr: np.ndarray,
    layers: list[dict[str, Any]],
    shift_scale: float = 25.0,
    angle_degrees: float = -30.0,
) -> np.ndarray:
    """
    Simulates a 2.5D camera tilt by shifting layers proportionally to their suggested depth:
    Background plate remains static at depth=0.0.
    Foreground layers (characters, bubbles, props) translate along (dx, dy).
    Returns: BGR diagnostic composite.
    """
    h, w = background_bgr.shape[:2]
    composite = background_bgr.astype(np.float32).copy()

    # Sort layers by suggested_z_order so closest layers render on top
    sorted_layers = sorted(layers, key=lambda l: l.get("suggested_z_order", 0))

    rad = np.radians(angle_degrees)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    for lyr in sorted_layers:
        rgba = lyr.get("rgba")
        if rgba is None or rgba.shape[0] != h or rgba.shape[1] != w:
            continue

        depth = float(lyr.get("suggested_depth", 0.5))
        # Relative depth offset (0.0 at background to 1.0 at closest)
        rel_depth = max(0.0, min(1.0, (depth - 0.05) / 0.95))

        dx = shift_scale * rel_depth * cos_a
        dy = shift_scale * rel_depth * sin_a

        # 2D translation affine matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_rgba = cv2.warpAffine(rgba, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Alpha composite onto current background canvas
        fg_rgb = shifted_rgba[:, :, :3].astype(np.float32)
        alpha = (shifted_rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]

        composite = fg_rgb * alpha + composite * (1.0 - alpha)

    res_bgr = np.clip(composite, 0, 255).astype(np.uint8)

    # Add diagnostic side-by-side or banner
    banner = np.full((32, w, 3), 25, dtype=np.uint8)
    msg = f"PARALLAX GAP TEST (Shift: {shift_scale}px @ {angle_degrees}deg) - Layer Seam & Halo Check"
    cv2.putText(banner, msg, (15, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    return np.vstack([banner, res_bgr])
