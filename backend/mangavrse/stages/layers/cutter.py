"""
Cutter: extracts transparent RGBA PNG cutouts and matching vector SVGs
with anti-halo mask padding and sub-pixel edge feathering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def pad_and_feather_mask(
    binary_mask: np.ndarray,
    padding: int = 2,
    feather_radius: float = 2.0,
) -> np.ndarray:
    """
    Pads *binary_mask* outward by *padding* pixels to avoid dark edge fringes,
    then applies Gaussian edge feathering over *feather_radius* for smooth alpha compositing.
    Returns: uint8 alpha mask (0 - 255).
    """
    h, w = binary_mask.shape[:2]
    mask_bool = binary_mask > 127

    if np.count_nonzero(mask_bool) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    mask_u8 = mask_bool.astype(np.uint8) * 255

    # 1. Padding / dilation
    if padding > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding * 2 + 1, padding * 2 + 1))
        padded = cv2.dilate(mask_u8, kernel)
    else:
        padded = mask_u8

    # 2. Sub-pixel Gaussian alpha feathering
    if feather_radius > 0.1:
        ksize = int(round(feather_radius * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(padded.astype(np.float32), (ksize, ksize), feather_radius)
        # Rescale so the core remains 255 and edges fade smoothly to 0
        feathered = np.clip(blurred * 1.05, 0, 255).astype(np.uint8)
        return feathered

    return padded


def extract_layer_rgba(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    padding: int = 2,
    feather_radius: float = 2.0,
) -> np.ndarray:
    """
    Extracts a transparent 4-channel RGBA image corresponding to the mask.
    Pixels outside the mask are completely transparent (alpha = 0).
    """
    h, w = image_bgr.shape[:2]
    alpha = pad_and_feather_mask(mask, padding=padding, feather_radius=feather_radius)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha
    return rgba


def create_clipped_svg(
    svg_source_path: Path | None,
    mask: np.ndarray,
    out_svg_path: Path,
    width: int,
    height: int,
    layer_id: str,
) -> bool:
    """
    Produces a matching layer SVG (layers/<id>.svg).
    If an existing panel SVG (lines.svg) is available, creates a standalone SVG
    with a <clipPath> derived from the mask's outer polygon contour.
    """
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask > 127).astype(np.uint8) * 255

    # Find external contours of mask to form SVG clipPath polygon
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    polygon_elements: list[str] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20:
            continue
        pts_str = " ".join(f"{p[0][0]},{p[0][1]}" for p in cnt)
        polygon_elements.append(f'<polygon points="{pts_str}" />')

    clip_id = f"clip_{layer_id}"
    clip_def = f'<clipPath id="{clip_id}">\n    ' + "\n    ".join(polygon_elements) + "\n  </clipPath>"

    inner_content = ""
    if svg_source_path and svg_source_path.exists():
        try:
            tree = ET.parse(str(svg_source_path))
            root = tree.getroot()
            # Extract child tags (paths, patterns, etc.)
            for child in root:
                tag_name = child.tag.split("}")[-1]
                if tag_name in ("defs", "path", "g", "polygon"):
                    inner_content += ET.tostring(child, encoding="unicode")
        except Exception:
            inner_content = ""

    if not inner_content:
        # Fallback: render the polygon fill itself
        poly_fills = "\n  ".join(f'<polygon points="{p.split(chr(34))[1]}" fill="#000000" />' for p in polygon_elements)
        inner_content = poly_fills

    svg_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    {clip_def}
  </defs>
  <g clip-path="url(#{clip_id})">
    {inner_content}
  </g>
</svg>
"""
    out_svg_path.write_text(svg_str, encoding="utf-8")
    return True
