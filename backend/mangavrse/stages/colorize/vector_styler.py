"""
Vector Styler: applies color fills to SVG vector layers and
assembles the layered scene with ink line art and screentone shading on top.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import re

import cv2
import numpy as np


def apply_color_to_svg(
    svg_content: str,
    color_hex: str,
    opacity: float = 1.0,
) -> str:
    """Injects or updates fill color and opacity on SVG shapes and groups."""
    # If the SVG has <g> or <polygon> elements without fill, set fill
    # First, replace existing fill if present
    def repl_fill(match: re.Match) -> str:
        tag = match.group(0)
        if "fill=" in tag:
            tag = re.sub(r'fill="[^"]*"', f'fill="{color_hex}"', tag)
        else:
            tag = tag[:-1] + f' fill="{color_hex}">'
        if "fill-opacity=" in tag:
            tag = re.sub(r'fill-opacity="[^"]*"', f'fill-opacity="{opacity}"', tag)
        else:
            tag = tag[:-1] + f' fill-opacity="{opacity}">'
        return tag

    # Apply to top-level group or polygons
    if "<g " in svg_content or "<g>" in svg_content:
        styled = re.sub(r'<g(\s+[^>]*)?>', f'<g fill="{color_hex}" fill-opacity="{opacity}">', svg_content, count=1)
    else:
        styled = re.sub(r'<polygon\s+[^>]*>', repl_fill, svg_content)

    return styled


def style_layer_svg(
    in_svg_path: Path,
    out_svg_path: Path,
    color_hex: str,
    opacity: float = 1.0,
) -> bool:
    """Reads layer SVG, applies the color fill, and writes to out_svg_path."""
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    if not in_svg_path.exists():
        return False

    content = in_svg_path.read_text(encoding="utf-8")
    styled_content = apply_color_to_svg(content, color_hex, opacity)
    out_svg_path.write_text(styled_content, encoding="utf-8")
    return True


def assemble_colored_panel_svg(
    width: int,
    height: int,
    layers: list[dict[str, Any]],
    layer_colors: dict[str, str],
    out_svg_path: Path,
    lines_svg_path: Path | None = None,
    tones_svg_path: Path | None = None,
) -> bool:
    """
    Assembles a master colored vector SVG:
    Layer 0: Color fills per layer (sorted by suggested_z_order).
    Layer 1: Screentone pattern overlay (tones.svg) providing natural manga shading.
    Layer 2: Structural ink linework (lines.svg) rendered ON TOP of all colors.
    """
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_layers = sorted(layers, key=lambda l: l.get("suggested_z_order", 0))

    defs_content = ""
    color_layer_elements: list[str] = []

    for lyr in sorted_layers:
        lid = lyr["id"]
        color = layer_colors.get(lid, "#E0E0E0")
        svg_p_rel = lyr.get("svgPath")
        if not svg_p_rel:
            continue

        base_dir = out_svg_path.parent
        svg_file = base_dir / svg_p_rel
        if not svg_file.exists():
            continue

        try:
            tree = ET.parse(str(svg_file))
            root = tree.getroot()
            for child in root:
                tag = child.tag.split("}")[-1]
                if tag == "defs":
                    for d_child in child:
                        defs_content += "\n    " + ET.tostring(d_child, encoding="unicode")
                elif tag in ("g", "polygon", "path"):
                    # Wrap in styled group
                    c_str = ET.tostring(child, encoding="unicode")
                    color_layer_elements.append(f'  <!-- Layer {lid} -->\n  <g fill="{color}">\n    {c_str}\n  </g>')
        except Exception:
            continue

    # Extract tones SVG content (screentone texture on top of color)
    tones_elements = ""
    if tones_svg_path and tones_svg_path.exists():
        try:
            tones_tree = ET.parse(str(tones_svg_path))
            t_root = tones_tree.getroot()
            for child in t_root:
                tag = child.tag.split("}")[-1]
                if tag == "defs":
                    for d_child in child:
                        defs_content += "\n    " + ET.tostring(d_child, encoding="unicode")
                elif tag in ("g", "path"):
                    tones_elements += "\n  " + ET.tostring(child, encoding="unicode")
        except Exception:
            pass

    # Extract lines SVG content (black ink linework on top)
    lines_elements = ""
    if lines_svg_path and lines_svg_path.exists():
        try:
            lines_tree = ET.parse(str(lines_svg_path))
            l_root = lines_tree.getroot()
            for child in l_root:
                tag = child.tag.split("}")[-1]
                if tag == "defs":
                    for d_child in child:
                        defs_content += "\n    " + ET.tostring(d_child, encoding="unicode")
                elif tag in ("g", "path"):
                    lines_elements += "\n  " + ET.tostring(child, encoding="unicode")
        except Exception:
            pass

    color_layers_str = "\n".join(color_layer_elements)

    master_svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    {defs_content}
  </defs>

  <!-- 1. Semantic Color Fills -->
  <g id="color_fills">
{color_layers_str}
  </g>

  <!-- 2. Screentone Shading (under lines, over colors) -->
  <g id="screentone_shading" opacity="0.85">
    {tones_elements}
  </g>

  <!-- 3. Ink Linework (top layer) -->
  <g id="line_art">
    {lines_elements}
  </g>
</svg>
"""
    out_svg_path.write_text(master_svg, encoding="utf-8")
    return True
