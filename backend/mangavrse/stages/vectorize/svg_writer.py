"""
SVG Writer: produces standards-compliant lines.svg and tones.svg.
Renders screentones as pattern fills or tone envelopes, NOT thousands of traced dots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracer import VectorPath


class SVGWriter:
    """Exports vectorized paths into clean SVG documents."""

    @staticmethod
    def paths_to_svg_d(paths: list[VectorPath]) -> tuple[str, int]:
        """
        Converts a list of VectorPaths to an SVG path 'd' attribute string.
        Returns (d_string, total_nodes).
        """
        d_parts: list[str] = []
        node_count = 0

        for path in paths:
            sx, sy = round(path.start_point[0], 2), round(path.start_point[1], 2)
            d_parts.append(f"M {sx} {sy}")
            node_count += 1

            for seg in path.segments:
                ep_x, ep_y = round(seg.end_point[0], 2), round(seg.end_point[1], 2)
                if seg.is_corner and seg.c:
                    cx, cy = round(seg.c[0], 2), round(seg.c[1], 2)
                    d_parts.append(f"L {cx} {cy} L {ep_x} {ep_y}")
                    node_count += 2
                elif not seg.is_corner and seg.c1 and seg.c2:
                    c1x, c1y = round(seg.c1[0], 2), round(seg.c1[1], 2)
                    c2x, c2y = round(seg.c2[0], 2), round(seg.c2[1], 2)
                    d_parts.append(f"C {c1x} {c1y} {c2x} {c2y} {ep_x} {ep_y}")
                    node_count += 3
                else:
                    d_parts.append(f"L {ep_x} {ep_y}")
                    node_count += 1

            if path.is_closed:
                d_parts.append("Z")

        return " ".join(d_parts), node_count

    @classmethod
    def write_lines_svg(
        cls,
        paths: list[VectorPath],
        width: int,
        height: int,
        out_path: Path,
        fill_color: str = "#000000",
    ) -> tuple[int, int, int]:
        """
        Writes lines.svg containing vector line art.
        Returns (path_count, node_count, file_size_bytes).
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        d_str, node_count = cls.paths_to_svg_d(paths)

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <path d="{d_str}" fill="{fill_color}" fill-rule="evenodd" />
</svg>"""

        out_path.write_text(svg_content, encoding="utf-8")
        file_size = out_path.stat().st_size
        return len(paths), node_count, file_size

    @classmethod
    def write_tones_svg(
        cls,
        tone_paths: list[VectorPath],
        width: int,
        height: int,
        out_path: Path,
        dot_density: float = 0.25,
        dot_radius: float = 1.8,
        dot_spacing: float = 6.0,
    ) -> tuple[int, int, int]:
        """
        Writes tones.svg using an SVG pattern definition instead of individual dots.
        Prevents file size explosion for screentone panels.
        Returns (path_count, node_count, file_size_bytes).
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        d_str, node_count = cls.paths_to_svg_d(tone_paths)

        # Dynamic radius based on dot density
        r = max(1.0, min(dot_radius * (dot_density / 0.25), dot_spacing / 2.0 - 0.5))

        svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <pattern id="manga_screentone" width="{dot_spacing}" height="{dot_spacing}" patternUnits="userSpaceOnUse" patternTransform="rotate(45 0 0)">
      <circle cx="{dot_spacing / 2.0}" cy="{dot_spacing / 2.0}" r="{r:.2f}" fill="#333333" />
    </pattern>
  </defs>
  <path d="{d_str}" fill="url(#manga_screentone)" fill-rule="evenodd" />
</svg>"""

        out_path.write_text(svg_content, encoding="utf-8")
        file_size = out_path.stat().st_size
        return len(tone_paths), node_count, file_size
