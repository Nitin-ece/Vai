"""
Colorize Stage: transforms segmented vector layers and transparent cutouts
into editable, palette-driven colored vector scenes with lines & screentone on top.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mangavrse.core.stage import AbstractStage, StageResult

from .colorizer import ColorizerProvider
from .palette_manager import PaletteManager
from .renderer import render_colored_panel
from .vector_styler import assemble_colored_panel_svg, style_layer_svg
from .visualizer import create_colorize_comparison

logger = logging.getLogger(__name__)


class ColorizeStage(AbstractStage):
    """
    Colorize stage producing colored SVG layers, master colored SVG,
    rendered PNG preview, character palette JSON, and before/after comparison.
    """

    name = "colorize"

    def __init__(self, colorizer: ColorizerProvider | None = None) -> None:
        super().__init__(name="colorize")
        self.colorizer = colorizer or ColorizerProvider()

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        col_cfg = config.get("colorize", {})
        default_char_id = str(col_cfg.get("character_id", "char_001"))
        external_palette_path = col_cfg.get("palette_path")

        # Discover target panels
        target_panels: list[tuple[str, Path, Path, Path | None, Path | None, Path]] = []

        direct_manifest = input_dir / "layers_manifest.json"
        direct_segments = input_dir / "segments.json"
        direct_enhanced = input_dir / "enhanced.png"
        direct_original = input_dir / "original.png"

        if (direct_manifest.exists() or direct_segments.exists()) and (direct_enhanced.exists() or direct_original.exists()):
            img_p = direct_enhanced if direct_enhanced.exists() else direct_original
            meta_p = direct_manifest if direct_manifest.exists() else direct_segments
            lines_p = input_dir / "lines.svg" if (input_dir / "lines.svg").exists() else None
            tones_p = input_dir / "tones.svg" if (input_dir / "tones.svg").exists() else None
            target_panels.append(("panel_000", img_p, meta_p, lines_p, tones_p, output_dir))
        else:
            # Search subdirectories for layers_manifest.json or segments.json
            found_meta_files = sorted(input_dir.glob("**/layers_manifest.json"))
            if not found_meta_files:
                found_meta_files = sorted(input_dir.glob("**/segments.json"))

            for meta_file in found_meta_files:
                panel_folder = meta_file.parent
                rel_parent = panel_folder.relative_to(input_dir)
                panel_id = rel_parent.as_posix().replace("/", "_")
                target_dir = output_dir / rel_parent

                # Locate enhanced / original image
                img_p = panel_folder / "enhanced.png"
                if not img_p.exists():
                    img_p = panel_folder / "original.png"
                if not img_p.exists():
                    alt = Path("data/output/benchmark_enhanced") / rel_parent / "enhanced.png"
                    if alt.exists():
                        img_p = alt

                # Locate lines.svg and tones.svg
                lines_p = panel_folder / "lines.svg"
                if not lines_p.exists():
                    alt_lines = Path("data/output/benchmark_vectorized") / rel_parent / "lines.svg"
                    lines_p = alt_lines if alt_lines.exists() else None

                tones_p = panel_folder / "tones.svg"
                if not tones_p.exists():
                    alt_tones = Path("data/output/benchmark_vectorized") / rel_parent / "tones.svg"
                    tones_p = alt_tones if alt_tones.exists() else None

                if img_p.exists():
                    target_panels.append((panel_id, img_p, meta_file, lines_p, tones_p, target_dir))

        if not target_panels:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No panels with layers_manifest.json or segments.json found in {input_dir}"},
            )

        artifacts: dict[str, Path] = {}
        panel_summaries: dict[str, Any] = {}

        # Load global or external palette if specified
        palette_mgr = PaletteManager.load(external_palette_path) if external_palette_path else PaletteManager()

        for panel_id, img_path, meta_path, lines_svg_p, tones_svg_p, target_dir in target_panels:
            target_dir.mkdir(parents=True, exist_ok=True)
            colored_layers_dir = target_dir / "colored_layers"
            colored_layers_dir.mkdir(parents=True, exist_ok=True)

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            t0 = time.perf_counter()

            # Load metadata (either layers_manifest.json or segments.json)
            meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
            layers_list = meta_data.get("layers", meta_data.get("segments", []))

            # 1. Resolve colors via ColorizerProvider
            for lyr in layers_list:
                lyr.setdefault("character_id", default_char_id)

            layer_colors = self.colorizer.predict((layers_list, palette_mgr))

            # Check for manual color overrides in config
            overrides = col_cfg.get("overrides", {})
            for lid, override_color in overrides.items():
                if lid in layer_colors:
                    layer_colors[lid] = override_color

            # 2. Load masks for raster preview
            masks_dict: dict[str, np.ndarray] = {}
            masks_base = meta_path.parent
            for lyr in layers_list:
                lid = lyr["id"]
                # Check layers/<id>.png first, then masks/<id>*.png
                lyr_png = masks_base / "layers" / f"{lid}.png"
                if lyr_png.exists():
                    rgba = cv2.imread(str(lyr_png), cv2.IMREAD_UNCHANGED)
                    if rgba is not None and rgba.shape[-1] == 4:
                        masks_dict[lid] = rgba[:, :, 3]
                        continue

                m_rel = lyr.get("mask_path")
                if m_rel:
                    full_m = masks_base / m_rel
                    if full_m.exists():
                        mask_gray = cv2.imread(str(full_m), cv2.IMREAD_GRAYSCALE)
                        if mask_gray is not None:
                            masks_dict[lid] = mask_gray

            # 3. Style individual layer SVGs
            for lyr in layers_list:
                lid = lyr["id"]
                color_hex = layer_colors.get(lid, "#E0E0E0")
                in_svg = masks_base / "layers" / f"{lid}.svg"
                if in_svg.exists():
                    out_svg = colored_layers_dir / f"{lid}.svg"
                    style_layer_svg(in_svg, out_svg, color_hex)

            # 4. Assemble master colored vector SVG
            master_svg_path = target_dir / "colored_panel.svg"
            assemble_colored_panel_svg(
                width=w,
                height=h,
                layers=layers_list,
                layer_colors=layer_colors,
                out_svg_path=master_svg_path,
                lines_svg_path=lines_svg_p,
                tones_svg_path=tones_svg_p,
            )
            artifacts[f"{panel_id}/colored_svg"] = master_svg_path

            # 5. Render colored raster image
            # Try to use background_filled.png if available
            bg_plate_path = meta_path.parent / "background_filled.png"
            bg_plate = cv2.imread(str(bg_plate_path)) if bg_plate_path.exists() else None

            colored_bgr = render_colored_panel(
                width=w,
                height=h,
                layers=layers_list,
                layer_colors=layer_colors,
                masks_dict=masks_dict,
                background_plate=bg_plate,
                line_art=img_bgr,
                screentone=None,
            )
            colored_png_path = target_dir / "colored_panel.png"
            cv2.imwrite(str(colored_png_path), colored_bgr)
            artifacts[f"{panel_id}/colored_png"] = colored_png_path

            # 6. Generate side-by-side Before/After comparison
            comparison_img = create_colorize_comparison(img_bgr, colored_bgr)
            comp_path = target_dir / "comparison.png"
            cv2.imwrite(str(comp_path), comparison_img)
            artifacts[f"{panel_id}/comparison"] = comp_path

            # 7. Save palette.json
            palette_out_path = target_dir / "palette.json"
            palette_mgr.save(palette_out_path)
            artifacts[f"{panel_id}/palette"] = palette_out_path

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            panel_summaries[panel_id] = {
                "panel_id": panel_id,
                "colored_layers": len(layer_colors),
                "palette_used": default_char_id,
                "time_ms": elapsed_ms,
            }

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata={
                "panels_processed": len(target_panels),
                "summaries": panel_summaries,
            },
        )
