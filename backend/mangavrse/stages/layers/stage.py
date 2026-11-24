"""
Layers Stage: cuts segmented panel elements into transparent RGBA PNGs and SVGs,
inpaints occluded background regions, and generates parallax-gap diagnostic renders.
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

from .cutter import create_clipped_svg, extract_layer_rgba
from .inpainter import InpainterProvider
from .parallax_tester import render_parallax_test
from .quality import evaluate_inpaint_quality

logger = logging.getLogger(__name__)


class LayersStage(AbstractStage):
    """
    Layers stage producing transparent cutouts, infilled background plate,
    layers manifest, and parallax-gap test renders.
    """

    name = "layers"

    def __init__(self, inpainter: InpainterProvider | None = None) -> None:
        super().__init__(name="layers")
        self.inpainter = inpainter or InpainterProvider()

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        lyr_cfg = config.get("layers", {})
        mask_padding = int(lyr_cfg.get("mask_padding", 2))
        feather_radius = float(lyr_cfg.get("feather_radius", 2.0))

        inpaint_cfg = lyr_cfg.get("inpainting", {})
        method = str(inpaint_cfg.get("method", "telea_texture"))
        radius = int(inpaint_cfg.get("radius", 5))
        line_ext = bool(inpaint_cfg.get("line_extension", True))
        tone_syn = bool(inpaint_cfg.get("screentone_synthesis", True))

        self.inpainter.method = method
        self.inpainter.inpaint_radius = radius
        self.inpainter.line_extension = line_ext
        self.inpainter.screentone_synthesis = tone_syn

        parallax_cfg = lyr_cfg.get("parallax_test", {})
        shift_scale = float(parallax_cfg.get("shift_scale", 25.0))
        angle_degrees = float(parallax_cfg.get("angle_degrees", -30.0))

        # Discover target panels
        target_panels: list[tuple[str, Path, Path, Path | None, Path]] = []

        direct_enhanced = input_dir / "enhanced.png"
        direct_original = input_dir / "original.png"
        direct_segments = input_dir / "segments.json"

        if (direct_enhanced.exists() or direct_original.exists()) and direct_segments.exists():
            img_p = direct_enhanced if direct_enhanced.exists() else direct_original
            svg_p = input_dir / "lines.svg" if (input_dir / "lines.svg").exists() else None
            target_panels.append(("panel_000", img_p, direct_segments, svg_p, output_dir))
        else:
            for seg_json in sorted(input_dir.glob("**/segments.json")):
                panel_folder = seg_json.parent
                rel_parent = panel_folder.relative_to(input_dir)
                panel_id = rel_parent.as_posix().replace("/", "_")
                target_dir = output_dir / rel_parent

                img_p = panel_folder / "enhanced.png"
                if not img_p.exists():
                    img_p = panel_folder / "original.png"
                if not img_p.exists():
                    # Check matching enhanced dir in data/output/benchmark_enhanced
                    alt_enhanced = Path("data/output/benchmark_enhanced") / rel_parent / "enhanced.png"
                    if alt_enhanced.exists():
                        img_p = alt_enhanced

                svg_p = panel_folder / "lines.svg"
                if not svg_p.exists():
                    alt_svg = Path("data/output/benchmark_vectorized") / rel_parent / "lines.svg"
                    if alt_svg.exists():
                        svg_p = alt_svg
                    else:
                        svg_p = None

                if img_p.exists():
                    target_panels.append((panel_id, img_p, seg_json, svg_p, target_dir))

        if not target_panels:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No panels with segments.json found in {input_dir}"},
            )

        artifacts: dict[str, Path] = {}
        panel_summaries: dict[str, Any] = {}

        for panel_id, img_path, seg_path, svg_path, target_dir in target_panels:
            target_dir.mkdir(parents=True, exist_ok=True)
            layers_dir = target_dir / "layers"
            layers_dir.mkdir(parents=True, exist_ok=True)

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            t0 = time.perf_counter()

            seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
            segments_list = seg_data.get("segments", [])

            # 1. Load masks and construct union occlusion mask for background inpainting
            masks_base = seg_path.parent
            occlusion_mask = np.zeros((h, w), dtype=np.uint8)

            extracted_layers: list[dict[str, Any]] = []

            for s in segments_list:
                sid = s["id"]
                label = s.get("label", "character")
                part = s.get("part")
                m_path_rel = s.get("mask_path")
                if not m_path_rel:
                    continue

                full_m_path = masks_base / m_path_rel
                if not full_m_path.exists():
                    continue

                mask_gray = cv2.imread(str(full_m_path), cv2.IMREAD_GRAYSCALE)
                if mask_gray is None:
                    continue

                # Skip any full-canvas background mask (>95% area) so the old image is not duplicated
                m_count = int(np.count_nonzero(mask_gray > 127))
                if m_count >= int(h * w * 0.95):
                    continue

                # Add non-background items to occlusion mask
                if label not in ("background", "environment"):
                    occlusion_mask = cv2.bitwise_or(occlusion_mask, (mask_gray > 127).astype(np.uint8) * 255)

                # Extract transparent RGBA cutout
                rgba = extract_layer_rgba(
                    image_bgr=img_bgr,
                    mask=mask_gray,
                    padding=mask_padding,
                    feather_radius=feather_radius,
                )
                png_name = f"{sid}.png"
                png_path = layers_dir / png_name
                cv2.imwrite(str(png_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

                # Extract matching SVG
                svg_name = f"{sid}.svg"
                svg_layer_path = layers_dir / svg_name
                create_clipped_svg(
                    svg_source_path=svg_path,
                    mask=mask_gray,
                    out_svg_path=svg_layer_path,
                    width=w,
                    height=h,
                    layer_id=sid,
                )

                extracted_layers.append({
                    "id": sid,
                    "label": label,
                    "part": part,
                    "parentId": s.get("parentId"),
                    "suggested_depth": s.get("suggested_depth", 0.5),
                    "suggested_z_order": s.get("suggested_z_order", 0),
                    "bbox": s.get("bbox", [0, 0, w, h]),
                    "area": s.get("area", 0),
                    "pngPath": f"layers/{png_name}",
                    "svgPath": f"layers/{svg_name}",
                    "rgba": rgba,
                })

            # 2. Blank white manga paper canvas with clean outer panel border
            # (Completely removes old background image so segments sit cleanly on blank paper)
            bg_filled = np.full((h, w, 3), 255, dtype=np.uint8)
            border_t = max(2, int(min(h, w) * 0.003))
            cv2.rectangle(bg_filled, (border_t, border_t), (w - border_t, h - border_t), (0, 0, 0), border_t)

            bg_path = target_dir / "background_filled.png"
            cv2.imwrite(str(bg_path), bg_filled)
            artifacts[f"{panel_id}/background_filled"] = bg_path

            # Also create canvas_white.png for explicit white background layer in Three.js
            canvas_white_p = layers_dir / "canvas_white.png"
            cv2.imwrite(str(canvas_white_p), bg_filled)

            # 3. Evaluate inpaint quality along seams
            quality_metrics = evaluate_inpaint_quality(img_bgr, bg_filled, occlusion_mask)

            # 4. Parallax gap test render
            parallax_img = render_parallax_test(
                background_bgr=bg_filled,
                layers=extracted_layers,
                shift_scale=shift_scale,
                angle_degrees=angle_degrees,
            )
            parallax_path = target_dir / "parallax_test.png"
            cv2.imwrite(str(parallax_path), parallax_img)
            artifacts[f"{panel_id}/parallax_test"] = parallax_path

            # 5. Layers manifest JSON
            manifest_layers: list[dict[str, Any]] = []
            for lyr in extracted_layers:
                manifest_layers.append({
                    "id": lyr["id"],
                    "label": lyr["label"],
                    "part": lyr["part"],
                    "parentId": lyr["parentId"],
                    "depth": lyr["suggested_depth"],
                    "zOrder": lyr["suggested_z_order"],
                    "bbox": lyr["bbox"],
                    "pngPath": lyr["pngPath"],
                    "svgPath": lyr["svgPath"],
                })

            manifest_payload = {
                "panel": {
                    "panel_id": panel_id,
                    "width": w,
                    "height": h,
                    "source": str(img_path.name),
                },
                "total_layers": len(manifest_layers),
                "inpaint_metrics": quality_metrics,
                "background_plate": "background_filled.png",
                "parallax_test": "parallax_test.png",
                "layers": manifest_layers,
            }
            manifest_path = target_dir / "layers_manifest.json"
            manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
            artifacts[f"{panel_id}/manifest"] = manifest_path

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            panel_summaries[panel_id] = {
                "panel_id": panel_id,
                "layer_count": len(manifest_layers),
                "inpaint_quality": quality_metrics["inpaint_quality"],
                "seam_disparity": quality_metrics["seam_disparity"],
                "needs_manual_review": quality_metrics["needs_manual_review"],
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
