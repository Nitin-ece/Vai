"""
Vectorize Stage — orchestrates line art extraction, screentone separation,
Potrace vector tracing, Bezier optimization, SVG output, and fidelity benchmarking.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mangavrse.core.stage import AbstractStage, StageResult

from .extractor import extract_lines_and_tones
from .fidelity import compute_fidelity_score
from .optimizer import PathOptimizer
from .preview import generate_preview_4x
from .svg_writer import SVGWriter
from .tracer import PotraceVectorProvider


class VectorizeStage(AbstractStage):
    """Traces enhanced panel images into clean SVG line art and screentone fills."""

    def __init__(self, tracer: PotraceVectorProvider | None = None) -> None:
        super().__init__(name="vectorize")
        self.tracer = tracer or PotraceVectorProvider()
        self.optimizer = PathOptimizer()

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        artifacts: dict[str, Path] = {}
        metadata: dict[str, Any] = {
            "panels_processed": 0,
            "stats": {},
        }

        v_cfg = config.get("vectorize", config)
        extractor_cfg = v_cfg.get("extractor", {})
        tracer_cfg = v_cfg.get("tracer", {})
        opt_cfg = v_cfg.get("optimizer", {})
        tone_cfg = v_cfg.get("screentone", {})

        # Configure tracer and optimizer parameters
        self.tracer.turdsize = int(tracer_cfg.get("turdsize", 2))
        self.tracer.alphamax = float(tracer_cfg.get("alphamax", 1.0))
        self.tracer.opttolerance = float(tracer_cfg.get("opttolerance", 0.8))

        self.optimizer.min_path_area = float(opt_cfg.get("min_path_area", 12.0))
        self.optimizer.rdp_tolerance = float(opt_cfg.get("rdp_tolerance", 1.2))
        self.optimizer.smooth_bezier = bool(opt_cfg.get("smooth_bezier", True))
        self.optimizer.merge_distance = float(opt_cfg.get("merge_distance", 2.0))

        # Discover panel images (prefer enhanced.png, fallback to original.png)
        target_images: list[tuple[str, Path, Path]] = []

        if (input_dir / "enhanced.png").exists():
            target_images.append(("panel", input_dir / "enhanced.png", output_dir))
        elif (input_dir / "original.png").exists():
            target_images.append(("panel", input_dir / "original.png", output_dir))
        else:
            # Check for subdirectories
            for enh_file in sorted(input_dir.glob("**/enhanced.png")):
                panel_id = enh_file.parent.name
                rel_parent = enh_file.parent.relative_to(input_dir)
                target_dir = output_dir / rel_parent
                target_images.append((panel_id, enh_file, target_dir))

            if not target_images:
                for orig_file in sorted(input_dir.glob("**/original.png")):
                    panel_id = orig_file.parent.name
                    rel_parent = orig_file.parent.relative_to(input_dir)
                    target_dir = output_dir / rel_parent
                    target_images.append((panel_id, orig_file, target_dir))

        if not target_images:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No enhanced.png or original.png found in {input_dir}"},
            )

        # Process each panel
        for panel_id, img_path, target_dir in target_images:
            target_dir.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            t0 = time.perf_counter()

            # 1. Line art and screentone separation
            line_mask, tone_mask, tone_density = extract_lines_and_tones(
                image_bgr=img_bgr,
                adaptive_block_size=int(extractor_cfg.get("adaptive_block_size", 15)),
                adaptive_c=int(extractor_cfg.get("adaptive_c", 8)),
                screentone_separation=bool(extractor_cfg.get("screentone_separation", True)),
                min_tone_frequency=float(extractor_cfg.get("min_tone_frequency", 0.12)),
            )

            # 2. Vector tracing
            raw_line_paths = self.tracer.predict(line_mask > 127)
            raw_tone_paths = self.tracer.predict(tone_mask > 127) if np.any(tone_mask > 127) else []

            # 3. Optimization (speck removal, Bezier smoothing, point reduction)
            opt_line_paths = self.optimizer.optimize(raw_line_paths)
            opt_tone_paths = self.optimizer.optimize(raw_tone_paths)

            # 4. Write lines.svg
            lines_svg_path = target_dir / "lines.svg"
            line_path_count, line_node_count, line_size_bytes = SVGWriter.write_lines_svg(
                paths=opt_line_paths,
                width=w,
                height=h,
                out_path=lines_svg_path,
            )
            artifacts[f"{panel_id}/lines_svg"] = lines_svg_path

            # 5. Write tones.svg
            tones_svg_path = target_dir / "tones.svg"
            tone_path_count, tone_node_count, tone_size_bytes = SVGWriter.write_tones_svg(
                tone_paths=opt_tone_paths,
                width=w,
                height=h,
                out_path=tones_svg_path,
                dot_density=tone_density,
                dot_radius=float(tone_cfg.get("default_dot_size", 2.0)),
                dot_spacing=float(tone_cfg.get("default_spacing", 5.0)),
            )
            artifacts[f"{panel_id}/tones_svg"] = tones_svg_path

            # 6. Compute objective fidelity score against source line mask
            fidelity_scores = compute_fidelity_score(
                source_mask=line_mask,
                paths=opt_line_paths,
                width=w,
                height=h,
            )

            # 7. Generate 4x zoomed preview comparison
            preview_img = generate_preview_4x(
                source_bgr=img_bgr,
                paths=opt_line_paths,
                crop_size=160,
                zoom_scale=4,
            )
            preview_path = target_dir / "preview_4x.png"
            cv2.imwrite(str(preview_path), preview_img)
            artifacts[f"{panel_id}/preview_4x"] = preview_path

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            # 8. Write stats.json
            panel_stats = {
                "panel_id": panel_id,
                "resolution": [w, h],
                "lines": {
                    "path_count": line_path_count,
                    "node_count": line_node_count,
                    "file_size_bytes": line_size_bytes,
                },
                "tones": {
                    "path_count": tone_path_count,
                    "node_count": tone_node_count,
                    "file_size_bytes": tone_size_bytes,
                    "dot_density": tone_density,
                },
                "total_svg_size_bytes": line_size_bytes + tone_size_bytes,
                "fidelity": fidelity_scores,
                "time_ms": elapsed_ms,
            }

            stats_path = target_dir / "stats.json"
            stats_path.write_text(json.dumps(panel_stats, indent=2), encoding="utf-8")
            artifacts[f"{panel_id}/stats"] = stats_path

            metadata["panels_processed"] += 1
            metadata["stats"][panel_id] = panel_stats

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata=metadata,
        )
