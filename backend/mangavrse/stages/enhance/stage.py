"""
Enhance Stage — orchestrates panel enhancement, comparison rendering, and metrics persistence.

Inputs:
  A directory containing a panel's `original.png` (or multiple panel folders).

Outputs:
  enhanced.png: upscaled, denoised, sharpened panel
  comparison.png: high-resolution side-by-side comparison
  metrics.json: resolution, sharpness score (Laplacian, Tenengrad), step timings
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from mangavrse.core.stage import AbstractStage, StageResult
from mangavrse.models.realesrgan import RealESRGANAnimeProvider

from .comparison import generate_comparison_image
from .pipeline import EnhancePipeline


class EnhanceStage(AbstractStage):
    """Enhances manga panels: denoise, contrast correction, deblur, upscale, sharpen."""

    def __init__(self, model_provider: RealESRGANAnimeProvider | None = None) -> None:
        super().__init__(name="enhance")
        self.provider = model_provider or RealESRGANAnimeProvider()
        self.pipeline = EnhancePipeline(model_provider=self.provider)

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        artifacts: dict[str, Path] = {}
        metadata: dict[str, Any] = {
            "panels_processed": 0,
            "metrics": {},
        }

        # Find target panel images in input_dir
        # Supports either:
        # a) input_dir directly contains 'original.png'
        # b) input_dir contains panel subdirectories (data/<project>/<page>/<panel>/original.png)
        target_images: list[tuple[str, Path, Path]] = []  # (panel_id, src_path, target_dir)

        if (input_dir / "original.png").exists():
            target_images.append(("panel", input_dir / "original.png", output_dir))
        else:
            # Check for subdirectories
            for panel_file in sorted(input_dir.glob("**/original.png")):
                panel_id = panel_file.parent.name
                rel_parent = panel_file.parent.relative_to(input_dir)
                target_dir = output_dir / rel_parent
                target_images.append((panel_id, panel_file, target_dir))

        # Also fallback to any image file if no original.png is found
        if not target_images:
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for img_file in sorted(input_dir.glob(ext)):
                    if not img_file.name.startswith(("debug_", "comparison", "enhanced")):
                        target_images.append((img_file.stem, img_file, output_dir))
                        break

        if not target_images:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No input panel images found in {input_dir}"},
            )

        # Process each panel image
        for panel_id, src_img_path, target_dir in target_images:
            target_dir.mkdir(parents=True, exist_ok=True)
            img_bgr = cv2.imread(str(src_img_path))
            if img_bgr is None:
                continue

            result = self.pipeline.run(img_bgr, config=config)

            # 1. Save enhanced.png
            enhanced_path = target_dir / "enhanced.png"
            cv2.imwrite(str(enhanced_path), result.enhanced_image)
            artifacts[f"{panel_id}/enhanced"] = enhanced_path

            # 2. Save comparison.png
            comp_img = generate_comparison_image(
                original_bgr=img_bgr,
                enhanced_bgr=result.enhanced_image,
                title_original=f"ORIGINAL ({panel_id})",
                title_enhanced=f"ENHANCED ({result.enhanced_metrics['resolution'][0]}x{result.enhanced_metrics['resolution'][1]})",
            )
            comparison_path = target_dir / "comparison.png"
            cv2.imwrite(str(comparison_path), comp_img)
            artifacts[f"{panel_id}/comparison"] = comparison_path

            # 3. Save metrics.json
            metrics_data = {
                "panel_id": panel_id,
                "applied_steps": result.applied_steps,
                "step_timings_ms": result.step_timings_ms,
                "total_time_ms": round(sum(result.step_timings_ms.values()), 2),
                "original": result.original_metrics,
                "enhanced": result.enhanced_metrics,
            }
            metrics_path = target_dir / "metrics.json"
            metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")
            artifacts[f"{panel_id}/metrics"] = metrics_path

            metadata["panels_processed"] += 1
            metadata["metrics"][panel_id] = metrics_data

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata=metadata,
        )
