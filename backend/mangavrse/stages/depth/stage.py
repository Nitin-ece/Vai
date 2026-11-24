"""
Depth Stage: estimates monocular depth with Depth Anything / MiDaS,
converts pixel depth to per-layer median depth normalized to [0.0, 1.0],
applies manga semantic priors, manages manual overrides, and renders
layer order visualizations that feed directly into scene.json.
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

from .depth_extractor import build_scene_layers, extract_layer_depths
from .depth_provider import DepthEstimatorProvider
from .visualizer import render_depth_map, render_depth_visualization

logger = logging.getLogger(__name__)


class DepthStage(AbstractStage):
    """
    Depth stage producing depth_map.png, depth.json, and depth_visualization.png.
    """

    name = "depth"

    def __init__(self, depth_estimator: DepthEstimatorProvider | None = None) -> None:
        super().__init__(name="depth")
        self.depth_estimator = depth_estimator or DepthEstimatorProvider()

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        start_time = time.perf_counter()
        depth_cfg = config.get("depth", {})
        models_cfg = config.get("models", {}).get("depth_estimator", {})

        # Load weights or model if specified
        model_name = models_cfg.get("name", "depth_anything_v2")
        weights = models_cfg.get("weights")
        device = "cuda" if config.get("device", {}).get("prefer_cuda", False) else "cpu"

        if self.depth_estimator.name != model_name:
            self.depth_estimator = DepthEstimatorProvider(name=model_name)
        self.depth_estimator.load(device=device, weights=weights)

        # Discover target panels
        # Can process subdirectories (benchmark suite) or a single panel directory
        target_panels: list[tuple[str, Path, Path, Path]] = []

        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        has_sub_panels = any(
            (p / "layers_manifest.json").exists() or (p / "segments.json").exists()
            for p in subdirs
        )

        if has_sub_panels:
            for subdir in sorted(subdirs):
                panel_id = subdir.name
                # Source image: enhanced.png or background_filled.png or fallback
                img_path = subdir / "enhanced.png"
                if not img_path.exists():
                    img_path = subdir / "background_filled.png"
                if not img_path.exists():
                    pngs = list(subdir.glob("*.png"))
                    if pngs:
                        img_path = pngs[0]

                manifest_path = subdir / "layers_manifest.json"
                if not manifest_path.exists():
                    manifest_path = subdir / "segments.json"

                if img_path.exists() and manifest_path.exists():
                    target_panels.append((panel_id, subdir, img_path, manifest_path))
        else:
            # Single panel
            panel_id = input_dir.name
            img_path = input_dir / "enhanced.png"
            if not img_path.exists():
                img_path = input_dir / "background_filled.png"
            if not img_path.exists():
                pngs = list(input_dir.glob("*.png"))
                if pngs:
                    img_path = pngs[0]

            manifest_path = input_dir / "layers_manifest.json"
            if not manifest_path.exists():
                manifest_path = input_dir / "segments.json"

            if img_path.exists() and manifest_path.exists():
                target_panels.append((panel_id, input_dir, img_path, manifest_path))

        if not target_panels:
            logger.warning(f"No valid panels found in {input_dir}")
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": "No valid panels with image and layers/segments manifest found."},
            )

        artifacts: dict[str, str] = {}
        panel_summaries: list[dict[str, Any]] = []

        for panel_id, p_in_dir, img_path, manifest_path in target_panels:
            p_out_dir = output_dir / panel_id if has_sub_panels else output_dir
            p_out_dir.mkdir(parents=True, exist_ok=True)

            # Load image
            source_img = cv2.imread(str(img_path))
            if source_img is None:
                logger.error(f"Failed to read image at {img_path}")
                continue

            h, w = source_img.shape[:2]

            # 1. Run depth estimation
            preprocessed = self.depth_estimator.preprocess(source_img)
            raw_depth = self.depth_estimator.predict(preprocessed)
            depth_map = self.depth_estimator.postprocess(raw_depth)

            # 2. Save depth_map.png
            depth_map_path = p_out_dir / "depth_map.png"
            render_depth_map(depth_map, depth_map_path)

            # 3. Read layer / segment manifest
            layers_data: list[dict[str, Any]] = []
            try:
                manifest_content = json.loads(manifest_path.read_text(encoding="utf-8"))
                if "layers" in manifest_content and isinstance(manifest_content["layers"], list):
                    layers_data = manifest_content["layers"]
                elif "segments" in manifest_content and isinstance(manifest_content["segments"], list):
                    layers_data = manifest_content["segments"]
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_path}: {e}")

            # Check existing depth.json for user manual overrides
            existing_depth_json = p_out_dir / "depth.json"
            overrides: dict[str, float] = {}
            if existing_depth_json.exists():
                try:
                    prior_depth_data = json.loads(existing_depth_json.read_text(encoding="utf-8"))
                    overrides = prior_depth_data.get("overrides", {})
                except Exception:
                    pass

            # 4. Extract layer depths and sort z-order
            processed_layers = extract_layer_depths(
                depth_map=depth_map,
                layers=layers_data,
                panel_dir=p_in_dir,
                config=depth_cfg,
                overrides=overrides,
            )

            scene_layers = build_scene_layers(processed_layers)

            # 5. Write depth.json
            depth_output = {
                "panel_id": panel_id,
                "width": w,
                "height": h,
                "source": img_path.name,
                "depth_map": "depth_map.png",
                "visualization": "depth_visualization.png",
                "total_layers": len(processed_layers),
                "overrides": overrides,
                "layers": processed_layers,
                "scene_layers": scene_layers,
            }

            depth_json_path = p_out_dir / "depth.json"
            depth_json_path.write_text(json.dumps(depth_output, indent=2), encoding="utf-8")

            # 6. Render layer order visualization
            viz_path = p_out_dir / "depth_visualization.png"
            render_depth_visualization(
                depth_map=depth_map,
                layers=processed_layers,
                output_path=viz_path,
                panel_dir=p_in_dir,
                source_image=source_img,
            )

            # Record artifacts
            prefix = f"{panel_id}/" if has_sub_panels else ""
            artifacts[f"{prefix}depth_map"] = str(depth_map_path)
            artifacts[f"{prefix}depth_json"] = str(depth_json_path)
            artifacts[f"{prefix}visualization"] = str(viz_path)

            panel_summaries.append({
                "panel_id": panel_id,
                "total_layers": len(processed_layers),
                "overrides_count": len(overrides),
                "depth_range": [
                    min((l["depth"] for l in processed_layers), default=0.0),
                    max((l["depth"] for l in processed_layers), default=1.0),
                ],
            })

        elapsed = time.perf_counter() - start_time
        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            elapsed_seconds=elapsed,
            metadata={"panels": panel_summaries},
        )
