"""
Assembly Stage: combines layers, colors, and depth into a validated scene.json.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from mangavrse.core.stage import AbstractStage, StageResult

from .assembler import assemble_scene, validate_scene

logger = logging.getLogger(__name__)


class AssemblyStage(AbstractStage):
    """
    Assembly stage producing validated scene.json files.
    """

    name = "assembly"

    def __init__(self, schema_path: Path | None = None) -> None:
        super().__init__(name="assembly")
        self.schema_path = schema_path

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        start_time = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Discover panels
        target_panels: list[tuple[str, Path]] = []
        subdirs = [p for p in input_dir.iterdir() if p.is_dir()]
        has_sub_panels = any(
            (p / "depth.json").exists()
            or (p / "layers_manifest.json").exists()
            or (p / "scene.json").exists()
            for p in subdirs
        )

        if has_sub_panels:
            for subdir in sorted(subdirs):
                target_panels.append((subdir.name, subdir))
        else:
            target_panels.append((input_dir.name, input_dir))

        artifacts: dict[str, str] = {}
        panel_summaries: list[dict[str, Any]] = []

        for panel_id, p_in_dir in target_panels:
            p_out_dir = output_dir / panel_id if has_sub_panels else output_dir
            p_out_dir.mkdir(parents=True, exist_ok=True)

            # Check cross-stage references in data/output if p_in_dir lacks depth or colorize
            depth_data = None
            layers_data = None
            palette_data = None

            # 1. Depth data
            depth_file = p_in_dir / "depth.json"
            if not depth_file.exists():
                alt_depth = Path("data/output/benchmark_depth") / panel_id / "depth.json"
                if alt_depth.exists():
                    depth_file = alt_depth
            if depth_file.exists():
                try:
                    depth_data = json.loads(depth_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # 2. Layers manifest
            layers_file = p_in_dir / "layers_manifest.json"
            if not layers_file.exists():
                alt_layers = Path("data/output/benchmark_layers") / panel_id / "layers_manifest.json"
                if alt_layers.exists():
                    layers_file = alt_layers
            if layers_file.exists():
                try:
                    man = json.loads(layers_file.read_text(encoding="utf-8"))
                    layers_data = man.get("layers", [])
                except Exception:
                    pass

            # 3. Palette data
            pal_file = p_in_dir / "palette.json"
            if not pal_file.exists():
                alt_pal = Path("data/output/benchmark_colorized") / panel_id / "palette.json"
                if alt_pal.exists():
                    pal_file = alt_pal
            if pal_file.exists():
                try:
                    palette_data = json.loads(pal_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # Assemble scene
            scene = assemble_scene(
                panel_id=panel_id,
                panel_dir=p_in_dir,
                layers_data=layers_data,
                palette_data=palette_data,
                depth_data=depth_data,
                schema_path=self.schema_path,
            )

            # Write scene.json
            scene_path = p_out_dir / "scene.json"
            scene_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")

            # Also copy layer PNGs/SVGs and background plate if present
            # so the assembled output directory can be statically served as a standalone bundle
            for candidate_dir in [
                p_in_dir,
                Path("data/output/benchmark_layers") / panel_id,
                Path("data/output/benchmark_colorized") / panel_id,
                Path("data/output/benchmark_depth") / panel_id,
            ]:
                if not candidate_dir.exists():
                    continue
                # Copy layers folder
                src_layers = candidate_dir / "layers"
                if src_layers.exists():
                    dst_layers = p_out_dir / "layers"
                    dst_layers.mkdir(exist_ok=True)
                    for f in src_layers.glob("*.*"):
                        target_f = dst_layers / f.name
                        if not target_f.exists():
                            target_f.write_bytes(f.read_bytes())
                # Copy colored_layers
                src_col = candidate_dir / "colored_layers"
                if src_col.exists():
                    dst_col = p_out_dir / "colored_layers"
                    dst_col.mkdir(exist_ok=True)
                    for f in src_col.glob("*.*"):
                        target_f = dst_col / f.name
                        if not target_f.exists():
                            target_f.write_bytes(f.read_bytes())
                # Copy images
                for img_name in ["enhanced.png", "background_filled.png", "depth_map.png", "depth_visualization.png"]:
                    src_img = candidate_dir / img_name
                    dst_img = p_out_dir / img_name
                    if src_img.exists() and not dst_img.exists():
                        dst_img.write_bytes(src_img.read_bytes())

            prefix = f"{panel_id}/" if has_sub_panels else ""
            artifacts[f"{prefix}scene_json"] = str(scene_path)
            panel_summaries.append({
                "panel_id": panel_id,
                "layers_count": len(scene["layers"]),
                "camera_limits": scene["camera"]["limits"],
            })

        elapsed = time.perf_counter() - start_time
        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            elapsed_seconds=elapsed,
            metadata={"panels": panel_summaries},
        )
