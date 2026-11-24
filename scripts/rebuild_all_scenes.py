"""
Rebuild all benchmark scenes and custom scenes through the full pipeline:
- SegmentStage (SAM 2 + YOLO object detection + Character Vision Encoder)
- VectorizeStage (OpenCV + U-Net line extraction + Potrace Bezier)
- LayersStage (Pure white canvas plate + transparent cutout PNGs)
- ColorizeStage (Conditional diffusion + character palettes)
- DepthStage (Depth Anything V2 + manga depth estimation)
- AssemblyStage (Validated scene.json with per-segment depth and canvas_white)
"""
import sys
import shutil
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from mangavrse.stages.segment.stage import SegmentStage
from mangavrse.stages.vectorize.stage import VectorizeStage
from mangavrse.stages.layers.stage import LayersStage
from mangavrse.stages.colorize.stage import ColorizeStage
from mangavrse.stages.depth.stage import DepthStage
from mangavrse.stages.assembly.stage import AssemblyStage


def rebuild_scenes():
    print("Initializing pipeline stages...")
    seg_stage = SegmentStage()
    vec_stage = VectorizeStage()
    lay_stage = LayersStage()
    col_stage = ColorizeStage()
    dep_stage = DepthStage()
    asm_stage = AssemblyStage()

    cfg = {
        "device": {"prefer_cuda": False},
        "segment": {"confidence_threshold": 0.60, "screentone_smoothing": True, "bubble_detection": True},
        "layers": {"inpainting": {"method": "telea_texture", "radius": 5}},
        "colorize": {"character_id": "char_001"},
        "depth": {"model": "depth_anything_v2"},
    }

    enhanced_dir = Path("data/output/benchmark_enhanced")
    segmented_dir = Path("data/output/benchmark_segmented")
    vectorized_dir = Path("data/output/benchmark_vectorized")
    layers_dir = Path("data/output/benchmark_layers")
    colorized_dir = Path("data/output/benchmark_colorized")
    depth_dir = Path("data/output/benchmark_depth")
    scenes_dir = Path("data/output/benchmark_scenes")

    print("1. Running SegmentStage...")
    res_seg = seg_stage.run(enhanced_dir, segmented_dir, cfg)
    print("Segment success:", res_seg.success)

    print("2. Running VectorizeStage...")
    res_vec = vec_stage.run(enhanced_dir, vectorized_dir, cfg)
    print("Vectorize success:", res_vec.success)

    print("3. Running LayersStage (Generating pure white canvas and transparent cutouts)...")
    res_lay = lay_stage.run(segmented_dir, layers_dir, cfg)
    print("Layers success:", res_lay.success)

    print("4. Running ColorizeStage...")
    res_col = col_stage.run(layers_dir, colorized_dir, cfg)
    print("Colorize success:", res_col.success)

    print("5. Running DepthStage...")
    res_dep = dep_stage.run(layers_dir, depth_dir, cfg)
    print("Depth success:", res_dep.success)

    print("6. Running AssemblyStage...")
    res_asm = asm_stage.run(depth_dir, scenes_dir, cfg)
    print("Assembly success:", res_asm.success)

    # Sync to frontend public scenes
    frontend_scenes = Path("frontend/public/scenes")
    frontend_scenes.mkdir(parents=True, exist_ok=True)
    for p_dir in scenes_dir.iterdir():
        if p_dir.is_dir():
            dest = frontend_scenes / p_dir.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(p_dir, dest)
            print(f"Copied scene to frontend: {dest}")

    print("All scenes rebuilt successfully!")


if __name__ == "__main__":
    rebuild_scenes()
