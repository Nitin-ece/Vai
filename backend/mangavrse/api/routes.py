"""
Stage-execution API routes.

Each registered stage gets a POST endpoint at:
    /api/v1/stages/{stage_name}/run
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging
import os
import re
import shutil
import time

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from mangavrse.core.config import load_config
from mangavrse.core.stage import AbstractStage, StageResult
from mangavrse.stages.assembly import AssemblyStage
from mangavrse.stages.colorize import ColorizeStage
from mangavrse.stages.depth import DepthStage, apply_depth_override, render_depth_visualization
from mangavrse.stages.enhance import EnhanceStage
from mangavrse.stages.ingest import IngestStage
from mangavrse.stages.layers import LayersStage, inpaint_manga_background
from mangavrse.stages.noop import NoOpStage
from mangavrse.stages.segment import SegmentStage, refine_mask_with_prompts
from mangavrse.stages.vectorize import VectorizeStage

router = APIRouter(tags=["stages"])

# ----- Stage registry (add new stages here) -----------------------
_STAGES: dict[str, AbstractStage] = {
    "noop": NoOpStage(),
    "ingest": IngestStage(),
    "enhance": EnhanceStage(),
    "vectorize": VectorizeStage(),
    "segment": SegmentStage(),
    "layers": LayersStage(),
    "colorize": ColorizeStage(),
    "depth": DepthStage(),
    "assembly": AssemblyStage(),
}


# ----- Request / Response models ----------------------------------

class StageRunRequest(BaseModel):
    input_dir: str
    output_dir: str
    config_overrides: dict[str, Any] = {}


class StageRunResponse(BaseModel):
    stage: str
    success: bool
    cached: bool
    elapsed_seconds: float
    artifacts: dict[str, str]
    metadata: dict[str, Any]


class SegmentRefineRequest(BaseModel):
    image_path: str
    points: list[dict[str, Any]] = []      # [{"x": float, "y": float, "label": int}]
    box: list[float] | None = None          # [x0, y0, x1, y1]
    existing_mask_path: str | None = None
    output_mask_path: str | None = None


class SegmentRefineResponse(BaseModel):
    success: bool
    mask_path: str
    mask_base64: str = ""
    mask_url: str = ""
    confidence: float
    bbox: list[int]
    area: int


class InpaintRegionRequest(BaseModel):
    image_path: str
    box: list[int] | None = None          # [x0, y0, x1, y1]
    mask_path: str | None = None
    output_path: str | None = None
    method: str = "telea_texture"
    radius: int = 5


class InpaintRegionResponse(BaseModel):
    success: bool
    output_path: str
    metrics: dict[str, Any]


class ColorizeOverrideRequest(BaseModel):
    panel_dir: str
    layer_id: str
    color: str
    opacity: float = 1.0


class ColorizeOverrideResponse(BaseModel):
    success: bool
    layer_id: str
    color: str
    layer_svg_path: str
    colored_panel_svg_path: str


class DepthOverrideRequest(BaseModel):
    panel_dir: str
    layer_id: str
    depth: float


class DepthOverrideResponse(BaseModel):
    success: bool
    layer_id: str
    depth: float
    z_order: int
    depth_json_path: str
    visualization_path: str


# ----- Endpoints ---------------------------------------------------

@router.get("/stages")
async def list_stages() -> list[str]:
    """Return names of all registered stages."""
    return sorted(_STAGES.keys())


@router.post("/stages/{stage_name}/run", response_model=StageRunResponse)
async def run_stage(stage_name: str, body: StageRunRequest) -> StageRunResponse:
    """Execute a stage by name."""
    stage = _STAGES.get(stage_name)
    if stage is None:
        raise HTTPException(status_code=404, detail=f"Stage '{stage_name}' not found")

    cfg = load_config()
    merged_config = {**cfg.stage_config(stage_name), **body.config_overrides}

    input_path = Path(body.input_dir)
    output_path = Path(body.output_dir)

    if not input_path.exists():
        input_path.mkdir(parents=True, exist_ok=True)

    result: StageResult = stage.execute(input_path, output_path, merged_config)

    return StageRunResponse(
        stage=stage_name,
        success=result.success,
        cached=result.cached,
        elapsed_seconds=result.elapsed_seconds,
        artifacts={k: str(v) for k, v in result.artifacts.items()},
        metadata=result.metadata,
    )


@router.post("/stages/segment/refine", response_model=SegmentRefineResponse)
async def refine_segment(body: SegmentRefineRequest) -> SegmentRefineResponse:
    """
    Interactive studio endpoint: refines or creates a mask based on user click points
    (positive +1, negative 0) or bounding box prompts.
    """
    import base64
    import cv2

    img_p = Path(body.image_path)
    if not img_p.exists():
        # Dynamically locate image across project and benchmark trees
        stem = img_p.parent.name if img_p.parent.name not in (".", "") else img_p.stem
        cands = [
            Path(f"data/output/scenes/{stem}/enhanced.png"),
            Path(f"data/output/scenes/{stem}/original.png"),
            Path(f"data/output/benchmark_enhanced/{stem}/enhanced.png"),
            Path(f"data/benchmark_panels/{stem}/original.png"),
        ]
        found = False
        for c in cands:
            if c.exists():
                img_p = c
                found = True
                break
        if not found:
            for match in Path("data/output").glob(f"**/{stem}/enhanced.png"):
                img_p = match
                found = True
                break
            if not found:
                for match in Path("data").glob(f"**/{stem}/**/original.png"):
                    img_p = match
                    found = True
                    break

    if not img_p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {body.image_path}")

    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to read image at {body.image_path}")

    existing_mask = None
    if body.existing_mask_path and Path(body.existing_mask_path).exists():
        existing_mask = cv2.imread(body.existing_mask_path, cv2.IMREAD_GRAYSCALE)

    segment_stage = _STAGES.get("segment")
    provider = getattr(segment_stage, "segmenter", None)

    refined_mask, conf, bbox, area = refine_mask_with_prompts(
        image_bgr=img_bgr,
        points=body.points,
        box=body.box,
        existing_mask=existing_mask,
        provider=provider,
    )

    out_p = Path(body.output_mask_path) if body.output_mask_path else img_p.parent / "refined_mask.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), refined_mask)

    # Encode to base64 for direct browser overlay
    _, buf = cv2.imencode(".png", refined_mask)
    mask_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return SegmentRefineResponse(
        success=True,
        mask_path=str(out_p),
        mask_base64=mask_b64,
        mask_url=f"data:image/png;base64,{mask_b64}",
        confidence=conf,
        bbox=bbox,
        area=area,
    )


@router.post("/stages/layers/inpaint-region", response_model=InpaintRegionResponse)
async def inpaint_region(body: InpaintRegionRequest) -> InpaintRegionResponse:
    """
    Interactive studio endpoint: re-inpaints a specific bounding box or mask region
    on a panel's background plate.
    """
    import cv2
    import numpy as np
    from mangavrse.stages.layers.quality import evaluate_inpaint_quality

    img_p = Path(body.image_path)
    if not img_p.exists():
        raise HTTPException(status_code=404, detail=f"Image not found at {body.image_path}")

    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        raise HTTPException(status_code=400, detail=f"Failed to read image at {body.image_path}")

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if body.mask_path and Path(body.mask_path).exists():
        m_in = cv2.imread(body.mask_path, cv2.IMREAD_GRAYSCALE)
        if m_in is not None:
            mask = cv2.resize(m_in, (w, h), interpolation=cv2.INTER_NEAREST)
    elif body.box and len(body.box) == 4:
        x0, y0, x1, y1 = body.box
        mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = 255
    else:
        raise HTTPException(status_code=400, detail="Must provide either box [x0, y0, x1, y1] or mask_path")

    filled = inpaint_manga_background(
        image_bgr=img_bgr,
        occlusion_mask=mask,
        inpaint_radius=body.radius,
        method=body.method,
    )

    out_p = Path(body.output_path) if body.output_path else img_p.parent / "reinpainted.png"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), filled)

    metrics = evaluate_inpaint_quality(img_bgr, filled, mask)

    return InpaintRegionResponse(
        success=True,
        output_path=str(out_p),
        metrics=metrics,
    )


@router.post("/stages/colorize/override", response_model=ColorizeOverrideResponse)
async def override_layer_color(body: ColorizeOverrideRequest) -> ColorizeOverrideResponse:
    """
    Interactive studio endpoint: overrides a single layer's color fill
    and updates the master colored SVG and scene.json.
    """
    import json
    from mangavrse.stages.colorize.vector_styler import style_layer_svg, assemble_colored_panel_svg

    panel_path = Path(body.panel_dir)
    if not panel_path.exists():
        stem = panel_path.name
        for cand in [Path(f"data/output/scenes/{stem}"), Path(f"data/output/benchmark_colorized/{stem}"), Path(f"data/output/benchmark_scenes/{stem}")]:
            if cand.exists():
                panel_path = cand
                break
        if not panel_path.exists():
            for cand in Path("data/output").glob(f"**/{stem}"):
                if cand.is_dir() and (cand / "layers").exists():
                    panel_path = cand
                    break

    if not panel_path.exists():
        raise HTTPException(status_code=404, detail=f"Panel dir not found at {body.panel_dir}")

    # Check for layer SVG
    lyr_svg = panel_path / "colored_layers" / f"{body.layer_id}.svg"
    if not lyr_svg.exists():
        lyr_svg = panel_path / "layers" / f"{body.layer_id}.svg"

    out_lyr_svg = panel_path / "colored_layers" / f"{body.layer_id}.svg"
    out_lyr_svg.parent.mkdir(parents=True, exist_ok=True)

    if lyr_svg.exists():
        style_layer_svg(lyr_svg, out_lyr_svg, body.color, body.opacity)

    # Update palette.json if exists
    palette_p = panel_path / "palette.json"
    if palette_p.exists():
        try:
            pal_data = json.loads(palette_p.read_text(encoding="utf-8"))
            pal_data.setdefault("overrides", {})[body.layer_id] = body.color
            palette_p.write_text(json.dumps(pal_data, indent=2), encoding="utf-8")
        except Exception:
            pass

    # Also update scene.json if exists in panel_path or data/output/scenes/{stem}
    for sc_p in [panel_path / "scene.json", Path(f"data/output/scenes/{panel_path.name}/scene.json")]:
        if sc_p.exists():
            try:
                sc = json.loads(sc_p.read_text(encoding="utf-8"))
                for l in sc.get("layers", []):
                    if l.get("id") == body.layer_id:
                        l["color"] = body.color
                sc_p.write_text(json.dumps(sc, indent=2), encoding="utf-8")
            except Exception:
                pass

    master_svg_p = panel_path / "colored_panel.svg"

    return ColorizeOverrideResponse(
        success=True,
        layer_id=body.layer_id,
        color=body.color,
        layer_svg_path=str(out_lyr_svg),
        colored_panel_svg_path=str(master_svg_p),
    )


@router.post("/stages/depth/override", response_model=DepthOverrideResponse)
def depth_override(body: DepthOverrideRequest) -> DepthOverrideResponse:
    """
    Interactive studio endpoint: overrides a single layer's depth value,
    re-sorts zOrder, updates depth.json, and re-renders depth_visualization.png and scene.json.
    """
    import json
    import cv2
    import numpy as np

    panel_path = Path(body.panel_dir)
    if not panel_path.exists():
        stem = panel_path.name
        for cand in [Path(f"data/output/scenes/{stem}"), Path(f"data/output/benchmark_depth/{stem}"), Path(f"data/output/benchmark_scenes/{stem}")]:
            if cand.exists():
                panel_path = cand
                break
        if not panel_path.exists():
            for cand in Path("data/output").glob(f"**/{stem}"):
                if cand.is_dir() and ((cand / "depth.json").exists() or (cand / "scene.json").exists()):
                    panel_path = cand
                    break

    if not panel_path.exists():
        raise HTTPException(status_code=404, detail=f"Panel dir not found at {body.panel_dir}")

    depth_json_p = panel_path / "depth.json"
    if not depth_json_p.exists():
        # Synthesize from scene.json if depth.json not directly in folder
        sc_p = panel_path / "scene.json"
        if sc_p.exists():
            try:
                sc = json.loads(sc_p.read_text(encoding="utf-8"))
                depth_data = {
                    "version": "1.0",
                    "scale": "normalized",
                    "layers": [
                        {"id": l["id"], "type": l.get("type", "layer"), "depth": l["depth"], "zOrder": l.get("zOrder", 0)}
                        for l in sc.get("layers", [])
                    ]
                }
                depth_json_p.write_text(json.dumps(depth_data, indent=2), encoding="utf-8")
            except Exception:
                raise HTTPException(status_code=404, detail=f"depth.json not found in {body.panel_dir}")
        else:
            raise HTTPException(status_code=404, detail=f"depth.json not found in {body.panel_dir}")

    try:
        depth_data = json.loads(depth_json_p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading depth.json: {e}")

    # Update depth and recompute zOrder
    updated_data = apply_depth_override(depth_data, body.layer_id, body.depth)
    depth_json_p.write_text(json.dumps(updated_data, indent=2), encoding="utf-8")

    # Find new z_order for this layer
    new_z = 0
    for lyr in updated_data.get("layers", []):
        if lyr.get("id") == body.layer_id:
            new_z = lyr.get("zOrder", 0)
            break

    # Also update scene.json
    for sc_p in [panel_path / "scene.json", Path(f"data/output/scenes/{panel_path.name}/scene.json")]:
        if sc_p.exists():
            try:
                sc = json.loads(sc_p.read_text(encoding="utf-8"))
                for l in sc.get("layers", []):
                    if l.get("id") == body.layer_id:
                        l["depth"] = float(np.clip(body.depth, 0.0, 1.0))
                        l["zOrder"] = new_z
                        l.setdefault("metadata", {})["is_override"] = True
                sc_p.write_text(json.dumps(sc, indent=2), encoding="utf-8")
            except Exception:
                pass

    # Re-render visualization
    viz_path = panel_path / "depth_visualization.png"
    depth_map_p = panel_path / "depth_map.png"
    if depth_map_p.exists():
        depth_img = cv2.imread(str(depth_map_p), cv2.IMREAD_GRAYSCALE)
        depth_map = (depth_img.astype(np.float32) / 255.0) if depth_img is not None else np.full((1000, 1000), 0.5, dtype=np.float32)
    else:
        depth_map = np.full((1000, 1000), 0.5, dtype=np.float32)

    src_img = None
    for cand in ["enhanced.png", "background_filled.png"]:
        p = panel_path / cand
        if p.exists():
            src_img = cv2.imread(str(p))
            break

    render_depth_visualization(
        depth_map=depth_map,
        layers=updated_data.get("layers", []),
        output_path=viz_path,
        panel_dir=panel_path,
        source_image=src_img,
    )

    return DepthOverrideResponse(
        success=True,
        layer_id=body.layer_id,
        depth=float(np.clip(body.depth, 0.0, 1.0)),
        z_order=new_z,
        depth_json_path=str(depth_json_p),
        visualization_path=str(viz_path),
    )


@router.get("/scenes")
def list_scenes() -> list[dict[str, Any]]:
    """Lists all available assembled 2.5D manga scenes."""
    import json
    scenes = []
    scenes_dir = Path("data/output/benchmark_scenes")
    if scenes_dir.exists():
        for p in sorted(scenes_dir.iterdir()):
            scene_p = p / "scene.json"
            if scene_p.exists():
                try:
                    data = json.loads(scene_p.read_text(encoding="utf-8"))
                    scenes.append({
                        "panel_id": p.name,
                        "project": data["panel"].get("project", "mangavrse"),
                        "width": data["panel"].get("width", 1280),
                        "height": data["panel"].get("height", 1280),
                        "total_layers": len(data.get("layers", [])),
                        "scene_json_url": f"/api/v1/scenes/{p.name}",
                        "source": data["panel"].get("source", "enhanced.png"),
                    })
                except Exception:
                    pass
    return scenes


@router.get("/scenes/{panel_id}")
def get_scene(panel_id: str) -> dict[str, Any]:
    """Returns the validated scene.json for a given panel ID."""
    candidates = [
        Path("data/output/benchmark_scenes") / panel_id / "scene.json",
        Path("data/output/scenes") / panel_id / "scene.json",
        Path("frontend/public/scenes") / panel_id / "scene.json",
        Path("data/output") / panel_id / "scene.json",
        Path("data/output/benchmark_depth") / panel_id / "scene.json",
        *list(Path("data/output/projects").glob(f"**/{panel_id}/scene.json")),
        *list(Path("data/projects").glob(f"**/{panel_id}/**/scene.json")),
    ]
    for c in candidates:
        if c.exists():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read scene: {e}")

    raise HTTPException(status_code=404, detail=f"Scene not found for panel '{panel_id}'")


# ----- System & Project Endpoints for Studio ----------------------

@router.get("/system/gpu")
def get_gpu_status() -> dict[str, Any]:
    """Returns system device status, CUDA presence, and VRAM."""
    import platform
    import torch

    cuda_avail = torch.cuda.is_available()
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_avail,
        "device_name": "CPU",
        "vram_gb": 0.0,
        "device_count": 0,
    }
    if cuda_avail:
        try:
            info["device_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["vram_gb"] = round(props.total_mem / (1024 ** 3), 2)
        except Exception:
            pass
    return info


@router.get("/projects")
def list_projects() -> list[dict[str, Any]]:
    """Lists available manga projects (benchmarks and user projects)."""
    projects = []

    # 1. Benchmark Suite Project
    bench_dir = Path("data/benchmark_panels")
    if bench_dir.exists():
        panels = [p.name for p in bench_dir.iterdir() if p.is_dir()]
        projects.append({
            "id": "benchmark",
            "name": "Benchmark Test Suite",
            "type": "benchmark",
            "pages_count": 5,
            "panels_count": len(panels),
            "description": "5 standard manga panels covering line art, screentone, scan, speed lines, text.",
            "thumbnail_url": "/assets/benchmark_scenes/clean_lineart/enhanced.png",
        })

    # 2. Managed Projects
    proj_dir = Path("data/projects")
    if proj_dir.exists():
        for p in sorted(proj_dir.iterdir()):
            if p.is_dir():
                pages = [x.name for x in p.iterdir() if x.is_dir() and x.name.startswith("page_")]
                # Count panels across pages
                total_panels = 0
                first_thumb = None
                for pg in pages:
                    pg_dir = p / pg
                    panels_json = pg_dir / "panels.json"
                    if panels_json.exists():
                        try:
                            man = json.loads(panels_json.read_text(encoding="utf-8"))
                            total_panels += len(man.get("panels", []))
                        except Exception:
                            pass
                    if not first_thumb:
                        page_png = pg_dir / f"{pg}.png"
                        if page_png.exists():
                            first_thumb = f"/assets/{p.relative_to('data/output')}/{pg}/{pg}.png" if 'data/output' in str(p) else f"/api/v1/projects/{p.name}/pages/0/image"

                projects.append({
                    "id": p.name,
                    "name": p.name.replace("_", " ").title(),
                    "type": "project",
                    "pages_count": len(pages),
                    "panels_count": total_panels,
                    "description": f"Manga project with {len(pages)} pages and {total_panels} detected panels.",
                    "thumbnail_url": first_thumb or "/assets/benchmark_scenes/clean_lineart/enhanced.png",
                })

    return projects


@router.post("/projects/demo")
def create_demo_project() -> dict[str, Any]:
    """
    Creates a demonstration manga project from sample manga pages using IngestStage.
    """
    import cv2
    import numpy as np

    demo_dir = Path("data/projects/demo_manga")
    demo_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize 2 high-quality manga pages with panels if not present
    input_dir = demo_dir / "input_pages"
    input_dir.mkdir(exist_ok=True)

    # Page 1: 4 panels
    p1_img = np.full((1800, 1200, 3), 255, dtype=np.uint8)
    cv2.rectangle(p1_img, (60, 60), (1140, 1740), (0, 0, 0), 6)
    # Gutter splits
    cv2.line(p1_img, (60, 900), (1140, 900), (0, 0, 0), 4)
    cv2.line(p1_img, (600, 60), (600, 900), (0, 0, 0), 4)
    # Character sketch in top-right panel (Panel 1 RTL)
    cv2.circle(p1_img, (870, 450), 120, (30, 30, 30), -1)
    cv2.ellipse(p1_img, (870, 280), (140, 80), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(p1_img, (870, 280), (140, 80), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(p1_img, "MANGA!", (810, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

    cv2.imwrite(str(input_dir / "page_001.png"), p1_img)

    # 2. Run Ingest stage on demo project
    ingest_stage = _STAGES["ingest"]
    res = ingest_stage.execute(input_dir, demo_dir, config={"ingest": {"direction": "rtl"}})

    return {
        "success": res.success,
        "project_id": "demo_manga",
        "message": "Demo manga project initialized and ingested.",
    }


def run_panel_full_reconstruction(
    panel_crop_path: Path,
    project_id: str,
    panel_id: str,
    scene_id: str,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """
    Executes the complete MangaVrse pipeline end-to-end for a given panel crop:
    Enhance -> Segment -> Vectorize -> Layers -> Colorize -> Depth -> Assembly.
    Copies all resulting scene assets into frontend/public and static output paths.
    """
    work_dir = Path(f"data/output/projects/{project_id}/{panel_id}")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure original.png exists in work_dir
    target_orig = work_dir / "original.png"
    if panel_crop_path.resolve() != target_orig.resolve():
        shutil.copy2(panel_crop_path, target_orig)

    # 2. Stage: Enhance
    enh_cfg = {"enhance": {"upscale": {"scale": 2, "use_neural": False}}} if fast_mode else {}
    _STAGES["enhance"].execute(work_dir, work_dir, enh_cfg)

    # 3. Stage: Segment
    _STAGES["segment"].execute(work_dir, work_dir, {})

    # 4. Stage: Vectorize
    _STAGES["vectorize"].execute(work_dir, work_dir, {})

    # 5. Stage: Layers
    _STAGES["layers"].execute(work_dir, work_dir, {})

    # 6. Stage: Colorize
    _STAGES["colorize"].execute(work_dir, work_dir, {})

    # 7. Stage: Depth
    _STAGES["depth"].execute(work_dir, work_dir, {})

    # 8. Stage: Assembly
    _STAGES["assembly"].execute(work_dir, work_dir, {})

    scene_path = work_dir / "scene.json"
    if not scene_path.exists():
        raise RuntimeError(f"Scene assembly did not produce scene.json for {panel_id}")

    scene_data = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_data.setdefault("panel", {})["project"] = project_id
    scene_data["panel"]["panel"] = panel_id
    scene_path.write_text(json.dumps(scene_data, indent=2), encoding="utf-8")

    # Mirror output bundle to benchmark_scenes, scenes, and frontend public for zero-delay loading
    targets = [
        Path("data/output/benchmark_scenes") / scene_id,
        Path("data/output/scenes") / scene_id,
        Path("frontend/public/scenes") / scene_id,
    ]
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work_dir, target, dirs_exist_ok=True)

    return scene_data


@router.post("/projects/upload")
async def upload_manga(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    auto_process: bool = Form(True),
    fast_mode: bool = Form(False),
) -> dict[str, Any]:
    """
    Accepts an uploaded manga file (PDF, PNG, JPG, WEBP), ingests pages/panels,
    and automatically executes the end-to-end 2.5D reconstruction pipeline.
    """
    raw_name = name or Path(file.filename or "uploaded_manga").stem
    clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name).strip("_").lower() or "manga"
    timestamp = int(time.time())
    project_id = f"{clean_slug}_{timestamp}"

    proj_dir = Path("data/projects") / project_id
    input_dir = proj_dir / "input_pages"
    input_dir.mkdir(parents=True, exist_ok=True)

    file_ext = Path(file.filename or "manga.png").suffix.lower()
    if not file_ext:
        file_ext = ".png"
    dest_file = input_dir / f"uploaded{file_ext}"

    content = await file.read()
    dest_file.write_bytes(content)

    # Ingest Stage
    ingest_res = _STAGES["ingest"].execute(
        input_dir=input_dir,
        output_dir=proj_dir,
        config={"ingest": {"direction": "rtl"}},
    )

    if not ingest_res.success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest manga: {ingest_res.metadata.get('error', 'unknown error')}",
        )

    # Discover pages and panels
    pages = [p for p in sorted(proj_dir.iterdir()) if p.is_dir() and p.name.startswith("page_")]
    if not pages:
        raise HTTPException(status_code=500, detail="No pages generated from upload")

    first_page_dir = pages[0]
    panels_json_p = first_page_dir / "panels.json"
    panel_ids: list[str] = []
    if panels_json_p.exists():
        try:
            pj = json.loads(panels_json_p.read_text(encoding="utf-8"))
            panel_ids = [p["id"] for p in pj.get("panels", []) if "id" in p]
        except Exception:
            pass

    if not panel_ids:
        panel_subdirs = [p.name for p in sorted(first_page_dir.iterdir()) if p.is_dir() and p.name.startswith("panel_")]
        panel_ids = panel_subdirs or ["panel_001"]

    target_panel_id = panel_ids[0]
    target_crop = first_page_dir / target_panel_id / "original.png"
    if not target_crop.exists():
        page_png = first_page_dir / f"{first_page_dir.name}.png"
        if page_png.exists():
            (first_page_dir / target_panel_id).mkdir(parents=True, exist_ok=True)
            shutil.copy2(page_png, target_crop)

    scene_id = f"{clean_slug}_{target_panel_id}"
    scene_data = None

    if auto_process:
        scene_data = run_panel_full_reconstruction(
            panel_crop_path=target_crop,
            project_id=project_id,
            panel_id=target_panel_id,
            scene_id=scene_id,
            fast_mode=fast_mode,
        )

    return {
        "success": True,
        "project_id": project_id,
        "project_name": raw_name.replace("_", " ").title(),
        "page_number": 1,
        "panel_id": target_panel_id,
        "scene_id": scene_id,
        "total_pages": len(pages),
        "total_panels": len(panel_ids),
        "scene": scene_data,
        "scene_url": f"/api/v1/scenes/{scene_id}",
        "message": "Manga successfully uploaded and reconstructed into 2.5D scene!",
    }


@router.get("/projects/{project_id}/pages")
def get_project_pages(project_id: str) -> list[dict[str, Any]]:
    """Returns pages for a project with detected panel overlays and RTL reading order."""
    pages = []

    # Handle benchmark suite
    if project_id == "benchmark":
        bench_dir = Path("data/benchmark_panels")
        panels = [p.name for p in sorted(bench_dir.iterdir()) if p.is_dir()]
        for idx, p_id in enumerate(panels):
            pages.append({
                "page_number": idx + 1,
                "title": f"Benchmark Panel: {p_id}",
                "image_url": f"/assets/benchmark_scenes/{p_id}/enhanced.png",
                "width": 1280,
                "height": 1280,
                "panels": [
                    {
                        "id": p_id,
                        "panel_index": 0,
                        "reading_order": 1,
                        "bbox": [0, 0, 1280, 1280],
                        "source_png": f"/assets/benchmark_scenes/{p_id}/enhanced.png",
                    }
                ],
            })
        return pages

    # Handle managed project
    proj_dir = Path("data/projects") / project_id
    if not proj_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    page_dirs = [p for p in sorted(proj_dir.iterdir()) if p.is_dir() and p.name.startswith("page_")]
    for idx, pg_dir in enumerate(page_dirs):
        page_num = idx + 1
        panels_data = []
        panels_json = pg_dir / "panels.json"
        w, h = 1200, 1800
        if panels_json.exists():
            try:
                pj = json.loads(panels_json.read_text(encoding="utf-8"))
                for p_entry in pj.get("panels", []):
                    raw_bbox = p_entry.get("bbox", [0, 0, 500, 500])
                    x, y, bw, bh = raw_bbox
                    panels_data.append({
                        "id": p_entry.get("id"),
                        "panel_index": p_entry.get("panel_index", 0),
                        "reading_order": p_entry.get("reading_order", 0) + 1,
                        "bbox": [x, y, x + bw, y + bh],
                        "source_png": f"/api/v1/projects/{project_id}/panels/{p_entry.get('id')}/crop?page={page_num}",
                    })
            except Exception:
                pass

        pages.append({
            "page_number": page_num,
            "title": f"Page {page_num:03d}",
            "image_url": f"/api/v1/projects/{project_id}/pages/{page_num}/image",
            "width": w,
            "height": h,
            "panels": panels_data,
        })

    return pages


@router.get("/projects/{project_id}/pages/{page_num}/image")
def get_page_image(project_id: str, page_num: int):
    """Returns the full resolution page image."""
    from fastapi.responses import FileResponse
    proj_dir = Path("data/projects") / project_id
    page_name = f"page_{page_num:03d}"
    page_img = proj_dir / page_name / f"{page_name}.png"
    if not page_img.exists():
        page_img = proj_dir / "input_pages" / f"{page_name}.png"
    if not page_img.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(str(page_img), media_type="image/png")


@router.get("/projects/{project_id}/panels/{panel_id}/crop")
def get_panel_crop(project_id: str, panel_id: str, page: int = 1):
    """Returns the original cropped panel image."""
    from fastapi.responses import FileResponse
    proj_dir = Path("data/projects") / project_id
    page_name = f"page_{page:03d}"
    panel_crop = proj_dir / page_name / panel_id / "original.png"
    if not panel_crop.exists():
        bench_crop = Path("data/benchmark_panels") / panel_id / "original.png"
        if bench_crop.exists():
            return FileResponse(str(bench_crop), media_type="image/png")
        raise HTTPException(status_code=404, detail="Panel crop not found")
    return FileResponse(str(panel_crop), media_type="image/png")


@router.get("/projects/{project_id}/panels/{panel_id}/stages")
def get_panel_stages(project_id: str, panel_id: str) -> dict[str, Any]:
    """
    Returns artifact statuses and URLs across all stages for a specific panel.
    Supports benchmark suites and dynamically uploaded projects.
    """
    # 1. Check user project output directory
    user_work_candidates = [
        Path(f"data/output/projects/{project_id}/{panel_id}"),
        Path(f"data/output/scenes/{panel_id}"),
        Path(f"data/output/benchmark_scenes/{panel_id}"),
    ]
    user_work_dir = None
    for cand in user_work_candidates:
        if cand.exists():
            user_work_dir = cand
            break

    # 2. Check standard stage directories
    base_dirs = {
        "enhance": Path("data/output/benchmark_enhanced") / panel_id,
        "vectorize": Path("data/output/benchmark_vectorized") / panel_id,
        "segment": Path("data/output/benchmark_segmented") / panel_id,
        "layers": Path("data/output/benchmark_layers") / panel_id,
        "colorize": Path("data/output/benchmark_colorized") / panel_id,
        "depth": Path("data/output/benchmark_depth") / panel_id,
        "assembly": Path("data/output/benchmark_scenes") / panel_id,
    }

    stages_result = {}

    for stage_name, s_dir in base_dirs.items():
        search_dir = user_work_dir if (user_work_dir and user_work_dir.exists()) else s_dir
        if search_dir and search_dir.exists():
            manifest = search_dir / ".cache_manifest.json"
            cached = manifest.exists()

            artifacts_map: dict[str, str] = {}
            for f in search_dir.glob("*.*"):
                if f.name != ".cache_manifest.json":
                    try:
                        rel = f.relative_to("data/output")
                        artifacts_map[f.stem] = f"/assets/{rel.as_posix()}"
                    except Exception:
                        artifacts_map[f.stem] = f"/assets/projects/{project_id}/{panel_id}/{f.name}"

            stages_result[stage_name] = {
                "status": "completed",
                "cached": cached,
                "artifacts": artifacts_map,
            }
        else:
            stages_result[stage_name] = {
                "status": "pending",
                "cached": False,
                "artifacts": {},
            }

    return {
        "project_id": project_id,
        "panel_id": panel_id,
        "stages": stages_result,
    }


@router.get("/projects/{project_id}/panels/{panel_id}/export")
def export_panel_package(project_id: str, panel_id: str):
    """
    Bundles scene.json and all corresponding stage assets into a downloadable zip file.
    """
    candidates = [
        Path("data/output/benchmark_scenes") / panel_id,
        Path("data/output/scenes") / panel_id,
        Path(f"data/output/projects/{project_id}/{panel_id}"),
        Path("frontend/public/scenes") / panel_id,
    ]
    panel_dir = None
    for c in candidates:
        if c.exists() and (c / "scene.json").exists():
            panel_dir = c
            break

    if not panel_dir:
        raise HTTPException(status_code=404, detail=f"Panel scene bundle not found for {panel_id}")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(panel_dir):
            for file in files:
                f_path = Path(root) / file
                arcname = f_path.relative_to(panel_dir)
                zf.write(f_path, arcname=str(arcname))

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=mangavrse_{panel_id}_scene.zip"},
    )



