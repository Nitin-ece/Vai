"""
Scene Assembler: combines layers, colors, and depth into a validated scene.json.
Validates the scene against docs/scene_schema.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[4] / "docs" / "scene_schema.json"


def load_schema(schema_path: Path | None = None) -> dict[str, Any]:
    """Loads the scene JSON schema from disk."""
    p = schema_path or DEFAULT_SCHEMA_PATH
    if not p.exists():
        # Fallback relative to project root
        p = Path("docs/scene_schema.json")
    if not p.exists():
        raise FileNotFoundError(f"Scene schema not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def validate_scene(scene_data: dict[str, Any], schema_path: Path | None = None) -> tuple[bool, str | None]:
    """
    Validates a scene dictionary against docs/scene_schema.json.
    Returns (True, None) if valid, or (False, error_description) if invalid.
    """
    try:
        schema = load_schema(schema_path)
        jsonschema.validate(instance=scene_data, schema=schema)
        return True, None
    except jsonschema.ValidationError as err:
        return False, f"Validation error at {'/'.join(str(p) for p in err.path)}: {err.message}"
    except Exception as e:
        return False, f"Schema validation exception: {e}"


def assemble_scene(
    panel_id: str,
    panel_dir: Path,
    layers_data: list[dict[str, Any]] | None = None,
    palette_data: dict[str, Any] | None = None,
    depth_data: dict[str, Any] | None = None,
    width: int | None = None,
    height: int | None = None,
    source_name: str | None = None,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """
    Combines layers, colors, and depth into a complete, validated scene dictionary.
    """
    # 1. Discover or load depth data
    if depth_data is None:
        depth_json_p = panel_dir / "depth.json"
        if depth_json_p.exists():
            try:
                depth_data = json.loads(depth_json_p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to read depth.json: {e}")

    # 2. Discover or load layers data
    if layers_data is None:
        layers_manifest_p = panel_dir / "layers_manifest.json"
        if not layers_manifest_p.exists():
            layers_manifest_p = panel_dir / "segments.json"
        if layers_manifest_p.exists():
            try:
                man = json.loads(layers_manifest_p.read_text(encoding="utf-8"))
                layers_data = man.get("layers") or man.get("segments") or []
                if not width and "panel" in man:
                    width = man["panel"].get("width")
                    height = man["panel"].get("height")
            except Exception as e:
                logger.warning(f"Failed to read layers manifest: {e}")

    # If depth_data contains layers, use them as baseline for depth/zOrder
    depth_layers_map: dict[str, dict[str, Any]] = {}
    if depth_data and "layers" in depth_data:
        for dl in depth_data["layers"]:
            if "id" in dl:
                depth_layers_map[dl["id"]] = dl

    # 3. Discover or load palette data
    if palette_data is None:
        palette_p = panel_dir / "palette.json"
        if palette_p.exists():
            try:
                palette_data = json.loads(palette_p.read_text(encoding="utf-8"))
            except Exception:
                pass

    color_overrides = (palette_data or {}).get("overrides", {})
    palette_colors = (palette_data or {}).get("colors", {})

    # Determine canvas dimensions
    w = width or 1280
    h = height or 1280
    src = source_name or "enhanced.png"

    # 4. Construct unified layers
    raw_layers = layers_data or []
    if not raw_layers and depth_data and "layers" in depth_data:
        raw_layers = depth_data["layers"]

    assembled_layers: list[dict[str, Any]] = []

    for idx, lyr in enumerate(raw_layers):
        layer_id = str(lyr.get("id", f"layer_{idx:03d}"))
        l_type = lyr.get("label") or lyr.get("type") or "character"
        # Map unknown types to valid enum
        valid_types = {
            "background", "character", "character_part", "speech_bubble",
            "sfx", "foreground", "effect", "prop", "environment"
        }
        if l_type not in valid_types:
            if "bubble" in l_type:
                l_type = "speech_bubble"
            elif "part" in l_type:
                l_type = "character_part"
            else:
                l_type = "character"

        parent_id = lyr.get("parentId")
        svg_path = lyr.get("svgPath") or f"layers/{layer_id}.svg"
        png_path = lyr.get("pngPath") or f"layers/{layer_id}.png"

        # Check color
        color = lyr.get("color")
        if not color:
            color = color_overrides.get(layer_id) or palette_colors.get(layer_id, "#FFFFFF")

        # Depth and zOrder from depth stage
        depth_info = depth_layers_map.get(layer_id, {})
        depth_val = float(depth_info.get("depth", lyr.get("depth", 0.50)))
        z_order_val = int(depth_info.get("zOrder", lyr.get("zOrder", idx)))

        # Metadata
        part = lyr.get("part")
        bbox = lyr.get("bbox") or [0, 0, w, h]
        conf = float(lyr.get("confidence", 1.0))
        model_depth = float(depth_info.get("model_depth", depth_val))
        prior_depth = float(depth_info.get("prior_depth", depth_val))
        is_override = bool(depth_info.get("is_override", False))

        # Animation hooks (off by default)
        animations: dict[str, Any] = {}
        if l_type in ("character", "character_part"):
            animations["sway"] = {"enabled": False, "speed": 1.0, "amplitude": 0.02}
        if l_type in ("speech_bubble", "sfx", "effect"):
            animations["float"] = {"enabled": False, "speed": 1.2, "amplitude": 0.03}
        if part == "eyes" or "eye" in layer_id:
            animations["blink"] = {"enabled": False, "interval": 4.0}

        layer_obj = {
            "id": layer_id,
            "type": l_type,
            "parentId": parent_id,
            "svgPath": svg_path,
            "pngPath": png_path,
            "color": color,
            "depth": round(max(0.0, min(1.0, depth_val)), 3),
            "zOrder": z_order_val,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scaleX": 1.0,
                "scaleY": 1.0,
                "rotation": 0.0,
            },
            "metadata": {
                "part": part,
                "confidence": conf,
                "model_depth": model_depth,
                "prior_depth": prior_depth,
                "is_override": is_override,
                "bbox": [int(v) for v in bbox],
                "animations": animations,
            },
        }
        assembled_layers.append(layer_obj)

    # Ensure pure white paper canvas is present as the base background layer (zOrder 0, depth 0.01)
    # This guarantees the manga sits on a blank white page with no old image ghosting behind it
    has_canvas = any(l.get("id") == "canvas_white" or l.get("pngPath") == "background_filled.png" for l in assembled_layers)
    if not has_canvas:
        assembled_layers.insert(0, {
            "id": "canvas_white",
            "type": "background",
            "parentId": None,
            "svgPath": "layers/canvas_white.svg",
            "pngPath": "background_filled.png",
            "color": "#FFFFFF",
            "depth": 0.01,
            "zOrder": 0,
            "transform": {
                "x": 0.0,
                "y": 0.0,
                "scaleX": 1.0,
                "scaleY": 1.0,
                "rotation": 0.0,
            },
            "metadata": {
                "part": None,
                "confidence": 1.0,
                "model_depth": 0.01,
                "prior_depth": 0.01,
                "is_override": False,
                "bbox": [0, 0, int(w), int(h)],
                "animations": {},
            },
        })

    # Sort layers by depth ascending then zOrder
    assembled_layers.sort(key=lambda l: (l.get("depth", 0.0), l.get("zOrder", 0)))
    # Re-normalize zOrder sequentially
    for i, l in enumerate(assembled_layers):
        l["zOrder"] = i

    scene: dict[str, Any] = {
        "version": "1.0",
        "panel": {
            "project": "mangavrse",
            "page": 1,
            "panel": 1,
            "source": src,
            "width": int(w),
            "height": int(h),
        },
        "camera": {
            "fov": 45.0,
            "near": 0.1,
            "far": 100.0,
            "position": [0.0, 0.0, 5.0],
            "limits": {
                "minDistance": 3.2,
                "maxDistance": 5.5,
                "maxPitch": 0.045,
                "maxYaw": 0.060,
            },
        },
        "layers": assembled_layers,
    }

    # Validate against JSON schema
    is_valid, error = validate_scene(scene, schema_path=schema_path)
    if not is_valid:
        raise ValueError(f"Assembled scene.json failed schema validation: {error}")

    return scene
