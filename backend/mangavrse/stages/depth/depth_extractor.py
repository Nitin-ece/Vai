"""
Depth Extractor: extracts per-layer median depth from continuous depth maps,
blends with manga semantic priors, sorts zOrder, and manages manual depth overrides.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Manga semantic depth priors (0.0 = far/background, 1.0 = near/foreground)
SEMANTIC_DEPTH_PRIORS: dict[str, float] = {
    "speech_bubble": 0.95,
    "sfx": 0.92,
    "foreground": 0.85,
    "effect": 0.82,
    "character_eyes": 0.78,
    "character_hair": 0.76,
    "character_head": 0.73,
    "character_clothing": 0.68,
    "character_body": 0.68,
    "character": 0.70,
    "prop": 0.55,
    "environment": 0.35,
    "floor": 0.30,
    "background": 0.08,
    "sky": 0.04,
}


def get_semantic_prior(label: str | None, part: str | None = None) -> float:
    """Returns the expected manga depth prior for a given label and part."""
    lbl = (label or "character").lower().strip()
    prt = (part or "").lower().strip()

    if lbl == "character" and prt:
        key = f"character_{prt}"
        if key in SEMANTIC_DEPTH_PRIORS:
            return SEMANTIC_DEPTH_PRIORS[key]

    if lbl in SEMANTIC_DEPTH_PRIORS:
        return SEMANTIC_DEPTH_PRIORS[lbl]

    return 0.50


def load_layer_mask(layer: dict[str, Any], panel_dir: Path | None = None, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Attempts to retrieve or load the binary mask for a layer."""
    h, w = shape if shape is not None else (1000, 1000)

    # 1. Direct mask in dict
    if "mask" in layer and isinstance(layer["mask"], np.ndarray):
        return layer["mask"]

    # 2. From layer PNG alpha channel
    png_rel = layer.get("pngPath")
    if png_rel and panel_dir:
        png_p = panel_dir / png_rel if not Path(png_rel).is_absolute() else Path(png_rel)
        if png_p.exists():
            img = cv2.imread(str(png_p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] == 4:
                return img[:, :, 3]

    # 3. From masks/<id>.png
    layer_id = layer.get("id")
    if layer_id and panel_dir:
        mask_p = panel_dir / "masks" / f"{layer_id}.png"
        if mask_p.exists():
            m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return m

    # 4. Fallback to bbox
    bbox = layer.get("bbox")
    if bbox and len(bbox) == 4:
        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, x1 = max(0, min(w - 1, x0)), max(0, min(w, x1))
        y0, y1 = max(0, min(h - 1, y0)), max(0, min(h, y1))
        mask[y0:y1, x0:x1] = 255
        return mask

    return None


def extract_layer_depths(
    depth_map: np.ndarray,
    layers: list[dict[str, Any]],
    panel_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """
    Computes per-layer depth by taking the median depth inside each layer mask,
    blending with semantic priors, respecting manual overrides, and assigning zOrder.
    """
    cfg = config or {}
    model_weight_default = float(cfg.get("model_weight", 0.50))
    overrides = overrides or {}

    h, w = depth_map.shape[:2]
    processed_layers: list[dict[str, Any]] = []

    for lyr in layers:
        layer_dict = dict(lyr)
        layer_id = str(layer_dict.get("id", "unknown"))
        label = layer_dict.get("label") or layer_dict.get("type") or "character"
        part = layer_dict.get("part")

        prior = get_semantic_prior(label, part)

        if layer_id in overrides:
            final_depth = float(np.clip(overrides[layer_id], 0.0, 1.0))
            is_override = True
            model_median = float(prior)
        else:
            is_override = False
            mask = load_layer_mask(layer_dict, panel_dir=panel_dir, shape=(h, w))
            if mask is not None and np.any(mask > 128):
                # Sample depth inside mask
                vals = depth_map[mask > 128]
                model_median = float(np.median(vals))
            else:
                model_median = float(prior)

            # Adaptive blending based on semantic category
            if label in ("speech_bubble", "sfx"):
                weight = 0.05   # Strictly preserve speech bubble foreground position
            elif label in ("background", "sky"):
                weight = 0.15   # Ensure background plate stays safely behind figures
            else:
                weight = model_weight_default

            blended = weight * model_median + (1.0 - weight) * prior
            final_depth = float(np.clip(blended, 0.0, 1.0))

        layer_dict["depth"] = round(final_depth, 3)
        layer_dict["model_depth"] = round(model_median, 3)
        layer_dict["prior_depth"] = round(prior, 3)
        layer_dict["is_override"] = is_override

        processed_layers.append(layer_dict)

    # Sort layers by depth: lowest (furthest) -> highest (closest)
    processed_layers.sort(key=lambda x: (x["depth"], 0 if x.get("label") == "background" else 1))

    # Assign zOrder integers (0 = back, N-1 = front)
    for idx, lyr in enumerate(processed_layers):
        lyr["zOrder"] = idx

    return processed_layers


def build_scene_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts layer dicts to match the scene.json layer schema."""
    scene_layers: list[dict[str, Any]] = []

    for lyr in layers:
        scene_layer = {
            "id": lyr.get("id"),
            "type": lyr.get("label") or lyr.get("type") or "character",
            "parentId": lyr.get("parentId"),
            "svgPath": lyr.get("svgPath", ""),
            "pngPath": lyr.get("pngPath", ""),
            "color": lyr.get("color", "#FFFFFF"),
            "depth": lyr.get("depth", 0.0),
            "zOrder": lyr.get("zOrder", 0),
            "transform": lyr.get(
                "transform",
                {"x": 0, "y": 0, "scaleX": 1.0, "scaleY": 1.0, "rotation": 0.0},
            ),
            "metadata": {
                "part": lyr.get("part"),
                "confidence": lyr.get("confidence", 1.0),
                "model_depth": lyr.get("model_depth"),
                "prior_depth": lyr.get("prior_depth"),
                "is_override": lyr.get("is_override", False),
                "bbox": lyr.get("bbox"),
            },
        }
        scene_layers.append(scene_layer)

    return scene_layers


def apply_depth_override(
    depth_data: dict[str, Any],
    layer_id: str,
    new_depth: float,
) -> dict[str, Any]:
    """
    Updates or inserts a manual depth override for a layer and recalculates zOrder.
    """
    overrides = depth_data.setdefault("overrides", {})
    overrides[layer_id] = round(float(np.clip(new_depth, 0.0, 1.0)), 3)

    layers = depth_data.get("layers", [])
    found = False
    for lyr in layers:
        if lyr.get("id") == layer_id:
            lyr["depth"] = overrides[layer_id]
            lyr["is_override"] = True
            found = True
            break

    if not found:
        # Layer wasn't in layers list yet
        layers.append({
            "id": layer_id,
            "depth": overrides[layer_id],
            "is_override": True,
        })

    # Re-sort
    layers.sort(key=lambda x: (x["depth"], 0 if x.get("label") == "background" else 1))
    for idx, lyr in enumerate(layers):
        lyr["zOrder"] = idx

    depth_data["layers"] = layers
    depth_data["scene_layers"] = build_scene_layers(layers)
    return depth_data
