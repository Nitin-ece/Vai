"""
Unit and integration tests for the Depth stage.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mangavrse.stages.depth import (
    DepthEstimatorProvider,
    DepthStage,
    apply_depth_override,
    build_scene_layers,
    extract_layer_depths,
    get_semantic_prior,
    render_depth_map,
    render_depth_visualization,
)


def create_synthetic_panel_for_depth(panel_dir: Path, w: int = 400, h: int = 400) -> Path:
    """Sets up a panel directory matching the layers stage output."""
    panel_dir.mkdir(parents=True, exist_ok=True)

    img = np.full((h, w, 3), 250, dtype=np.uint8)  # White paper

    # Draw panel borders
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (10, 10, 10), 3)

    # Screentone background pattern (top half)
    for y in range(20, h // 2, 6):
        for x in range(20, w - 20, 6):
            cv2.circle(img, (x, y), 1, (120, 120, 120), -1)

    # Character head & body (center)
    cv2.circle(img, (w // 2, int(h * 0.55)), 60, (20, 20, 20), 4)  # Head
    cv2.rectangle(img, (w // 2 - 40, int(h * 0.7)), (w // 2 + 40, h - 30), (30, 30, 30), 4)  # Body

    # Speech bubble (top right)
    cv2.ellipse(img, (int(w * 0.75), 80), (70, 45), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (int(w * 0.75), 80), (70, 45), 0, 0, 360, (0, 0, 0), 2)

    cv2.imwrite(str(panel_dir / "enhanced.png"), img)

    lyr_dir = panel_dir / "layers"
    lyr_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = panel_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 1. Background layer
    bg_mask = np.full((h, w), 255, dtype=np.uint8)
    cv2.imwrite(str(mask_dir / "lyr_bg.png"), bg_mask)

    # 2. Character body layer
    char_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(char_mask, (w // 2, int(h * 0.55)), 60, 255, -1)
    cv2.rectangle(char_mask, (w // 2 - 40, int(h * 0.7)), (w // 2 + 40, h - 30), 255, -1)
    cv2.imwrite(str(mask_dir / "lyr_char.png"), char_mask)

    # 3. Speech bubble layer
    bubble_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (int(w * 0.75), 80), (70, 45), 0, 0, 360, 255, -1)
    cv2.imwrite(str(mask_dir / "lyr_bubble.png"), bubble_mask)

    for name, m in [("lyr_bg", bg_mask), ("lyr_char", char_mask), ("lyr_bubble", bubble_mask)]:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 3] = m
        cv2.imwrite(str(lyr_dir / f"{name}.png"), rgba)

    manifest = {
        "panel": {"panel_id": panel_dir.name, "width": w, "height": h},
        "layers": [
            {
                "id": "lyr_bg",
                "label": "background",
                "part": None,
                "parentId": None,
                "bbox": [0, 0, w, h],
                "pngPath": "layers/lyr_bg.png",
                "svgPath": "layers/lyr_bg.svg",
            },
            {
                "id": "lyr_char",
                "label": "character",
                "part": "body",
                "parentId": "lyr_bg",
                "bbox": [w // 2 - 60, int(h * 0.4), w // 2 + 60, h - 30],
                "pngPath": "layers/lyr_char.png",
                "svgPath": "layers/lyr_char.svg",
            },
            {
                "id": "lyr_bubble",
                "label": "speech_bubble",
                "part": None,
                "parentId": None,
                "bbox": [int(w * 0.75) - 70, 35, int(w * 0.75) + 70, 125],
                "pngPath": "layers/lyr_bubble.png",
                "svgPath": "layers/lyr_bubble.svg",
            },
        ],
    }
    (panel_dir / "layers_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return panel_dir


@pytest.fixture
def sample_manga_panel() -> np.ndarray:
    """Creates a synthetic manga panel with line art, screentone, and characters."""
    img = np.full((400, 400, 3), 250, dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (390, 390), (10, 10, 10), 3)
    cv2.circle(img, (200, 220), 60, (20, 20, 20), 4)
    return img


def test_depth_provider_pipeline(sample_manga_panel: np.ndarray) -> None:
    """Verifies that DepthEstimatorProvider conforms to the ModelProvider contract."""
    provider = DepthEstimatorProvider()
    provider.load(device="cpu")

    preprocessed = provider.preprocess(sample_manga_panel)
    assert preprocessed.shape == sample_manga_panel.shape

    raw_output = provider.predict(preprocessed)
    assert isinstance(raw_output, np.ndarray)

    depth_map = provider.postprocess(raw_output)
    assert depth_map.shape == sample_manga_panel.shape[:2]
    assert depth_map.min() >= 0.0
    assert depth_map.max() <= 1.0


def test_semantic_priors() -> None:
    """Verifies hierarchy of manga depth semantic priors."""
    bubble_prior = get_semantic_prior("speech_bubble")
    sfx_prior = get_semantic_prior("sfx")
    char_prior = get_semantic_prior("character")
    char_eyes_prior = get_semantic_prior("character", part="eyes")
    char_body_prior = get_semantic_prior("character", part="body")
    bg_prior = get_semantic_prior("background")
    sky_prior = get_semantic_prior("sky")

    assert bubble_prior > char_prior > bg_prior > sky_prior
    assert sfx_prior > char_prior
    assert char_eyes_prior >= char_body_prior
    assert bg_prior < 0.20
    assert bubble_prior > 0.90


def test_layer_depth_extraction_and_sorting() -> None:
    """Tests median depth calculation, semantic blending, and zOrder ordering."""
    h, w = 200, 200
    depth_map = np.linspace(0.1, 0.9, h, dtype=np.float32)[:, None]
    depth_map = np.tile(depth_map, (1, w))

    layers = [
        {"id": "bg", "label": "background", "bbox": [0, 0, w, h]},
        {"id": "char", "label": "character", "part": "body", "bbox": [50, 50, 150, 150]},
        {"id": "bubble", "label": "speech_bubble", "bbox": [20, 20, 80, 80]},
    ]

    processed = extract_layer_depths(depth_map, layers)
    assert len(processed) == 3

    id_by_z = {l["zOrder"]: l["id"] for l in processed}
    assert id_by_z[0] == "bg"
    assert id_by_z[2] == "bubble"

    depths = [l["depth"] for l in processed]
    assert depths[0] <= depths[1] <= depths[2]


def test_manual_override_functionality() -> None:
    """Tests that manual overrides strictly override automatic calculation and re-sort zOrder."""
    h, w = 100, 100
    depth_map = np.full((h, w), 0.5, dtype=np.float32)
    layers = [
        {"id": "l1", "label": "character", "depth": 0.7, "zOrder": 1},
        {"id": "l2", "label": "background", "depth": 0.1, "zOrder": 0},
    ]

    depth_data = {
        "overrides": {},
        "layers": layers,
        "scene_layers": build_scene_layers(layers),
    }

    # Override background to 0.95 (bringing it in front of character)
    updated = apply_depth_override(depth_data, "l2", 0.95)
    assert updated["overrides"]["l2"] == 0.95

    l2_entry = next(l for l in updated["layers"] if l["id"] == "l2")
    l1_entry = next(l for l in updated["layers"] if l["id"] == "l1")

    assert l2_entry["depth"] == 0.95
    assert l2_entry["is_override"] is True
    assert l2_entry["zOrder"] > l1_entry["zOrder"]


def test_depth_stage_execution_and_caching(tmp_path: Path) -> None:
    """Tests full DepthStage pipeline execution and hash-based caching."""
    in_dir = tmp_path / "panel_in"
    out_dir = tmp_path / "panel_out"
    in_dir.mkdir()
    out_dir.mkdir()

    p_in = in_dir / "panel_001"
    create_synthetic_panel_for_depth(p_in)

    stage = DepthStage()

    # Run 1: Fresh execution
    res1 = stage.execute(in_dir, out_dir, config={})
    assert res1.success is True
    assert res1.cached is False

    p_out = out_dir / "panel_001"
    assert (p_out / "depth_map.png").exists()
    assert (p_out / "depth.json").exists()
    assert (p_out / "depth_visualization.png").exists()

    depth_json = json.loads((p_out / "depth.json").read_text(encoding="utf-8"))
    assert depth_json["total_layers"] == 3
    assert len(depth_json["scene_layers"]) == 3
    for sl in depth_json["scene_layers"]:
        assert "id" in sl
        assert "depth" in sl
        assert "zOrder" in sl
        assert 0.0 <= sl["depth"] <= 1.0

    # Run 2: Cache hit
    res2 = stage.execute(in_dir, out_dir, config={})
    assert res2.success is True
    assert res2.cached is True
