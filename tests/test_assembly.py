"""
Unit and integration tests for the Assembly stage and scene.json schema validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mangavrse.stages.assembly import (
    AssemblyStage,
    assemble_scene,
    load_schema,
    validate_scene,
)


@pytest.fixture
def sample_valid_scene() -> dict:
    """Returns a valid scene structure matching docs/scene_schema.json."""
    return {
        "version": "1.0",
        "panel": {
            "project": "test_manga",
            "page": 1,
            "panel": 2,
            "source": "enhanced.png",
            "width": 1000,
            "height": 1000,
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
        "layers": [
            {
                "id": "bg_001",
                "type": "background",
                "parentId": None,
                "svgPath": "layers/bg_001.svg",
                "pngPath": "layers/bg_001.png",
                "color": "#FFFFFF",
                "depth": 0.1,
                "zOrder": 0,
                "transform": {
                    "x": 0.0,
                    "y": 0.0,
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "rotation": 0.0,
                },
                "metadata": {
                    "bbox": [0, 0, 1000, 1000],
                    "animations": {},
                },
            },
            {
                "id": "char_001",
                "type": "character",
                "parentId": "bg_001",
                "svgPath": "layers/char_001.svg",
                "pngPath": "layers/char_001.png",
                "color": "#E63946",
                "depth": 0.7,
                "zOrder": 1,
                "transform": {
                    "x": 0.0,
                    "y": 0.0,
                    "scaleX": 1.0,
                    "scaleY": 1.0,
                    "rotation": 0.0,
                },
                "metadata": {
                    "part": "body",
                    "bbox": [200, 200, 800, 800],
                    "animations": {
                        "sway": {"enabled": False, "speed": 1.0, "amplitude": 0.02},
                    },
                },
            },
        ],
    }


def test_schema_loading() -> None:
    """Verifies that the JSON schema loads properly from docs/scene_schema.json."""
    schema = load_schema()
    assert schema["$id"] == "https://mangavrse.dev/schemas/scene.v1.json"
    assert "version" in schema["required"]
    assert "layers" in schema["required"]


def test_scene_validation(sample_valid_scene: dict) -> None:
    """Tests schema validation on valid and invalid scene definitions."""
    # Valid scene
    is_valid, err = validate_scene(sample_valid_scene)
    assert is_valid is True
    assert err is None

    # Missing camera
    invalid_scene = dict(sample_valid_scene)
    del invalid_scene["camera"]
    is_valid, err = validate_scene(invalid_scene)
    assert is_valid is False
    assert "camera" in (err or "")

    # Invalid depth out of range
    invalid_depth_scene = dict(sample_valid_scene)
    invalid_depth_scene["layers"] = [
        dict(sample_valid_scene["layers"][0], depth=1.5)
    ]
    is_valid, err = validate_scene(invalid_depth_scene)
    assert is_valid is False
    assert "depth" in (err or "")


def test_assemble_scene_logic(tmp_path: Path) -> None:
    """Tests combining layers, depth, and palette data into scene.json."""
    p_dir = tmp_path / "panel_test"
    p_dir.mkdir()

    layers_data = [
        {"id": "lyr_1", "label": "background", "depth": 0.1, "bbox": [0, 0, 500, 500]},
        {"id": "lyr_2", "label": "character", "part": "hair", "depth": 0.75, "bbox": [100, 100, 300, 300]},
        {"id": "lyr_3", "label": "speech_bubble", "depth": 0.95, "bbox": [50, 50, 150, 150]},
    ]
    palette_data = {
        "overrides": {"lyr_2": "#FF0000"},
        "colors": {"lyr_1": "#FFFFFF"},
    }
    depth_data = {
        "layers": [
            {"id": "lyr_1", "depth": 0.08, "zOrder": 0},
            {"id": "lyr_2", "depth": 0.72, "zOrder": 1},
            {"id": "lyr_3", "depth": 0.94, "zOrder": 2},
        ]
    }

    scene = assemble_scene(
        panel_id="panel_test",
        panel_dir=p_dir,
        layers_data=layers_data,
        palette_data=palette_data,
        depth_data=depth_data,
        width=500,
        height=500,
    )

    assert scene["version"] == "1.0"
    assert len(scene["layers"]) == 3

    # Check hair layer
    hair_lyr = next(l for l in scene["layers"] if l["id"] == "lyr_2")
    assert hair_lyr["color"] == "#FF0000"
    assert hair_lyr["depth"] == 0.72
    assert "sway" in hair_lyr["metadata"]["animations"]
    assert hair_lyr["metadata"]["animations"]["sway"]["enabled"] is False

    # Check speech bubble animation
    bubble_lyr = next(l for l in scene["layers"] if l["id"] == "lyr_3")
    assert "float" in bubble_lyr["metadata"]["animations"]
    assert bubble_lyr["metadata"]["animations"]["float"]["enabled"] is False


def test_assembly_stage_run_and_cache(tmp_path: Path) -> None:
    """Tests execution and caching of AssemblyStage."""
    in_dir = tmp_path / "stage_in"
    out_dir = tmp_path / "stage_out"
    in_dir.mkdir()
    out_dir.mkdir()

    p_in = in_dir / "panel_001"
    p_in.mkdir()

    # Create dummy depth.json and layers_manifest.json
    (p_in / "layers_manifest.json").write_text(
        json.dumps({
            "panel": {"width": 800, "height": 800},
            "layers": [
                {"id": "seg_01", "label": "background", "bbox": [0, 0, 800, 800]},
                {"id": "seg_02", "label": "character", "bbox": [100, 100, 700, 700]},
            ]
        }),
        encoding="utf-8"
    )
    (p_in / "depth.json").write_text(
        json.dumps({
            "layers": [
                {"id": "seg_01", "depth": 0.1, "zOrder": 0},
                {"id": "seg_02", "depth": 0.7, "zOrder": 1},
            ]
        }),
        encoding="utf-8"
    )

    stage = AssemblyStage()

    # Run 1
    res1 = stage.execute(in_dir, out_dir, config={})
    assert res1.success is True
    assert res1.cached is False
    assert (out_dir / "panel_001" / "scene.json").exists()

    # Validate output
    scene_data = json.loads((out_dir / "panel_001" / "scene.json").read_text(encoding="utf-8"))
    is_valid, err = validate_scene(scene_data)
    assert is_valid is True, err

    # Run 2: cached
    res2 = stage.execute(in_dir, out_dir, config={})
    assert res2.success is True
    assert res2.cached is True
