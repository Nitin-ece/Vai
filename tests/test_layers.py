"""
Unit tests for the LAYERS stage:
- Anti-halo mask padding & sub-pixel alpha feathering
- Transparent RGBA and clipped SVG cutout generation
- Manga-aware background inpainting (line extension + screentone synthesis)
- Parallax-gap test render
- Layers manifest structure
- End-to-end stage execution & caching
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mangavrse.stages.layers.cutter import create_clipped_svg, extract_layer_rgba, pad_and_feather_mask
from mangavrse.stages.layers.manga_inpaint import inpaint_manga_background, synthesize_screentone_fill
from mangavrse.stages.layers.parallax_tester import render_parallax_test
from mangavrse.stages.layers.quality import evaluate_inpaint_quality
from mangavrse.stages.layers.stage import LayersStage


def create_synthetic_panel_and_segments(tmp_dir: Path, w: int = 200, h: int = 200) -> tuple[Path, Path]:
    """Generates an enhanced panel image, segments.json, and mask PNGs."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Panel border
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (0, 0, 0), 2)

    # Character body in center
    cv2.circle(img, (100, 70), 25, (40, 40, 40), -1)
    cv2.rectangle(img, (75, 95), (125, 160), (80, 80, 80), -1)

    # Speech bubble on top-left
    cv2.ellipse(img, (45, 50), (30, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (45, 50), (30, 20), 0, 0, 360, (0, 0, 0), 2)

    img_path = tmp_dir / "enhanced.png"
    cv2.imwrite(str(img_path), img)

    # Masks
    masks_dir = tmp_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    char_mask = np.zeros((h, w), dtype=np.uint8)
    char_mask[45:165, 70:130] = 255
    cv2.imwrite(str(masks_dir / "seg_001_character.png"), char_mask)

    bubble_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(bubble_mask, (45, 50), (30, 20), 0, 0, 360, 255, -1)
    cv2.imwrite(str(masks_dir / "seg_002_speech_bubble.png"), bubble_mask)

    segments_json_path = tmp_dir / "segments.json"
    segments_payload = {
        "panel": {"panel_id": "test_panel", "width": w, "height": h, "source": "enhanced.png"},
        "total_segments": 2,
        "segments": [
            {
                "id": "seg_001",
                "label": "character",
                "part": None,
                "parentId": None,
                "confidence": 0.90,
                "needs_manual_review": False,
                "bbox": [70, 45, 130, 165],
                "area": int(np.count_nonzero(char_mask)),
                "mask_path": "masks/seg_001_character.png",
                "suggested_depth": 0.65,
                "suggested_z_order": 2,
            },
            {
                "id": "seg_002",
                "label": "speech_bubble",
                "part": None,
                "parentId": None,
                "confidence": 0.95,
                "needs_manual_review": False,
                "bbox": [15, 30, 75, 70],
                "area": int(np.count_nonzero(bubble_mask)),
                "mask_path": "masks/seg_002_speech_bubble.png",
                "suggested_depth": 0.94,
                "suggested_z_order": 4,
            },
        ],
    }
    segments_json_path.write_text(json.dumps(segments_payload, indent=2), encoding="utf-8")
    return img_path, segments_json_path


class TestLayersModules:
    def test_mask_padding_and_feathering(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        feathered = pad_and_feather_mask(mask, padding=2, feather_radius=2.0)
        assert feathered.shape == (100, 100)
        # Core should remain opaque
        assert feathered[50, 50] >= 250
        # Padded edge should have smooth transition
        assert 0 < feathered[28, 50] < 255
        # Background should be transparent
        assert feathered[10, 10] == 0

    def test_layer_rgba_cutout(self):
        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        rgba = extract_layer_rgba(img, mask, padding=1, feather_radius=1.5)
        assert rgba.shape == (100, 100, 4)
        assert rgba[50, 50, 3] >= 240
        assert rgba[5, 5, 3] == 0

    def test_clipped_svg_generation(self, tmp_path: Path):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)

        out_svg = tmp_path / "layer.svg"
        ok = create_clipped_svg(None, mask, out_svg, 100, 100, "test_lyr")
        assert ok is True
        assert out_svg.exists()
        svg_content = out_svg.read_text(encoding="utf-8")
        assert "<clipPath" in svg_content
        assert "<polygon" in svg_content

    def test_manga_inpaint_screentone_preservation(self):
        # Image with periodic halftone screentone and a hole
        h, w = 150, 150
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        # Fill with dots
        for y in range(0, h, 6):
            for x in range(0, w, 6):
                cv2.circle(img, (x, y), 1, (40, 40, 40), -1)

        occlusion = np.zeros((h, w), dtype=np.uint8)
        occlusion[50:100, 50:100] = 255

        filled = inpaint_manga_background(img, occlusion, screentone_synthesis=True)
        assert filled.shape == (h, w, 3)
        # Filled hole must have variation (dots), not be flat solid gray
        hole_patch = filled[60:90, 60:90]
        assert np.std(hole_patch) > 1.0

    def test_parallax_tester_render(self):
        bg = np.full((120, 120, 3), 240, dtype=np.uint8)
        rgba = np.zeros((120, 120, 4), dtype=np.uint8)
        rgba[40:80, 40:80] = [200, 50, 50, 255]

        layers = [{
            "id": "seg_001",
            "suggested_depth": 0.70,
            "suggested_z_order": 2,
            "rgba": rgba,
        }]

        test_img = render_parallax_test(bg, layers, shift_scale=20.0, angle_degrees=0.0)
        assert test_img is not None
        assert test_img.shape[0] == 152  # 120 + 32 banner


class TestLayersStage:
    def test_stage_run_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "panel_in"
        out_dir = tmp_path / "panel_out"
        in_dir.mkdir()

        create_synthetic_panel_and_segments(in_dir, 200, 200)

        stage = LayersStage()
        config = {
            "layers": {
                "mask_padding": 2,
                "feather_radius": 2.0,
                "inpainting": {"method": "telea_texture"},
                "parallax_test": {"shift_scale": 20.0},
            }
        }

        # Run 1
        res1 = stage.execute(in_dir, out_dir, config)
        assert res1.success is True
        assert res1.cached is False

        assert (out_dir / "background_filled.png").exists()
        assert (out_dir / "parallax_test.png").exists()
        assert (out_dir / "layers_manifest.json").exists()
        assert (out_dir / "layers").exists()
        assert len(list((out_dir / "layers").glob("*.png"))) == 2
        assert len(list((out_dir / "layers").glob("*.svg"))) == 2

        manifest = json.loads((out_dir / "layers_manifest.json").read_text(encoding="utf-8"))
        assert "layers" in manifest
        assert len(manifest["layers"]) == 2
        assert "inpaint_metrics" in manifest

        # Run 2: should be cached
        res2 = stage.execute(in_dir, out_dir, config)
        assert res2.success is True
        assert res2.cached is True
