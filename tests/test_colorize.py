"""
Unit tests for the COLORIZE stage:
- Palette manager & character palette storage
- Vector SVG styling & fill injection
- Acceptance test: changing hair color affects ONLY the hair layer
- Acceptance test: cross-panel character palette reuse
- Line art and screentone preservation (colors go under lines)
- End-to-end stage execution & caching
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mangavrse.stages.colorize.colorizer import ColorizerProvider
from mangavrse.stages.colorize.palette_manager import PaletteManager
from mangavrse.stages.colorize.stage import ColorizeStage
from mangavrse.stages.colorize.vector_styler import apply_color_to_svg, assemble_colored_panel_svg, style_layer_svg


def create_synthetic_panel_for_colorize(tmp_dir: Path, w: int = 150, h: int = 150) -> tuple[Path, Path]:
    """Sets up a panel folder containing enhanced.png, layers_manifest.json, layers/*.png, layers/*.svg."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (0, 0, 0), 2)
    # Hair
    cv2.ellipse(img, (75, 40), (25, 15), 0, 0, 360, (50, 50, 50), -1)
    # Face
    cv2.circle(img, (75, 60), 20, (200, 200, 200), -1)
    # Clothing
    cv2.rectangle(img, (50, 80), (100, 140), (120, 120, 120), -1)

    img_path = tmp_dir / "enhanced.png"
    cv2.imwrite(str(img_path), img)

    # Line art SVG
    lines_svg_path = tmp_dir / "lines.svg"
    lines_svg_path.write_text(
        f'<svg viewBox="0 0 {w} {h}"><path d="M 5 5 L 145 5 L 145 145 L 5 145 Z" stroke="#000" fill="none"/></svg>',
        encoding="utf-8",
    )

    layers_dir = tmp_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    # Hair layer SVG & PNG
    hair_mask = np.zeros((h, w, 4), dtype=np.uint8)
    hair_mask[25:55, 50:100] = [50, 50, 50, 255]
    cv2.imwrite(str(layers_dir / "seg_hair.png"), hair_mask)
    (layers_dir / "seg_hair.svg").write_text(
        f'<svg viewBox="0 0 {w} {h}"><polygon points="50,25 100,25 100,55 50,55" /></svg>',
        encoding="utf-8",
    )

    # Face layer SVG & PNG
    face_mask = np.zeros((h, w, 4), dtype=np.uint8)
    face_mask[50:80, 55:95] = [200, 200, 200, 255]
    cv2.imwrite(str(layers_dir / "seg_face.png"), face_mask)
    (layers_dir / "seg_face.svg").write_text(
        f'<svg viewBox="0 0 {w} {h}"><polygon points="55,50 95,50 95,80 55,80" /></svg>',
        encoding="utf-8",
    )

    # Clothing layer SVG & PNG
    cloth_mask = np.zeros((h, w, 4), dtype=np.uint8)
    cloth_mask[80:140, 50:100] = [120, 120, 120, 255]
    cv2.imwrite(str(layers_dir / "seg_clothing.png"), cloth_mask)
    (layers_dir / "seg_clothing.svg").write_text(
        f'<svg viewBox="0 0 {w} {h}"><polygon points="50,80 100,80 100,140 50,140" /></svg>',
        encoding="utf-8",
    )

    manifest_path = tmp_dir / "layers_manifest.json"
    manifest_payload = {
        "panel": {"panel_id": "test_panel", "width": w, "height": h},
        "layers": [
            {
                "id": "seg_hair",
                "label": "character_part",
                "part": "hair",
                "suggested_depth": 0.76,
                "suggested_z_order": 3,
                "svgPath": "layers/seg_hair.svg",
                "pngPath": "layers/seg_hair.png",
            },
            {
                "id": "seg_face",
                "label": "character_part",
                "part": "head",
                "suggested_depth": 0.73,
                "suggested_z_order": 2,
                "svgPath": "layers/seg_face.svg",
                "pngPath": "layers/seg_face.png",
            },
            {
                "id": "seg_clothing",
                "label": "character_part",
                "part": "clothing",
                "suggested_depth": 0.68,
                "suggested_z_order": 1,
                "svgPath": "layers/seg_clothing.svg",
                "pngPath": "layers/seg_clothing.png",
            },
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return img_path, manifest_path


class TestColorizeModules:
    def test_palette_manager_persistence(self, tmp_path: Path):
        pm = PaletteManager()
        pm.set_character_color("char_001", "hair", "#FF0000")
        save_p = tmp_path / "palette.json"
        pm.save(save_p)

        loaded_pm = PaletteManager.load(save_p)
        assert loaded_pm.resolve_layer_color("character_part", "hair", "char_001") == "#FF0000"

    def test_hair_color_isolation(self):
        """ACCEPTANCE: changing hair color affects only the hair layer."""
        pm = PaletteManager()
        initial_face_color = pm.resolve_layer_color("character_part", "head", "char_001")
        initial_clothing_color = pm.resolve_layer_color("character_part", "clothing", "char_001")

        # Change hair color to golden blonde
        pm.set_character_color("char_001", "hair", "#FFD700")

        # Hair color is updated
        assert pm.resolve_layer_color("character_part", "hair", "char_001") == "#FFD700"
        # Face and clothing colors remain strictly unchanged
        assert pm.resolve_layer_color("character_part", "head", "char_001") == initial_face_color
        assert pm.resolve_layer_color("character_part", "clothing", "char_001") == initial_clothing_color

    def test_cross_panel_palette_reuse(self, tmp_path: Path):
        """ACCEPTANCE: applying saved palette to a second panel of the same character."""
        # Panel 1: save a custom palette for 'protagonist'
        pm1 = PaletteManager()
        pm1.set_character_color("protagonist", "hair", "#9C27B0")  # Anime Purple
        pm1.set_character_color("protagonist", "clothing_primary", "#009688")  # Teal
        pal_file = tmp_path / "char_palette.json"
        pm1.save(pal_file)

        # Panel 2: load palette and query colors for 'protagonist'
        pm2 = PaletteManager.load(pal_file)
        color_hair_panel2 = pm2.resolve_layer_color("character_part", "hair", character_id="protagonist")
        color_clothing_panel2 = pm2.resolve_layer_color("character_part", "clothing", character_id="protagonist")

        assert color_hair_panel2 == "#9C27B0"
        assert color_clothing_panel2 == "#009688"

    def test_vector_styler_fill_injection(self):
        raw_svg = '<svg viewBox="0 0 100 100"><polygon points="10,10 90,10 90,90 10,90" /></svg>'
        styled = apply_color_to_svg(raw_svg, "#E91E63", opacity=0.9)
        assert '#E91E63' in styled
        assert 'fill-opacity="0.9"' in styled

    def test_line_art_and_tone_layering(self, tmp_path: Path):
        """Verifies line art is preserved on top of color fills."""
        layers = [{"id": "layer_001", "suggested_z_order": 1, "svgPath": "layers/layer_001.svg"}]
        (tmp_path / "layers").mkdir()
        (tmp_path / "layers" / "layer_001.svg").write_text('<svg><polygon points="0,0 50,50" /></svg>')
        lines_p = tmp_path / "lines.svg"
        lines_p.write_text('<svg><path id="ink_lines" d="M 0 0 L 10 10" /></svg>')

        out_svg = tmp_path / "colored_panel.svg"
        assemble_colored_panel_svg(
            width=100,
            height=100,
            layers=layers,
            layer_colors={"layer_001": "#FF5722"},
            out_svg_path=out_svg,
            lines_svg_path=lines_p,
        )

        svg_text = out_svg.read_text(encoding="utf-8")
        # Line art group must appear AFTER color fills in the SVG DOM order
        color_idx = svg_text.find('id="color_fills"')
        line_idx = svg_text.find('id="line_art"')
        assert color_idx != -1
        assert line_idx != -1
        assert line_idx > color_idx


class TestColorizeStage:
    def test_stage_run_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "panel_in"
        out_dir = tmp_path / "panel_out"
        in_dir.mkdir()

        create_synthetic_panel_for_colorize(in_dir, 150, 150)

        stage = ColorizeStage()
        config = {
            "colorize": {
                "character_id": "char_001",
                "overrides": {"seg_hair": "#D32F2F"},  # Explicit red hair override
            }
        }

        # Run 1
        res1 = stage.execute(in_dir, out_dir, config)
        assert res1.success is True
        assert res1.cached is False

        assert (out_dir / "colored_panel.svg").exists()
        assert (out_dir / "colored_panel.png").exists()
        assert (out_dir / "comparison.png").exists()
        assert (out_dir / "palette.json").exists()
        assert (out_dir / "colored_layers").exists()

        # Check hair layer SVG has override color
        hair_svg_text = (out_dir / "colored_layers" / "seg_hair.svg").read_text(encoding="utf-8")
        assert "#D32F2F" in hair_svg_text

        # Run 2: should be cached
        res2 = stage.execute(in_dir, out_dir, config)
        assert res2.success is True
        assert res2.cached is True
