"""Unit tests for the Vectorize stage (adaptive line extraction, screentone separation, Potrace, SVG generation)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mangavrse.stages.vectorize.extractor import extract_lines_and_tones
from mangavrse.stages.vectorize.fidelity import compute_fidelity_score, rasterize_vector_paths
from mangavrse.stages.vectorize.optimizer import PathOptimizer
from mangavrse.stages.vectorize.preview import generate_preview_4x
from mangavrse.stages.vectorize.stage import VectorizeStage
from mangavrse.stages.vectorize.svg_writer import SVGWriter
from mangavrse.stages.vectorize.tracer import PotraceVectorProvider, VectorPath, VectorSegment


def create_synthetic_panel_for_vectorize(w: int = 150, h: int = 150) -> np.ndarray:
    """Creates a panel with solid ink lines and a halftone dot patch."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # Border
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (0, 0, 0), 3)
    # Circle ink contour
    cv2.circle(img, (w // 2, 50), 25, (0, 0, 0), 3)

    # Screentone patch (isolated dots)
    for y in range(90, h - 20, 6):
        for x in range(20, w - 20, 6):
            img[y : y + 2, x : x + 2] = (0, 0, 0)

    return img


class TestVectorizeModules:
    def test_line_and_screentone_separation(self):
        img = create_synthetic_panel_for_vectorize(150, 150)
        line_mask, tone_mask, tone_density = extract_lines_and_tones(img, screentone_separation=True)

        assert line_mask.shape == (150, 150)
        assert tone_mask.shape == (150, 150)

        # Line mask should capture circle contour at y=50
        assert line_mask[50, 75] == 255 or line_mask[50, 50] == 255
        # Tone mask should capture region around y=100
        assert np.any(tone_mask[90:130, 20:130] == 255)

    def test_potrace_tracer(self):
        binary = np.zeros((100, 100), dtype=bool)
        binary[20:80, 20:80] = True

        tracer = PotraceVectorProvider(turdsize=2)
        paths = tracer.predict(binary)

        assert len(paths) >= 1
        assert paths[0].is_closed is True
        assert len(paths[0].segments) >= 4

    def test_path_optimizer_speck_removal(self):
        # Create a large path and a microscopic speck path
        large_path = VectorPath(
            start_point=(10.0, 10.0),
            segments=[
                VectorSegment(is_corner=True, end_point=(50.0, 10.0)),
                VectorSegment(is_corner=True, end_point=(50.0, 50.0)),
                VectorSegment(is_corner=True, end_point=(10.0, 50.0)),
            ],
            area=1600.0,
        )
        speck_path = VectorPath(
            start_point=(0.0, 0.0),
            segments=[
                VectorSegment(is_corner=True, end_point=(1.0, 1.0)),
            ],
            area=2.0,  # Below min_path_area threshold
        )

        optimizer = PathOptimizer(min_path_area=10.0)
        filtered = optimizer.optimize([large_path, speck_path])

        assert len(filtered) == 1
        assert filtered[0].area == 1600.0

    def test_svg_writer(self, tmp_path: Path):
        path = VectorPath(
            start_point=(10.0, 10.0),
            segments=[
                VectorSegment(is_corner=True, end_point=(60.0, 10.0)),
                VectorSegment(is_corner=True, end_point=(60.0, 60.0)),
                VectorSegment(is_corner=True, end_point=(10.0, 60.0)),
            ],
            area=2500.0,
        )

        lines_svg = tmp_path / "lines.svg"
        p_count, n_count, f_size = SVGWriter.write_lines_svg([path], 100, 100, lines_svg)
        assert p_count == 1
        assert n_count >= 4
        assert f_size > 0
        assert "<svg" in lines_svg.read_text(encoding="utf-8")

        tones_svg = tmp_path / "tones.svg"
        tp_count, tn_count, t_size = SVGWriter.write_tones_svg([path], 100, 100, tones_svg)
        assert "<pattern id=\"manga_screentone\"" in tones_svg.read_text(encoding="utf-8")

    def test_fidelity_metrics(self):
        src_mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(src_mask, (20, 20), (80, 80), 255, -1)

        tracer = PotraceVectorProvider()
        paths = tracer.predict(src_mask > 127)

        scores = compute_fidelity_score(src_mask, paths, 100, 100)
        assert scores["iou"] > 0.85
        assert scores["f1_score"] > 0.90
        assert scores["fidelity_score"] > 0.85

    def test_preview_4x_generation(self):
        img = create_synthetic_panel_for_vectorize(160, 160)
        tracer = PotraceVectorProvider()
        paths = tracer.predict(img[:, :, 0] < 127)

        preview = generate_preview_4x(img, paths, crop_size=80, zoom_scale=4)
        assert preview.shape[0] == (80 * 4) + 60
        assert preview.shape[1] == (80 * 4 * 2) + 6


class TestVectorizeStage:
    def test_stage_run_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "enhanced_in"
        out_dir = tmp_path / "vector_out"
        in_dir.mkdir()

        img = create_synthetic_panel_for_vectorize(150, 150)
        cv2.imwrite(str(in_dir / "enhanced.png"), img)

        stage = VectorizeStage()
        config = {
            "vectorize": {
                "extractor": {"screentone_separation": True},
                "optimizer": {"min_path_area": 10.0},
            }
        }

        # Run 1
        res1 = stage.execute(in_dir, out_dir, config)
        assert res1.success is True
        assert res1.cached is False

        assert (out_dir / "lines.svg").exists()
        assert (out_dir / "tones.svg").exists()
        assert (out_dir / "preview_4x.png").exists()
        assert (out_dir / "stats.json").exists()

        stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
        assert "lines" in stats
        assert "tones" in stats
        assert "fidelity" in stats
        assert stats["fidelity"]["fidelity_score"] > 0.70

        # Run 2: should be cached
        res2 = stage.execute(in_dir, out_dir, config)
        assert res2.success is True
        assert res2.cached is True
