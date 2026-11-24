"""Tests for the Ingest stage (PDF extraction, panel detection, reading order, studio overlay)."""
from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

from mangavrse.stages.ingest.classical_detector import ClassicalPanelDetector, PanelBox
from mangavrse.stages.ingest.page_classifier import PageClassifier
from mangavrse.stages.ingest.pdf_extractor import PDFExtractor
from mangavrse.stages.ingest.reading_order import sort_reading_order
from mangavrse.stages.ingest.stage import IngestStage
from mangavrse.stages.ingest.studio_overlay import StudioOverlayHelper


# ------------------------------------------------------------------
# Helpers to generate synthetic manga pages & PDFs
# ------------------------------------------------------------------

def create_synthetic_manga_image(width: int = 1200, height: int = 1800) -> np.ndarray:
    """Creates a synthetic grayscale manga page with 4 panels (2 rows x 2 cols) separated by white gutters."""
    # White background (gutters)
    page = np.full((height, width, 3), 255, dtype=np.uint8)

    # Margins and gutters
    mx = 60
    my = 60
    gx = 40  # vertical gutter
    gy = 50  # horizontal gutter

    panel_w = (width - 2 * mx - gx) // 2
    panel_h = (height - 2 * my - gy) // 2

    # Panel 1: Top-Right
    p1_x, p1_y = mx + panel_w + gx, my
    cv2.rectangle(page, (p1_x, p1_y), (p1_x + panel_w, p1_y + panel_h), (0, 0, 0), 4)
    cv2.circle(page, (p1_x + 100, p1_y + 100), 40, (50, 50, 50), -1)

    # Panel 2: Top-Left
    p2_x, p2_y = mx, my
    cv2.rectangle(page, (p2_x, p2_y), (p2_x + panel_w, p2_y + panel_h), (0, 0, 0), 4)
    cv2.line(page, (p2_x + 20, p2_y + 20), (p2_x + 200, p2_y + 200), (40, 40, 40), 3)

    # Panel 3: Bottom-Right
    p3_x, p3_y = mx + panel_w + gx, my + panel_h + gy
    cv2.rectangle(page, (p3_x, p3_y), (p3_x + panel_w, p3_y + panel_h), (0, 0, 0), 4)
    cv2.rectangle(page, (p3_x + 50, p3_y + 50), (p3_x + 150, p3_y + 150), (120, 120, 120), -1)

    # Panel 4: Bottom-Left
    p4_x, p4_y = mx, my + panel_h + gy
    cv2.rectangle(page, (p4_x, p4_y), (p4_x + panel_w, p4_y + panel_h), (0, 0, 0), 4)

    return page


def create_synthetic_pdf(output_pdf_path: Path) -> Path:
    """Creates a 2-page PDF: Page 1 is an embedded image (lossless scan test), Page 2 has vector content."""
    doc = fitz.open()

    # Page 1: Single embedded full-page raster image
    img_arr = create_synthetic_manga_image(1200, 1800)
    _, buf = cv2.imencode(".png", img_arr)
    img_bytes = buf.tobytes()

    page1 = doc.new_page(width=600, height=900)
    rect = fitz.Rect(0, 0, 600, 900)
    page1.insert_image(rect, stream=img_bytes)

    # Page 2: Vector text/shapes (tests 300 DPI fallback rendering)
    page2 = doc.new_page(width=600, height=900)
    page2.draw_rect(fitz.Rect(50, 50, 550, 850), color=(0, 0, 0), width=2)
    page2.insert_text(fitz.Point(100, 100), "Vector Manga Page", fontsize=24)

    doc.save(str(output_pdf_path))
    doc.close()
    return output_pdf_path


# ------------------------------------------------------------------
# Test PDF Extractor
# ------------------------------------------------------------------

class TestPDFExtractor:
    def test_extract_pdf_pages(self, tmp_path: Path):
        pdf_path = tmp_path / "test_manga.pdf"
        create_synthetic_pdf(pdf_path)

        extractor = PDFExtractor(target_dpi=300)
        out_dir = tmp_path / "extracted_pages"
        pages = extractor.extract_pages(pdf_path, out_dir)

        assert len(pages) == 2
        assert pages[0].page_number == 1
        assert pages[0].image_path.exists()
        # Page 1 should be extracted losslessly from the single embedded image
        assert pages[0].is_lossless_extraction is True

        # Page 2 should be rendered at target DPI
        assert pages[1].page_number == 2
        assert pages[1].image_path.exists()
        assert pages[1].is_lossless_extraction is False


# ------------------------------------------------------------------
# Test Page Classifier
# ------------------------------------------------------------------

class TestPageClassifier:
    def test_story_page(self, tmp_path: Path):
        img_arr = create_synthetic_manga_image()
        img_path = tmp_path / "story.png"
        cv2.imwrite(str(img_path), img_arr)

        classifier = PageClassifier()
        res = classifier.classify(img_path)
        assert res.is_story_page is True
        assert res.page_type == "story"

    def test_color_page_detection(self, tmp_path: Path):
        # Create high-saturation color image
        color_img = np.full((800, 600, 3), (0, 120, 255), dtype=np.uint8)
        img_path = tmp_path / "cover.png"
        cv2.imwrite(str(img_path), color_img)

        classifier = PageClassifier()
        res = classifier.classify(img_path)
        assert res.is_story_page is False
        assert res.page_type == "color_page"
        assert "high_color_saturation" in res.flags

    def test_blank_page_detection(self, tmp_path: Path):
        blank_img = np.full((800, 600, 3), 255, dtype=np.uint8)
        img_path = tmp_path / "blank.png"
        cv2.imwrite(str(img_path), blank_img)

        classifier = PageClassifier()
        res = classifier.classify(img_path)
        assert res.is_story_page is False
        assert res.page_type == "blank"
        assert "blank_page" in res.flags


# ------------------------------------------------------------------
# Test Classical Panel Detector
# ------------------------------------------------------------------

class TestClassicalPanelDetector:
    def test_detect_grid_panels(self):
        page = create_synthetic_manga_image(1200, 1800)
        detector = ClassicalPanelDetector()
        panels = detector.detect_panels(page)

        # Should detect at least the 4 rectangular panels
        assert len(panels) >= 4

    def test_detect_diagonal_panel(self):
        # Page with a slanted/diagonal panel polygon
        page = np.full((1000, 800, 3), 255, dtype=np.uint8)
        # Trapezoid / diagonal contour
        pts = np.array([[100, 100], [700, 180], [650, 600], [100, 500]], np.int32)
        cv2.fillPoly(page, [pts], (20, 20, 20))
        cv2.polylines(page, [pts], True, (0, 0, 0), 4)

        detector = ClassicalPanelDetector()
        panels = detector.detect_panels(page)
        assert len(panels) >= 1
        # At least one panel should be detected as diagonal
        assert any(p.is_diagonal for p in panels)


# ------------------------------------------------------------------
# Test Reading Order
# ------------------------------------------------------------------

class TestReadingOrder:
    def test_rtl_reading_order(self):
        # 4 panels:
        # Tier 1: p1 (right, x=600), p2 (left, x=100) -> top y=50
        # Tier 2: p3 (right, x=600), p4 (left, x=100) -> bottom y=600
        p1 = PanelBox(id="p1", bbox=[600, 50, 400, 400], polygon=[], area=160000)
        p2 = PanelBox(id="p2", bbox=[100, 50, 400, 400], polygon=[], area=160000)
        p3 = PanelBox(id="p3", bbox=[600, 600, 400, 400], polygon=[], area=160000)
        p4 = PanelBox(id="p4", bbox=[100, 600, 400, 400], polygon=[], area=160000)

        # Unordered input
        input_panels = [p4, p1, p3, p2]

        sorted_rtl = sort_reading_order(input_panels, direction="rtl")
        # In RTL manga: Tier 1 right (p1) -> Tier 1 left (p2) -> Tier 2 right (p3) -> Tier 2 left (p4)
        assert [p.bbox[0] for p in sorted_rtl] == [600, 100, 600, 100]
        assert [p.reading_order for p in sorted_rtl] == [0, 1, 2, 3]
        assert sorted_rtl[0].id == "panel_001"
        assert sorted_rtl[3].id == "panel_004"

    def test_ltr_reading_order(self):
        p1 = PanelBox(id="p1", bbox=[600, 50, 400, 400], polygon=[], area=160000)
        p2 = PanelBox(id="p2", bbox=[100, 50, 400, 400], polygon=[], area=160000)

        sorted_ltr = sort_reading_order([p1, p2], direction="ltr")
        # In LTR western: Left (p2, x=100) -> Right (p1, x=600)
        assert sorted_ltr[0].bbox[0] == 100
        assert sorted_ltr[1].bbox[0] == 600


# ------------------------------------------------------------------
# Test Studio Overlay & Manual Override
# ------------------------------------------------------------------

class TestStudioOverlay:
    def test_draw_overlay(self):
        page = create_synthetic_manga_image(800, 1000)
        panels = [
            PanelBox(id="panel_001", bbox=[450, 50, 300, 400], polygon=[], area=120000, reading_order=0),
            PanelBox(id="panel_002", bbox=[50, 50, 300, 400], polygon=[], area=120000, reading_order=1),
        ]
        overlay = StudioOverlayHelper.draw_debug_overlay(page, panels)
        assert overlay.shape == page.shape

    def test_manual_override(self, tmp_path: Path):
        page_dir = tmp_path / "page_001"
        page_dir.mkdir()

        # Write manual override file
        override_data = {
            "panels": [
                {"id": "panel_custom_1", "bbox": [10, 10, 200, 200], "reading_order": 0},
                {"id": "panel_custom_2", "bbox": [250, 10, 200, 200], "reading_order": 1},
            ]
        }
        (page_dir / "panels.override.json").write_text(json.dumps(override_data), encoding="utf-8")

        initial_panels = [
            PanelBox(id="panel_001", bbox=[0, 0, 50, 50], polygon=[], area=2500),
        ]
        final_panels, is_overridden = StudioOverlayHelper.check_and_apply_override(page_dir, initial_panels)

        assert is_overridden is True
        assert len(final_panels) == 2
        assert final_panels[0].id == "panel_custom_1"
        assert final_panels[1].id == "panel_custom_2"


# ------------------------------------------------------------------
# Test IngestStage End-to-End & Caching
# ------------------------------------------------------------------

class TestIngestStage:
    def test_ingest_stage_pdf_run_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "input"
        out_dir = tmp_path / "output"
        in_dir.mkdir()

        pdf_path = in_dir / "manga.pdf"
        create_synthetic_pdf(pdf_path)

        stage = IngestStage()
        r1 = stage.execute(in_dir, out_dir, {"reading_order": "rtl"})

        assert r1.success is True
        assert r1.cached is False
        assert r1.metadata["pages_processed"] == 2
        assert r1.metadata["total_panels"] > 0

        # Check output directory layout
        assert (out_dir / "page_001" / "page_001.png").exists()
        assert (out_dir / "page_001" / "debug_panels.png").exists()
        assert (out_dir / "page_001" / "panels.json").exists()
        assert (out_dir / "page_001" / "panel_001" / "original.png").exists()

        # Verify panels.json content
        panels_json = json.loads((out_dir / "page_001" / "panels.json").read_text(encoding="utf-8"))
        assert panels_json["page"] == 1
        assert panels_json["direction"] == "rtl"
        assert len(panels_json["panels"]) > 0

        # Second run should hit cache
        r2 = stage.execute(in_dir, out_dir, {"reading_order": "rtl"})
        assert r2.success is True
        assert r2.cached is True

    def test_ingest_stage_direct_images(self, tmp_path: Path):
        in_dir = tmp_path / "input_imgs"
        out_dir = tmp_path / "output_imgs"
        in_dir.mkdir()

        img = create_synthetic_manga_image()
        cv2.imwrite(str(in_dir / "page_001.png"), img)

        stage = IngestStage()
        res = stage.execute(in_dir, out_dir, {})
        assert res.success is True
        assert res.metadata["pages_processed"] == 1
        assert (out_dir / "page_001" / "panel_001" / "original.png").exists()
