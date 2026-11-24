"""
Unit tests for the SEGMENT stage:
- Screentone smoothing & preprocessors
- Speech bubble and SFX isolation
- Line art snapping
- Semantic classification
- Hierarchy matching scene.json schema
- Interactive click prompt refinement
- End-to-end stage execution & caching
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from mangavrse.stages.segment.classifier import classify_segment
from mangavrse.stages.segment.hierarchy import build_segment_hierarchy
from mangavrse.stages.segment.line_snapper import snap_mask_to_lines
from mangavrse.stages.segment.preprocessor import detect_candidate_bubbles_and_sfx, smooth_for_segmentation
from mangavrse.stages.segment.refiner import refine_mask_with_prompts
from mangavrse.stages.segment.sam_provider import SAM2SegmenterProvider
from mangavrse.stages.segment.stage import SegmentStage
from mangavrse.stages.segment.visualizer import create_segment_overlay


def create_synthetic_manga_panel(w: int = 200, h: int = 200) -> np.ndarray:
    """Creates a synthetic panel containing a character silhouette, screentone, and a speech bubble."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Panel border
    cv2.rectangle(img, (5, 5), (w - 5, h - 5), (0, 0, 0), 2)

    # Character body (torso and head)
    # Head (circle)
    cv2.circle(img, (100, 70), 25, (40, 40, 40), -1)
    # Hair on top of head (darker spiky patch)
    cv2.ellipse(img, (100, 55), (25, 12), 0, 180, 360, (10, 10, 10), -1)
    # Body (torso rectangle)
    cv2.rectangle(img, (75, 95), (125, 160), (80, 80, 80), -1)
    cv2.rectangle(img, (75, 95), (125, 160), (0, 0, 0), 2)

    # Speech bubble (white oval with text lines)
    cv2.ellipse(img, (45, 50), (30, 20), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (45, 50), (30, 20), 0, 0, 360, (0, 0, 0), 2)
    # Fake text strokes inside bubble
    cv2.line(img, (30, 47), (60, 47), (0, 0, 0), 1)
    cv2.line(img, (32, 53), (58, 53), (0, 0, 0), 1)

    # Screentone patch in background
    for y in range(110, 180, 5):
        for x in range(10, 65, 5):
            img[y : y + 2, x : x + 2] = (0, 0, 0)

    return img


class TestSegmentModules:
    def test_screentone_smoothing(self):
        img = create_synthetic_manga_panel(200, 200)
        smoothed = smooth_for_segmentation(img, screentone_smoothing=True)
        assert smoothed.shape == img.shape
        # Screentone patch variance should be lower in smoothed image
        raw_patch = img[110:180, 10:65]
        smooth_patch = smoothed[110:180, 10:65]
        assert np.std(smooth_patch) <= np.std(raw_patch)

    def test_bubble_detection(self):
        img = create_synthetic_manga_panel(200, 200)
        candidates = detect_candidate_bubbles_and_sfx(img)
        bubble_candidates = [c for c in candidates if c["label"] == "speech_bubble"]
        assert len(bubble_candidates) >= 1
        b = bubble_candidates[0]
        # Centroid of bubble should be near x=45, y=50
        assert b["bbox"][0] <= 45 <= b["bbox"][2]
        assert b["bbox"][1] <= 50 <= b["bbox"][3]
        assert b["confidence"] > 0.75

    def test_line_snapping(self):
        # A mask that bleeds across an ink border
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:70, 20:70] = 255
        lines = np.zeros((100, 100), dtype=np.uint8)
        # Ink line at x=50
        lines[:, 50] = 255

        snapped = snap_mask_to_lines(mask, lines)
        assert snapped.shape == (100, 100)

    def test_semantic_classifier(self):
        img = create_synthetic_manga_panel(200, 200)
        h, w = img.shape[:2]

        # Background mask (large area touching borders)
        bg_mask = np.zeros((h, w), dtype=np.uint8)
        bg_mask[0:h, 0:w] = 255
        lbl, part, conf, review = classify_segment(bg_mask, img)
        assert lbl == "background"
        assert conf >= 0.85

        # Character mask (centered figure)
        char_mask = np.zeros((h, w), dtype=np.uint8)
        char_mask[50:160, 70:130] = 255
        lbl, part, conf, review = classify_segment(char_mask, img)
        assert lbl == "character"
        assert conf >= 0.70

    def test_hierarchy_nesting(self):
        # Parent character and child hair
        char_mask = np.zeros((200, 200), dtype=np.uint8)
        char_mask[50:160, 70:130] = 255
        hair_mask = np.zeros((200, 200), dtype=np.uint8)
        hair_mask[50:70, 80:120] = 255  # completely contained inside character

        segments = [
            {
                "id": "seg_001",
                "label": "character",
                "part": None,
                "area": int(np.count_nonzero(char_mask)),
                "_mask": char_mask,
                "bbox": [70, 50, 130, 160],
            },
            {
                "id": "seg_002",
                "label": "character_part",
                "part": "hair",
                "area": int(np.count_nonzero(hair_mask)),
                "_mask": hair_mask,
                "bbox": [80, 50, 120, 70],
            },
        ]

        hierarchy = build_segment_hierarchy(segments, 200, 200)
        assert len(hierarchy) == 2
        # Hair should have char as parent
        hair_seg = next(s for s in hierarchy if s["id"] == "seg_002")
        assert hair_seg["parentId"] == "seg_001"
        assert hair_seg["suggested_depth"] > next(s for s in hierarchy if s["id"] == "seg_001")["suggested_depth"]

    def test_interactive_prompt_refinement(self):
        img = create_synthetic_manga_panel(200, 200)
        provider = SAM2SegmenterProvider()

        # Positive click inside character head at (100, 70)
        refined_mask, conf, bbox, area = refine_mask_with_prompts(
            image_bgr=img,
            points=[{"x": 100, "y": 70, "label": 1}],
            provider=provider,
        )
        assert refined_mask.shape == (200, 200)
        assert area > 0
        assert refined_mask[70, 100] == 255

    def test_visual_overlay_generation(self):
        img = create_synthetic_manga_panel(200, 200)
        m = np.zeros((200, 200), dtype=np.uint8)
        m[50:100, 50:100] = 255
        segs = [{
            "id": "seg_001",
            "label": "character",
            "part": None,
            "confidence": 0.90,
            "needs_manual_review": False,
            "bbox": [50, 50, 100, 100],
            "area": 2500,
        }]
        overlay = create_segment_overlay(img, segs, {"seg_001": m})
        assert overlay is not None
        assert overlay.shape[0] == 232  # 200 + 32 header bar


class TestSegmentStage:
    def test_stage_run_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "panel_in"
        out_dir = tmp_path / "panel_out"
        in_dir.mkdir()

        img = create_synthetic_manga_panel(200, 200)
        cv2.imwrite(str(in_dir / "enhanced.png"), img)

        stage = SegmentStage()
        config = {
            "segment": {
                "screentone_smoothing": True,
                "bubble_detection": True,
                "line_snapping": True,
            }
        }

        # Run 1
        res1 = stage.execute(in_dir, out_dir, config)
        assert res1.success is True
        assert res1.cached is False

        assert (out_dir / "segments.json").exists()
        assert (out_dir / "overlay.png").exists()
        assert (out_dir / "masks").exists()
        assert len(list((out_dir / "masks").glob("*.png"))) >= 1

        segments_data = json.loads((out_dir / "segments.json").read_text(encoding="utf-8"))
        assert "segments" in segments_data
        assert segments_data["total_segments"] >= 1

        # Run 2: should be cached
        res2 = stage.execute(in_dir, out_dir, config)
        assert res2.success is True
        assert res2.cached is True
