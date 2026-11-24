"""Unit tests for the Enhance stage (contrast, screentone-aware denoise, deblur, upscale, sharpen)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from mangavrse.models.realesrgan import RRDBNet, RealESRGANAnimeProvider
from mangavrse.stages.enhance.comparison import generate_comparison_image
from mangavrse.stages.enhance.metrics import compute_sharpness_metrics
from mangavrse.stages.enhance.pipeline import EnhancePipeline
from mangavrse.stages.enhance.post_processors import apply_line_sharpening
from mangavrse.stages.enhance.pre_processors import (
    apply_contrast,
    apply_deblur,
    apply_denoise,
    detect_screentone_mask,
)
from mangavrse.stages.enhance.stage import EnhanceStage
from mangavrse.stages.enhance.tiler import SeamlessTiler


# ----------------------------------------------------------------------
# Test Synthetic Panel Generators
# ----------------------------------------------------------------------

def create_synthetic_panel_with_screentone(w: int = 200, h: int = 200) -> np.ndarray:
    """Creates a panel with solid ink lines, flat paper background, and a halftone screentone patch."""
    img = np.full((h, w, 3), 240, dtype=np.uint8)  # slightly off-white paper

    # Draw solid black border and ink circle
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (15, 15, 15), 3)
    cv2.circle(img, (w // 2, 60), 30, (20, 20, 20), -1)

    # Draw periodic screentone dot pattern in lower half
    for y in range(110, h - 20, 4):
        for x in range(20, w - 20, 4):
            img[y, x] = (40, 40, 40)
            img[y, x + 1] = (40, 40, 40)

    return img


# ----------------------------------------------------------------------
# Pre & Post Processor Tests
# ----------------------------------------------------------------------

class TestEnhanceProcessors:
    def test_contrast_percentile_stretch(self):
        # Slightly gray background with dark gray ink
        img = np.full((100, 100, 3), 230, dtype=np.uint8)
        img[30:70, 30:70] = (30, 30, 30)

        stretched = apply_contrast(img, method="percentile_stretch", white_percentile=99.0, black_percentile=1.0)
        # Background should stretch near 255 and ink near 0
        assert stretched[10, 10, 0] >= 245
        assert stretched[50, 50, 0] <= 15

    def test_screentone_mask_detection(self):
        img = create_synthetic_panel_with_screentone(200, 200)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = detect_screentone_mask(gray, threshold=0.10)

        # In screentone region (y=130), mask should have high weight
        screentone_zone_weight = float(np.mean(mask[120:170, 30:170]))
        # In flat paper region (y=30, x=30), mask should be near 0
        paper_zone_weight = float(np.mean(mask[20:40, 20:40]))

        assert screentone_zone_weight > paper_zone_weight

    def test_denoise_preserves_screentone(self):
        img = create_synthetic_panel_with_screentone(200, 200)
        # Add slight gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        denoised = apply_denoise(noisy, method="nlmeans", h=3.0, preserve_screentone=True)
        assert denoised.shape == noisy.shape

        # Dot contrast in screentone zone should be preserved
        st_patch = denoised[120:170, 30:170]
        assert float(np.std(st_patch)) > 10.0

    def test_deblur(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.line(img, (20, 50), (80, 50), (0, 0, 0), 4)
        deblurred = apply_deblur(img, radius=1.0, amount=0.5)
        assert deblurred.shape == img.shape

    def test_line_sharpening(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.circle(img, (50, 50), 20, (30, 30, 30), 3)

        orig_sharpness = compute_sharpness_metrics(img)["laplacian_variance"]
        sharpened = apply_line_sharpening(img, amount=0.8, radius=0.8)
        enh_sharpness = compute_sharpness_metrics(sharpened)["laplacian_variance"]

        assert enh_sharpness >= orig_sharpness


# ----------------------------------------------------------------------
# Seamless Tiler Tests
# ----------------------------------------------------------------------

class TestSeamlessTiler:
    def test_tiler_reconstruction_without_seams(self):
        # Create a gradient image larger than tile_size (700 x 700)
        h, w = 700, 700
        y, x = np.mgrid[0:h, 0:w]
        gradient = ((x + y) * 255.0 / (h + w)).astype(np.uint8)
        img = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        tiler = SeamlessTiler(tile_size=256, tile_pad=32)

        # Mock identity 1x process function
        def mock_identity(tile: np.ndarray) -> np.ndarray:
            return tile

        reconstructed = tiler.process(img, mock_identity, scale=1)
        assert reconstructed.shape == img.shape

        # Check maximum absolute difference: should be tiny across smooth gradients
        max_diff = np.max(np.abs(reconstructed.astype(np.int32) - img.astype(np.int32)))
        assert max_diff <= 2


# ----------------------------------------------------------------------
# Model Provider Tests
# ----------------------------------------------------------------------

class TestRealESRGANProvider:
    def test_rrdbnet_forward_shape(self):
        net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=16, num_block=2, num_grow_ch=8)
        dummy_input = torch.zeros(1, 3, 32, 32)
        out = net(dummy_input)
        assert out.shape == (1, 3, 128, 128)

    def test_provider_predict(self):
        provider = RealESRGANAnimeProvider(scale=4)
        provider.load(device="cpu")
        tile = np.zeros((40, 40, 3), dtype=np.uint8)
        pre = provider.preprocess(tile)
        pred = provider.predict(pre)
        post = provider.postprocess(pred)
        assert post.shape == (160, 160, 3)


# ----------------------------------------------------------------------
# EnhanceStage End-to-End & Caching Tests
# ----------------------------------------------------------------------

class TestEnhanceStage:
    def test_enhance_stage_execution_and_cache(self, tmp_path: Path):
        in_dir = tmp_path / "panel_001"
        out_dir = tmp_path / "enhanced_output"
        in_dir.mkdir()

        # Write original.png
        panel_img = create_synthetic_panel_with_screentone(150, 150)
        cv2.imwrite(str(in_dir / "original.png"), panel_img)

        stage = EnhanceStage()
        config = {
            "enhance": {
                "enabled_steps": {
                    "contrast": True,
                    "denoise": True,
                    "deblur": True,
                    "upscale": True,
                    "sharpen": True,
                },
                "upscale": {
                    "scale": 2,  # 2x for fast unit test
                    "tile_size": 256,
                    "tile_pad": 16,
                },
            }
        }

        # Run 1
        res1 = stage.execute(in_dir, out_dir, config)
        assert res1.success is True
        assert res1.cached is False
        assert (out_dir / "enhanced.png").exists()
        assert (out_dir / "comparison.png").exists()
        assert (out_dir / "metrics.json").exists()

        # Check metrics file
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["original"]["resolution"] == [150, 150]
        assert metrics["enhanced"]["resolution"] == [300, 300]
        assert "sharpness_gain_ratio" in metrics["enhanced"]

        # Run 2: should be cached
        res2 = stage.execute(in_dir, out_dir, config)
        assert res2.success is True
        assert res2.cached is True

    def test_toggleable_substeps(self, tmp_path: Path):
        in_dir = tmp_path / "panel_substeps"
        out_dir = tmp_path / "out_substeps"
        in_dir.mkdir()

        img = np.full((80, 80, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(in_dir / "original.png"), img)

        stage = EnhanceStage()
        # Disable upscale and denoise
        cfg = {
            "enhance": {
                "enabled_steps": {
                    "contrast": True,
                    "denoise": False,
                    "deblur": False,
                    "upscale": False,
                    "sharpen": True,
                }
            }
        }
        res = stage.execute(in_dir, out_dir, cfg)
        assert res.success is True
        metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "denoise" not in metrics["applied_steps"]
        assert "upscale" not in metrics["applied_steps"]
        assert "contrast" in metrics["applied_steps"]
        assert "sharpen" in metrics["applied_steps"]
        # Resolution should remain 80x80 because upscale was disabled
        assert metrics["enhanced"]["resolution"] == [80, 80]
