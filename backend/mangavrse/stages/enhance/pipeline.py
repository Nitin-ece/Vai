"""
Enhancement Pipeline: coordinates contrast, denoise, deblur, upscaling, and sharpening.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from mangavrse.models.realesrgan import RealESRGANAnimeProvider

from .metrics import compute_sharpness_metrics
from .post_processors import apply_line_sharpening
from .pre_processors import apply_contrast, apply_deblur, apply_denoise
from .tiler import SeamlessTiler


@dataclass
class PipelineResult:
    enhanced_image: np.ndarray
    original_metrics: dict[str, Any]
    enhanced_metrics: dict[str, Any]
    step_timings_ms: dict[str, float]
    applied_steps: list[str] = field(default_factory=list)


class EnhancePipeline:
    """Configurable image enhancement pipeline with screentone preservation and seamless tiling."""

    def __init__(self, model_provider: RealESRGANAnimeProvider | None = None) -> None:
        self.model_provider = model_provider or RealESRGANAnimeProvider()
        self.tiler = SeamlessTiler(tile_size=512, tile_pad=32)

    def run(self, image_bgr: np.ndarray, config: dict[str, Any]) -> PipelineResult:
        cfg = config.get("enhance", config)
        enabled = cfg.get("enabled_steps", {})

        timings: dict[str, float] = {}
        applied: list[str] = []

        orig_metrics = {
            "resolution": [int(image_bgr.shape[1]), int(image_bgr.shape[0])],
            **compute_sharpness_metrics(image_bgr),
        }

        current = image_bgr.copy()

        # 1. Contrast Correction
        if enabled.get("contrast", True):
            t0 = time.perf_counter()
            c_cfg = cfg.get("contrast", {})
            current = apply_contrast(
                current,
                method=c_cfg.get("method", "percentile_stretch"),
                white_percentile=c_cfg.get("paper_white_percentile", 99.0),
                black_percentile=c_cfg.get("ink_black_percentile", 1.0),
            )
            timings["contrast_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            applied.append("contrast")

        # 2. Screentone-Aware Denoising
        if enabled.get("denoise", True):
            t0 = time.perf_counter()
            d_cfg = cfg.get("denoise", {})
            current = apply_denoise(
                current,
                method=d_cfg.get("method", "nlmeans"),
                h=d_cfg.get("h", 3.0),
                preserve_screentone=d_cfg.get("preserve_screentone", True),
                screentone_threshold=d_cfg.get("screentone_threshold", 0.12),
            )
            timings["denoise_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            applied.append("denoise")

        # 3. Deblurring
        if enabled.get("deblur", True):
            t0 = time.perf_counter()
            db_cfg = cfg.get("deblur", {})
            current = apply_deblur(
                current,
                radius=db_cfg.get("radius", 1.0),
                amount=db_cfg.get("amount", 0.6),
            )
            timings["deblur_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            applied.append("deblur")

        # 4. Upscaling via RealESRGAN & Seamless Tiling
        if enabled.get("upscale", True):
            t0 = time.perf_counter()
            u_cfg = cfg.get("upscale", {})
            target_scale = int(u_cfg.get("scale", 4))
            use_neural = bool(u_cfg.get("use_neural", True))

            if not use_neural:
                h, w = current.shape[:2]
                current = cv2.resize(current, (w * target_scale, h * target_scale), interpolation=cv2.INTER_CUBIC)
            else:
                tile_size = int(u_cfg.get("tile_size", 512))
                tile_pad = int(u_cfg.get("tile_pad", 32))

                # Configure tiler
                self.tiler.tile_size = tile_size
                self.tiler.tile_pad = tile_pad

                def _upscale_tile(tile: np.ndarray) -> np.ndarray:
                    pre = self.model_provider.preprocess(tile)
                    pred = self.model_provider.predict(pre)
                    return self.model_provider.postprocess(pred)

                # Native RealESRGAN is 4x
                upscaled_4x = self.tiler.process(current, _upscale_tile, scale=4)

                if target_scale == 2:
                    # Downsample 4x to 2x using area interpolation for maximum sharpness
                    h, w = current.shape[:2]
                    current = cv2.resize(upscaled_4x, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
                else:
                    current = upscaled_4x

            timings["upscale_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            applied.append("upscale")

        # 5. Line Sharpening
        if enabled.get("sharpen", True):
            t0 = time.perf_counter()
            s_cfg = cfg.get("sharpen", {})
            current = apply_line_sharpening(
                current,
                amount=s_cfg.get("amount", 0.7),
                radius=s_cfg.get("radius", 0.8),
            )
            timings["sharpen_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            applied.append("sharpen")

        enh_metrics = {
            "resolution": [int(current.shape[1]), int(current.shape[0])],
            **compute_sharpness_metrics(current),
        }

        # Calculate sharpness gain
        orig_lap = max(0.01, orig_metrics["laplacian_variance"])
        enh_lap = enh_metrics["laplacian_variance"]
        enh_metrics["sharpness_gain_ratio"] = round(enh_lap / orig_lap, 2)

        return PipelineResult(
            enhanced_image=current,
            original_metrics=orig_metrics,
            enhanced_metrics=enh_metrics,
            step_timings_ms=timings,
            applied_steps=applied,
        )
