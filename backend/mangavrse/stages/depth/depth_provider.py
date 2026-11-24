"""
Depth Estimator Provider conforming to the ModelProvider protocol.
Supports Depth Anything V2, MiDaS, and edge-preserving manga depth estimation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from mangavrse.core.model_provider import BaseModelProvider

logger = logging.getLogger(__name__)


class DepthEstimatorProvider(BaseModelProvider):
    """
    Model provider for monocular depth estimation on manga panels.
    Accommodates line art, screentone textures, and high-contrast ink boundaries.
    """

    def __init__(
        self,
        name: str = "depth_anything_v2",
        bilateral_d: int = 7,
        bilateral_sigma: float = 50.0,
        smooth_sigma: float = 1.5,
    ) -> None:
        super().__init__(name)
        self.bilateral_d = bilateral_d
        self.bilateral_sigma = bilateral_sigma
        self.smooth_sigma = smooth_sigma

        self._model: Any = None
        self._transform: Any = None
        self._device: str = "cpu"
        self._is_native: bool = False

    def load(self, device: str = "cpu", weights: str | None = None) -> None:
        """Load Depth Anything / MiDaS model or initialize manga fallback."""
        self._device = device

        # Check for local weights or torch hub availability
        if weights and Path(weights).exists():
            try:
                logger.info(f"Loading depth model weights from {weights} onto {device}")
                checkpoint = torch.load(weights, map_location=device)
                if hasattr(checkpoint, "eval"):
                    self._model = checkpoint.eval()
                    self._is_native = True
                    return
            except Exception as e:
                logger.warning(f"Could not load custom depth weights from {weights}: {e}")

        # Check torch hub for MiDaS / Depth Anything if explicitly requested and online
        if "midas" in self.name.lower():
            try:
                self._model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
                self._model.to(device)
                self._model.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                self._transform = midas_transforms.small_transform
                self._is_native = True
                logger.info("Loaded MiDaS small model from torch.hub")
                return
            except Exception as e:
                logger.info(f"Native MiDaS load skipped ({e}); using manga depth estimator.")
        elif "depth_anything" in self.name.lower():
            try:
                self._model = torch.hub.load("LiheYoung/depth_anything_v2", "depth_anything_v2_vits", pretrained=True, trust_repo=True)
                self._model.to(device)
                self._model.eval()
                self._is_native = True
                logger.info("Loaded Depth Anything V2 from torch.hub")
                return
            except Exception as e:
                logger.info(f"Native Depth Anything load skipped ({e}); using manga depth estimator.")

        # Default: robust manga-aware depth estimation
        self._is_native = False
        logger.info(f"DepthEstimatorProvider initialized with manga-aware estimation engine ({self.name}).")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess manga image:
        1. Handle channels (RGBA/Grayscale -> BGR).
        2. Edge-preserving bilateral filter to suppress screentone dots without blurring ink lines.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        if len(image.shape) == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()

        # Bilateral filter suppresses screentone frequency while retaining crisp outlines
        filtered = cv2.bilateralFilter(
            bgr,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma,
            sigmaSpace=self.bilateral_sigma,
        )
        return filtered

    def predict(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run depth inference or manga depth estimation."""
        if self._is_native and self._model is not None:
            try:
                with torch.no_grad():
                    if self._transform is not None:
                        input_batch = self._transform(preprocessed).to(self._device)
                        prediction = self._model(input_batch)
                        prediction = torch.nn.functional.interpolate(
                            prediction.unsqueeze(1),
                            size=preprocessed.shape[:2],
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze()
                        return prediction.cpu().numpy()
                    elif hasattr(self._model, "infer_image"):
                        return self._model.infer_image(preprocessed)
            except Exception as e:
                logger.warning(f"Native model prediction failed: {e}; falling back to manga estimator.")

        return self._estimate_manga_depth(preprocessed)

    def postprocess(self, raw_output: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to [0.0, 1.0] where:
          1.0 = Nearest (foreground / camera)
          0.0 = Furthest (background / sky)
        Followed by Gaussian anti-aliasing.
        """
        depth = raw_output.astype(np.float32)
        min_val = float(depth.min())
        max_val = float(depth.max())

        if max_val - min_val > 1e-6:
            normalized = (depth - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(depth, 0.5)

        if self.smooth_sigma > 0:
            normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=self.smooth_sigma, sigmaY=self.smooth_sigma)
            # Re-normalize slightly
            min_val = float(normalized.min())
            max_val = float(normalized.max())
            if max_val - min_val > 1e-6:
                normalized = (normalized - min_val) / (max_val - min_val)

        return np.clip(normalized, 0.0, 1.0)

    def _estimate_manga_depth(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Synthesizes a realistic relative depth disparity map tailored for manga art:
        1. Vertical perspective gradient (ground plane vs sky/ceiling).
        2. Radial framing gradient (characters centered, gutters distant).
        3. Ink complexity and distance transform (foreground figures possess high ink density and closed contours).
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # 1. Perspective baseline gradient: Y-ramp (lower panel = closer to observer)
        y_indices = np.linspace(0.2, 0.85, h, dtype=np.float32)[:, None]
        perspective_plane = np.tile(y_indices, (1, w))

        # 2. Central focus field: Characters are typically positioned around (0.5w, 0.55h)
        cx, cy = w * 0.5, h * 0.55
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_focus = np.sqrt(((x_grid - cx) / (w * 0.6)) ** 2 + ((y_grid - cy) / (h * 0.6)) ** 2)
        focus_weight = np.clip(1.0 - dist_from_focus * 0.5, 0.1, 1.0).astype(np.float32)

        # 3. Ink structure field: Detect dense ink strokes & local contrast
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32) / 255.0, (45, 45), 15.0)

        # 4. Interior object prominence (distance transform inside non-empty contours)
        # Manga characters usually have closed boundaries with shaded or white interiors
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dt_max = dist_transform.max()
        if dt_max > 0:
            dist_field = dist_transform / dt_max
        else:
            dist_field = np.zeros((h, w), dtype=np.float32)

        # Blend cues
        depth_raw = (
            perspective_plane * 0.40
            + focus_weight * 0.30
            + edge_density * 0.20
            + dist_field * 0.10
        )
        return depth_raw.astype(np.float32)
