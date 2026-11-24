"""
Page classifier: detects and flags non-story pages (covers, color pages, ads, blank pages).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PageClassification:
    is_story_page: bool
    page_type: str  # "story", "color_page", "cover", "ad", "blank"
    confidence: float
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class PageClassifier:
    """Identifies covers, color pages, blanks, and ads based on visual features."""

    def __init__(
        self,
        color_saturation_threshold: float = 0.18,
        blank_variance_threshold: float = 50.0,
    ) -> None:
        self.color_saturation_threshold = color_saturation_threshold
        self.blank_variance_threshold = blank_variance_threshold

    def classify(self, image_path: Path) -> PageClassification:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return PageClassification(
                is_story_page=False,
                page_type="corrupt",
                confidence=1.0,
                flags=["cannot_decode_image"],
            )

        h, w = img_bgr.shape[:2]
        flags: list[str] = []

        # 1. Color saturation detection (manga story pages are monochrome / grayscale)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        var_val = float(np.var(gray))
        mean_sat = float(np.mean(saturation))

        mask = (gray > 25) & (gray < 235)
        if np.count_nonzero(mask) > 1000:
            relevant_sat = saturation[mask]
            sat_ratio = float(np.count_nonzero(relevant_sat > 0.35) / len(relevant_sat))
        else:
            sat_ratio = float(np.count_nonzero(saturation > 0.35) / (h * w))

        is_color = mean_sat > self.color_saturation_threshold or sat_ratio > 0.15
        if is_color:
            flags.append("high_color_saturation")
            return PageClassification(
                is_story_page=False,
                page_type="color_page",
                confidence=0.9,
                flags=flags,
                metrics={
                    "mean_saturation": round(mean_sat, 4),
                    "high_sat_ratio": round(sat_ratio, 4),
                    "variance": round(var_val, 2),
                },
            )

        # 2. Blank detection (only for monochrome pages)
        if var_val < self.blank_variance_threshold:
            flags.append("blank_page")
            return PageClassification(
                is_story_page=False,
                page_type="blank",
                confidence=0.95,
                flags=flags,
                metrics={"mean_val": mean_val, "variance": var_val},
            )

        # 3. Regular story page
        return PageClassification(
            is_story_page=True,
            page_type="story",
            confidence=0.95,
            flags=flags,
            metrics={
                "mean_saturation": round(mean_sat, 4),
                "variance": round(var_val, 2),
            },
        )
