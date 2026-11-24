"""
Classical manga panel detector.

Techniques:
1. Gutter / Whitespace Analysis via binary projection profiles and morphological kernels.
2. Contour Tree Hierarchy (cv2.RETR_TREE) to capture nested, inset, and overlapping panels.
3. Polygonal approximation (cv2.approxPolyDP) to handle diagonal/slanted panels.
4. Bounding box & polygon extraction with area and aspect ratio filtering.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class PanelBox:
    id: str
    bbox: list[int]  # [x, y, w, h]
    polygon: list[list[int]]  # [[x1, y1], [x2, y2], ...]
    area: int
    is_diagonal: bool = False
    is_inset: bool = False
    confidence: float = 0.9
    flags: list[str] = field(default_factory=list)
    reading_order: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClassicalPanelDetector:
    """
    Detects manga panels using gutter analysis, edge contours,
    and polygon fitting without requiring an ML model.
    """

    def __init__(
        self,
        min_panel_area_ratio: float = 0.015,
        max_panel_area_ratio: float = 0.95,
        gutter_threshold_ratio: float = 0.96,
        poly_epsilon_ratio: float = 0.015,
    ) -> None:
        self.min_panel_area_ratio = min_panel_area_ratio
        self.max_panel_area_ratio = max_panel_area_ratio
        self.gutter_threshold_ratio = gutter_threshold_ratio
        self.poly_epsilon_ratio = poly_epsilon_ratio

    def detect_panels(self, image: np.ndarray) -> list[PanelBox]:
        """
        Detect panels in a BGR or grayscale image array.
        Returns a list of PanelBox instances (unordered).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        page_area = h * w
        min_area = int(page_area * self.min_panel_area_ratio)
        max_area = int(page_area * self.max_panel_area_ratio)

        # 1. Preprocess: remove screentone / halftone noise with median blur + bilateral
        blurred = cv2.medianBlur(gray, 3)

        # 2. Binary thresholding: invert so panel borders and art are foreground (white)
        # Manga gutters are usually white (high values in gray)
        _, binary = cv2.threshold(blurred, 230, 255, cv2.THRESH_BINARY_INV)

        # 3. Find connected contours with hierarchy
        # Using RETR_TREE allows detection of panels within page border and inset panels
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        raw_panels: list[PanelBox] = []
        if hierarchy is None or len(contours) == 0:
            return self._fallback_gutter_cut(gray)

        hierarchy = hierarchy[0]  # shape: (N, 4) -> [Next, Previous, First_Child, Parent]

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Check if this contour resembles a panel
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = bw / float(bh) if bh > 0 else 0

            # Discard extreme lines or banners that are too thin to be panels
            if aspect_ratio > 8.0 or aspect_ratio < 0.12:
                continue

            # Approximate polygon
            perimeter = cv2.arcLength(cnt, True)
            poly = cv2.approxPolyDP(cnt, self.poly_epsilon_ratio * perimeter, True)

            # Determine if panel is diagonal / non-orthogonal
            is_diagonal = self._is_polygon_diagonal(poly)

            # Check if panel is inset (has a valid parent that is also a candidate panel)
            parent_idx = hierarchy[idx][3]
            is_inset = False
            if parent_idx >= 0:
                parent_cnt = contours[parent_idx]
                parent_area = cv2.contourArea(parent_cnt)
                if min_area <= parent_area <= page_area * 0.98:
                    is_inset = True

            flags: list[str] = []
            if is_diagonal:
                flags.append("diagonal_panel")
            if is_inset:
                flags.append("inset_panel")

            poly_coords = [[int(pt[0][0]), int(pt[0][1])] for pt in poly]

            raw_panels.append(
                PanelBox(
                    id="",  # will be assigned during ordering
                    bbox=[int(x), int(y), int(bw), int(bh)],
                    polygon=poly_coords if len(poly_coords) >= 3 else [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                    area=int(area),
                    is_diagonal=is_diagonal,
                    is_inset=is_inset,
                    confidence=0.92 if not flags else 0.85,
                    flags=flags,
                )
            )

        # 4. Filter duplicates and overlapping non-inset containers
        filtered = self._filter_redundant_panels(raw_panels, page_area)

        # 5. If no panels found via contours (e.g. borderless panels / open gutters), use gutter projection fallback
        if len(filtered) == 0:
            filtered = self._fallback_gutter_cut(gray)

        # 6. Fallback to entire image as 1 panel if still no panels detected
        if len(filtered) == 0:
            filtered = [
                PanelBox(
                    id="panel_001",
                    bbox=[0, 0, w, h],
                    polygon=[[0, 0], [w, 0], [w, h], [0, h]],
                    area=w * h,
                    is_diagonal=False,
                    confidence=1.0,
                    flags=["full_page_fallback"],
                )
            ]

        return filtered

    def _is_polygon_diagonal(self, poly: np.ndarray) -> bool:
        """Check if polygon edges deviate significantly from horizontal and vertical axes."""
        if len(poly) < 3:
            return False

        pts = [pt[0] for pt in poly]
        n = len(pts)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            length = math.hypot(dx, dy)
            if length < 20:
                continue
            # Angle with horizontal
            angle_deg = math.degrees(math.atan2(dy, dx))
            # If not close to 0/180 or 90/270, it's slanted
            if 5.0 < angle_deg < 85.0:
                return True
        return False

    def _filter_redundant_panels(self, panels: list[PanelBox], page_area: int) -> list[PanelBox]:
        """
        Suppresses redundant outer bounding boxes and keeps most specific panels.
        """
        if not panels:
            return []

        # Sort by area descending
        panels.sort(key=lambda p: p.area, reverse=True)
        keep: list[PanelBox] = []

        for p in panels:
            x1, y1, w1, h1 = p.bbox
            area1 = w1 * h1
            suppress = False

            for kept in keep:
                x2, y2, w2, h2 = kept.bbox
                # Compute intersection
                ix = max(x1, x2)
                iy = max(y1, y2)
                iw = max(0, min(x1 + w1, x2 + w2) - ix)
                ih = max(0, min(y1 + h1, y2 + h2) - iy)
                intersection = iw * ih

                # If almost identical to an already kept panel (IoU > 0.8)
                union = area1 + (w2 * h2) - intersection
                iou = intersection / union if union > 0 else 0
                if iou > 0.80:
                    suppress = True
                    break

            if not suppress:
                keep.append(p)

        return keep

    def _fallback_gutter_cut(self, gray: np.ndarray) -> list[PanelBox]:
        """
        Fallback for borderless pages: scans projection profiles to find gutters.
        """
        h, w = gray.shape
        # Page borders margin (ignore outermost 2% margins)
        margin_x = int(w * 0.03)
        margin_y = int(h * 0.03)

        # Threshold to find white gutters (values > 240)
        white_mask = (gray > 240).astype(np.uint8)

        # Horizontal projection (average white pixel ratio across rows)
        h_profile = np.mean(white_mask[:, margin_x : w - margin_x], axis=1)
        gutters_y = np.where(h_profile > self.gutter_threshold_ratio)[0]

        # Find continuous gutter intervals
        split_y = [0]
        if len(gutters_y) > 0:
            diffs = np.diff(gutters_y)
            gap_indices = np.where(diffs > 1)[0]
            start_idx = 0
            for gi in gap_indices:
                gutter_center = int(np.mean(gutters_y[start_idx : gi + 1]))
                if margin_y < gutter_center < h - margin_y:
                    split_y.append(gutter_center)
                start_idx = gi + 1
            if start_idx < len(gutters_y):
                gutter_center = int(np.mean(gutters_y[start_idx:]))
                if margin_y < gutter_center < h - margin_y:
                    split_y.append(gutter_center)
        split_y.append(h)
        split_y = sorted(list(set(split_y)))

        panels: list[PanelBox] = []
        for i in range(len(split_y) - 1):
            y_start = split_y[i]
            y_end = split_y[i + 1]
            panel_h = y_end - y_start
            if panel_h < h * self.min_panel_area_ratio:
                continue

            # Within each horizontal slice, check vertical gutters
            slice_mask = white_mask[y_start:y_end, :]
            v_profile = np.mean(slice_mask, axis=0)
            gutters_x = np.where(v_profile > self.gutter_threshold_ratio)[0]

            split_x = [0]
            if len(gutters_x) > 0:
                diffs_x = np.diff(gutters_x)
                gap_x = np.where(diffs_x > 1)[0]
                sx = 0
                for gx in gap_x:
                    cx = int(np.mean(gutters_x[sx : gx + 1]))
                    if margin_x < cx < w - margin_x:
                        split_x.append(cx)
                    sx = gx + 1
                if sx < len(gutters_x):
                    cx = int(np.mean(gutters_x[sx:]))
                    if margin_x < cx < w - margin_x:
                        split_x.append(cx)
            split_x.append(w)
            split_x = sorted(list(set(split_x)))

            for j in range(len(split_x) - 1):
                x_start = split_x[j]
                x_end = split_x[j + 1]
                panel_w = x_end - x_start
                if panel_w < w * 0.1:
                    continue

                bbox = [x_start, y_start, panel_w, panel_h]
                poly = [
                    [x_start, y_start],
                    [x_end, y_start],
                    [x_end, y_end],
                    [x_start, y_end],
                ]
                panels.append(
                    PanelBox(
                        id="",
                        bbox=bbox,
                        polygon=poly,
                        area=panel_w * panel_h,
                        is_diagonal=False,
                        confidence=0.80,
                        flags=["gutter_projection_detected"],
                    )
                )

        return panels
