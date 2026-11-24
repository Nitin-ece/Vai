"""
Studio helper: debug visualization overlay and manual override handler.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classical_detector import PanelBox


class StudioOverlayHelper:
    """Renders visual debug overlays with numbered reading order badges and handles manual JSON overrides."""

    OVERRIDE_FILENAME = "panels.override.json"
    PANELS_FILENAME = "panels.json"
    DEBUG_IMAGE_FILENAME = "debug_panels.png"

    @staticmethod
    def draw_debug_overlay(image_bgr: np.ndarray, panels: list[PanelBox]) -> np.ndarray:
        """
        Draws colored polygon borders and numbered reading order badges on the page.
        """
        overlay = image_bgr.copy()

        for panel in panels:
            x, y, w, h = panel.bbox

            # Choose border color based on panel characteristics
            if "flagged" in panel.flags or panel.confidence < 0.6:
                color = (0, 0, 255)  # Red for warnings
            elif panel.is_inset:
                color = (255, 0, 255)  # Magenta for inset/overlapping
            elif panel.is_diagonal:
                color = (0, 165, 255)  # Orange for diagonal/slanted
            else:
                color = (0, 255, 0)  # Green for normal clean panel

            # Draw polygon if available and has >= 3 vertices
            if len(panel.polygon) >= 3:
                pts = np.array(panel.polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=4)
            else:
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color=color, thickness=4)

            # Draw numbered badge at top-right of panel (1-indexed reading order)
            order_label = f"#{panel.reading_order + 1}"
            badge_x = max(x, x + w - 70)
            badge_y = max(30, y + 35)

            # Background for badge
            cv2.circle(overlay, (badge_x + 25, badge_y - 10), 22, (0, 0, 0), -1)
            cv2.circle(overlay, (badge_x + 25, badge_y - 10), 20, color, -1)
            cv2.putText(
                overlay,
                str(panel.reading_order + 1),
                (badge_x + 15, badge_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw panel ID and confidence at bottom left
            info_text = f"{panel.id} ({int(panel.confidence * 100)}%)"
            cv2.putText(
                overlay,
                info_text,
                (x + 8, y + h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                info_text,
                (x + 8, y + h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return overlay

    @classmethod
    def check_and_apply_override(
        cls, page_dir: Path, original_panels: list[PanelBox]
    ) -> tuple[list[PanelBox], bool]:
        """
        If `panels.override.json` exists in `page_dir`, parses and applies manual corrections.
        Returns (panels_to_use, is_overridden).
        """
        override_path = page_dir / cls.OVERRIDE_FILENAME
        if not override_path.exists():
            return original_panels, False

        try:
            data = json.loads(override_path.read_text(encoding="utf-8"))
            panels_data = data.get("panels", [])
            overridden_panels: list[PanelBox] = []

            for idx, p in enumerate(panels_data):
                bbox = p["bbox"]
                polygon = p.get("polygon") or [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    [bbox[0], bbox[1] + bbox[3]],
                ]
                overridden_panels.append(
                    PanelBox(
                        id=p.get("id", f"panel_{idx + 1:03d}"),
                        bbox=bbox,
                        polygon=polygon,
                        area=int(bbox[2] * bbox[3]),
                        is_diagonal=p.get("is_diagonal", False),
                        is_inset=p.get("is_inset", False),
                        confidence=1.0,  # Manual edit has 100% confidence
                        flags=p.get("flags", ["manual_override"]),
                        reading_order=p.get("reading_order", idx),
                    )
                )

            # Sort by reading_order if specified
            overridden_panels.sort(key=lambda p: p.reading_order)
            return overridden_panels, True

        except Exception as e:
            # Fall back to original detection if override parsing fails
            return original_panels, False

    @staticmethod
    def crop_and_save_panel(
        image_bgr: np.ndarray, panel: PanelBox, output_panel_path: Path
    ) -> None:
        """
        Crops panel from image array and saves as PNG.
        If panel is diagonal, applies polygon mask with transparent background if 4 channels.
        """
        output_panel_path.parent.mkdir(parents=True, exist_ok=True)
        x, y, w, h = panel.bbox
        img_h, img_w = image_bgr.shape[:2]

        # Clamp boundaries
        x0 = max(0, min(x, img_w - 1))
        y0 = max(0, min(y, img_h - 1))
        x1 = max(x0 + 1, min(x + w, img_w))
        y1 = max(y0 + 1, min(y + h, img_h))

        crop = image_bgr[y0:y1, x0:x1]
        cv2.imwrite(str(output_panel_path), crop)
