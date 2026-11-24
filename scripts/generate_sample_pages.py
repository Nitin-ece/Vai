"""
Generates 3 representative sample manga pages in data/samples/ for validation:
  1. Standard 4-panel grid with horizontal and vertical gutters
  2. Dynamic action page with diagonal/slanted panels and inset reaction panel
  3. Borderless page with open gutters and speech bubble elements
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def generate_samples(samples_dir: Path) -> None:
    samples_dir.mkdir(parents=True, exist_ok=True)
    w, h = 1200, 1800

    # Page 1: Standard 4-panel grid (RTL: Top-Right -> Top-Left -> Bottom-Right -> Bottom-Left)
    p1 = np.full((h, w, 3), 255, dtype=np.uint8)
    mx, my, gx, gy = 80, 80, 40, 60
    pw = (w - 2 * mx - gx) // 2
    ph = (h - 2 * my - gy) // 2

    # Panel 1 (Top-Right)
    cv2.rectangle(p1, (mx + pw + gx, my), (mx + 2 * pw + gx, my + ph), (0, 0, 0), 4)
    cv2.circle(p1, (mx + pw + gx + 200, my + 200), 80, (60, 60, 60), -1)

    # Panel 2 (Top-Left)
    cv2.rectangle(p1, (mx, my), (mx + pw, my + ph), (0, 0, 0), 4)
    cv2.line(p1, (mx + 50, my + 50), (mx + pw - 50, my + ph - 50), (30, 30, 30), 4)

    # Panel 3 (Bottom-Right)
    cv2.rectangle(p1, (mx + pw + gx, my + ph + gy), (mx + 2 * pw + gx, my + 2 * ph + gy), (0, 0, 0), 4)
    cv2.rectangle(p1, (mx + pw + gx + 100, my + ph + gy + 100), (mx + pw + gx + 350, my + ph + gy + 400), (100, 100, 100), -1)

    # Panel 4 (Bottom-Left)
    cv2.rectangle(p1, (mx, my + ph + gy), (mx + pw, my + 2 * ph + gy), (0, 0, 0), 4)
    cv2.ellipse(p1, (mx + pw // 2, my + ph + gy + ph // 2), (180, 120), 0, 0, 360, (70, 70, 70), -1)

    cv2.imwrite(str(samples_dir / "sample_page_001.png"), p1)

    # Page 2: Dynamic Action layout with diagonal cut and inset panel
    p2 = np.full((h, w, 3), 255, dtype=np.uint8)
    # Top banner panel
    cv2.rectangle(p2, (80, 80), (w - 80, 450), (0, 0, 0), 4)
    cv2.putText(p2, "TOP WIDE PANEL", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3)

    # Middle diagonal panel (slanted frame cutting across)
    diag_pts = np.array([[80, 520], [w - 80, 620], [w - 80, 1150], [80, 1020]], np.int32)
    cv2.fillPoly(p2, [diag_pts], (240, 240, 240))
    cv2.polylines(p2, [diag_pts], True, (0, 0, 0), 4)

    # Inset panel inside the diagonal panel
    cv2.rectangle(p2, (w - 450, 700), (w - 150, 950), (0, 0, 0), 4)
    cv2.putText(p2, "INSET", (w - 380, 840), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)

    # Bottom split panels (2 columns)
    cv2.rectangle(p2, (w // 2 + 20, 1220), (w - 80, h - 80), (0, 0, 0), 4)
    cv2.rectangle(p2, (80, 1220), (w // 2 - 20, h - 80), (0, 0, 0), 4)

    cv2.imwrite(str(samples_dir / "sample_page_002.png"), p2)

    # Page 3: 3-tier vertical page with borderless middle panel (open gutters)
    p3 = np.full((h, w, 3), 255, dtype=np.uint8)
    # Top panel
    cv2.rectangle(p3, (80, 80), (w - 80, 500), (0, 0, 0), 4)

    # Middle panel: borderless, distinguished by content surrounded by white gutters
    cv2.rectangle(p3, (120, 600), (w - 120, 1150), (220, 220, 220), -1)
    cv2.circle(p3, (w // 2, 875), 150, (60, 60, 60), -1)

    # Bottom panel: standard bordered
    cv2.rectangle(p3, (80, 1250), (w - 80, h - 80), (0, 0, 0), 4)

    cv2.imwrite(str(samples_dir / "sample_page_003.png"), p3)
    print(f"Generated 3 sample pages in {samples_dir}")


if __name__ == "__main__":
    generate_samples(Path("data/samples"))
