"""
Generates 5 distinct manga benchmark panels for the Enhance stage evaluation:
  1. Dense screentone (varying halftone frequencies)
  2. Clean line art (delicate curves and cross-hatching)
  3. Speed lines (radiating high-frequency action strokes)
  4. Text-heavy (dialogue speech bubbles with small glyphs)
  5. Low-res scan (downsampled, noisy, JPEG compression artifacts)
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def generate_benchmark_panels(base_dir: Path) -> list[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    panel_names = ["dense_screentone", "clean_lineart", "speed_lines", "text_heavy", "lowres_scan"]

    w, h = 320, 320

    # -------------------------------------------------------------
    # 1. Dense Screentone
    # -------------------------------------------------------------
    p1 = np.full((h, w, 3), 242, dtype=np.uint8)
    cv2.rectangle(p1, (8, 8), (w - 8, h - 8), (15, 15, 15), 4)
    # Character silhouette
    cv2.ellipse(p1, (w // 2, h // 2), (90, 110), 0, 0, 360, (20, 20, 20), -1)
    # Halftone screentone background (grid of 2x2 dots spaced by 5px)
    for y in range(16, h - 16, 5):
        for x in range(16, w - 16, 5):
            if (x - w // 2) ** 2 / 95**2 + (y - h // 2) ** 2 / 115**2 > 1.0:
                p1[y : y + 2, x : x + 2] = (50, 50, 50)
    p1_dir = base_dir / "dense_screentone"
    p1_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p1_dir / "original.png"), p1)

    # -------------------------------------------------------------
    # 2. Clean Line Art
    # -------------------------------------------------------------
    p2 = np.full((h, w, 3), 248, dtype=np.uint8)
    cv2.rectangle(p2, (8, 8), (w - 8, h - 8), (15, 15, 15), 4)
    # Delicate concentric bezier curves and hatching
    for r in range(25, 130, 15):
        cv2.circle(p2, (w // 2, h // 2), r, (25, 25, 25), 2, cv2.LINE_AA)
    # Fine 45-degree cross-hatching
    for d in range(40, 260, 10):
        cv2.line(p2, (d, 80), (d + 60, 220), (35, 35, 35), 1, cv2.LINE_AA)
    p2_dir = base_dir / "clean_lineart"
    p2_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p2_dir / "original.png"), p2)

    # -------------------------------------------------------------
    # 3. Speed Lines
    # -------------------------------------------------------------
    p3 = np.full((h, w, 3), 250, dtype=np.uint8)
    cv2.rectangle(p3, (8, 8), (w - 8, h - 8), (15, 15, 15), 4)
    center_x, center_y = w // 2, h // 2
    # Radiating sharp action strokes
    for angle in np.linspace(0, 2 * math.pi, 80, endpoint=False):
        # Varying lengths
        r_start = np.random.uniform(50, 90)
        r_end = np.random.uniform(140, 155)
        x0 = int(center_x + r_start * math.cos(angle))
        y0 = int(center_y + r_start * math.sin(angle))
        x1 = int(center_x + r_end * math.cos(angle))
        y1 = int(center_y + r_end * math.sin(angle))
        cv2.line(p3, (x0, y0), (x1, y1), (20, 20, 20), 2, cv2.LINE_AA)
    p3_dir = base_dir / "speed_lines"
    p3_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p3_dir / "original.png"), p3)

    # -------------------------------------------------------------
    # 4. Text-Heavy
    # -------------------------------------------------------------
    p4 = np.full((h, w, 3), 246, dtype=np.uint8)
    cv2.rectangle(p4, (8, 8), (w - 8, h - 8), (15, 15, 15), 4)
    # Speech bubble ellipse
    cv2.ellipse(p4, (w // 2, h // 2), (130, 120), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(p4, (w // 2, h // 2), (130, 120), 0, 0, 360, (20, 20, 20), 3, cv2.LINE_AA)
    # Small dialogue text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(p4, "N-NANI?!", (90, 110), font, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(p4, "THIS POWER...", (75, 150), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(p4, "IT'S IMPOSSIBLE!", (60, 190), font, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(p4, "(CHAPTER 42)", (95, 230), font, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    p4_dir = base_dir / "text_heavy"
    p4_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p4_dir / "original.png"), p4)

    # -------------------------------------------------------------
    # 5. Low-Res Scan (Downsampled, noisy, JPEG artifacts)
    # -------------------------------------------------------------
    p5_hi = np.full((h, w, 3), 235, dtype=np.uint8)
    cv2.rectangle(p5_hi, (8, 8), (w - 8, h - 8), (20, 20, 20), 4)
    cv2.circle(p5_hi, (100, 100), 50, (30, 30, 30), 3)
    cv2.line(p5_hi, (50, 250), (270, 250), (30, 30, 30), 4)
    cv2.putText(p5_hi, "SCAN", (130, 180), font, 1.2, (30, 30, 30), 3)

    # Downsample to 80x80 then upscale back with blur & jpeg compression
    p5_small = cv2.resize(p5_hi, (80, 80), interpolation=cv2.INTER_AREA)
    _, p5_jpg = cv2.imencode(".jpg", p5_small, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    p5_noisy = cv2.imdecode(p5_jpg, cv2.IMREAD_COLOR)
    p5 = cv2.resize(p5_noisy, (w, h), interpolation=cv2.INTER_NEAREST)

    p5_dir = base_dir / "lowres_scan"
    p5_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p5_dir / "original.png"), p5)

    print(f"Generated 5 benchmark panels in {base_dir}")
    return panel_names


if __name__ == "__main__":
    generate_benchmark_panels(Path("data/benchmark_panels"))
