"""
Fidelity Evaluator: renders vector paths back to raster and computes
objective fidelity scores against the source binary line art.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .tracer import VectorPath


def rasterize_vector_paths(
    paths: list[VectorPath],
    width: int,
    height: int,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Renders vector paths to a binary uint8 raster canvas (255 for ink, 0 for white).
    Uses SVG standard even-odd winding rule (XOR) to correctly handle nested holes and rings.
    """
    ch = int(height * scale)
    cw = int(width * scale)
    canvas = np.zeros((ch, cw), dtype=np.uint8)

    for path in paths:
        pts: list[list[int]] = []
        sx, sy = path.start_point
        pts.append([int(round(sx * scale)), int(round(sy * scale))])

        for seg in path.segments:
            ep = seg.end_point
            if seg.is_corner and seg.c:
                pts.append([int(round(seg.c[0] * scale)), int(round(seg.c[1] * scale))])
                pts.append([int(round(ep[0] * scale)), int(round(ep[1] * scale))])
            elif not seg.is_corner and seg.c1 and seg.c2:
                # Sample intermediate points along cubic Bezier curve (16 segments for high precision)
                c1 = seg.c1
                c2 = seg.c2
                p0 = (sx, sy)
                p3 = ep
                for t in np.linspace(0.0625, 1.0, 16):
                    t2 = t * t
                    t3 = t2 * t
                    mt = 1.0 - t
                    mt2 = mt * mt
                    mt3 = mt2 * mt
                    bx = mt3 * p0[0] + 3 * mt2 * t * c1[0] + 3 * mt * t2 * c2[0] + t3 * p3[0]
                    by = mt3 * p0[1] + 3 * mt2 * t * c1[1] + 3 * mt * t2 * c2[1] + t3 * p3[1]
                    pts.append([int(round(bx * scale)), int(round(by * scale))])
            else:
                pts.append([int(round(ep[0] * scale)), int(round(ep[1] * scale))])
            sx, sy = ep

        if len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            poly_layer = np.zeros((ch, cw), dtype=np.uint8)
            cv2.fillPoly(poly_layer, [poly], 255)
            # Even-odd rule: overlapping path boundaries invert fill
            canvas = cv2.bitwise_xor(canvas, poly_layer)

    return canvas


def compute_fidelity_score(
    source_mask: np.ndarray,
    paths: list[VectorPath],
    width: int,
    height: int,
) -> dict[str, float]:
    """
    Computes objective overlap fidelity metrics:
    - IoU (Intersection over Union)
    - Precision, Recall, F1-Score
    """
    vector_raster = rasterize_vector_paths(paths, width, height, scale=1.0)
    src_bool = source_mask > 127
    vec_bool = vector_raster > 127

    intersection = np.count_nonzero(src_bool & vec_bool)
    union = np.count_nonzero(src_bool | vec_bool)

    iou = float(intersection) / float(union) if union > 0 else 1.0

    src_count = np.count_nonzero(src_bool)
    vec_count = np.count_nonzero(vec_bool)

    precision = float(intersection) / float(vec_count) if vec_count > 0 else 1.0
    recall = float(intersection) / float(src_count) if src_count > 0 else 1.0
    f1 = (2.0 * precision * recall) / (precision + recall + 1e-6)

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "fidelity_score": round(0.4 * iou + 0.6 * f1, 4),
    }
