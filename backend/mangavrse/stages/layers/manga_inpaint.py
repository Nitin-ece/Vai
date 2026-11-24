"""
Manga-specific Inpainting Engine:
- Directional structural line extension across hole boundaries.
- Halftone screentone texture synthesis to prevent flat gray blur patches.
- Fast marching / Navier-Stokes gradient continuity.
"""
from __future__ import annotations

import cv2
import numpy as np


def extend_structural_lines(
    image_bgr: np.ndarray,
    occlusion_mask: np.ndarray,
    line_thickness: int = 2,
    search_band: int = 12,
) -> np.ndarray:
    """
    Finds structural ink lines intersecting the occlusion hole perimeter and
    projects their trajectories across the hole to prevent broken background geometry.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    hole_bool = occlusion_mask > 127

    if not np.any(hole_bool):
        return image_bgr.copy()

    # Border perimeter of the hole
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (search_band, search_band))
    dilated_hole = cv2.dilate((hole_bool).astype(np.uint8) * 255, kernel)
    border_band = cv2.bitwise_and(dilated_hole, cv2.bitwise_not((hole_bool).astype(np.uint8) * 255))

    # Ink edges in the surrounding border
    _, ink_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    border_lines = cv2.bitwise_and(ink_binary, border_band)

    # Detect line segments using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(border_lines, 1, np.pi / 180, threshold=15, minLineLength=10, maxLineGap=4)
    canvas = image_bgr.copy()

    if lines is not None:
        for line in lines:
            coords = line.flatten()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length < 5:
                continue

            # Unit direction vector
            ux = dx / length
            uy = dy / length

            # Project line forward into the hole (up to 40 pixels)
            proj_x1 = int(round(x1 - ux * 25))
            proj_y1 = int(round(y1 - uy * 25))
            proj_x2 = int(round(x2 + ux * 25))
            proj_y2 = int(round(y2 + uy * 25))

            # Draw extended line only inside the hole
            line_canvas = np.zeros((h, w), dtype=np.uint8)
            cv2.line(line_canvas, (proj_x1, proj_y1), (proj_x2, proj_y2), 255, thickness=line_thickness)
            line_in_hole = cv2.bitwise_and(line_canvas, (hole_bool).astype(np.uint8) * 255)

            canvas[line_in_hole > 0] = (20, 20, 20)

    return canvas


def synthesize_screentone_fill(
    image_bgr: np.ndarray,
    occlusion_mask: np.ndarray,
    sample_radius: int = 15,
) -> tuple[np.ndarray, bool]:
    """
    Checks if the hole borders a screentone field.
    If so, synthesizes a continuous halftone dot pattern across the hole.
    Returns: (synthesized_image, is_screentone_active).
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    hole_bool = occlusion_mask > 127

    # Measure surrounding perimeter tone frequency
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sample_radius * 2 + 1, sample_radius * 2 + 1))
    dilated_hole = cv2.dilate((hole_bool).astype(np.uint8) * 255, kernel)
    perimeter_band = cv2.bitwise_and(dilated_hole, cv2.bitwise_not((hole_bool).astype(np.uint8) * 255))

    # Dot count in perimeter
    _, dark_dots = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    dots_in_perimeter = cv2.bitwise_and(dark_dots, perimeter_band)

    perimeter_pixels = np.count_nonzero(perimeter_band)
    if perimeter_pixels == 0:
        return image_bgr.copy(), False

    dot_ratio = float(np.count_nonzero(dots_in_perimeter)) / float(perimeter_pixels)

    # If > 8% of surrounding area is halftone dots, synthesize screentone
    if dot_ratio > 0.08:
        # Determine average dot spacing (typically 4 - 8 px in manga scans)
        spacing = 6
        out = image_bgr.copy()
        
        # Base fill with mean perimeter tone
        mean_tone = int(np.mean(gray[perimeter_band > 0]))
        out[hole_bool] = (mean_tone, mean_tone, mean_tone)

        # Draw lattice dots inside hole
        y_idxs, x_idxs = np.where(hole_bool)
        if len(y_idxs) > 0:
            min_y, max_y = np.min(y_idxs), np.max(y_idxs)
            min_x, max_x = np.min(x_idxs), np.max(x_idxs)

            dot_r = 1 if dot_ratio < 0.20 else 2
            for y in range(min_y, max_y + 1, spacing):
                for x in range(min_x, max_x + 1, spacing):
                    if hole_bool[y, x]:
                        cv2.circle(out, (x, y), dot_r, (40, 40, 40), -1)

        return out, True

    return image_bgr.copy(), False


def inpaint_manga_background(
    image_bgr: np.ndarray,
    occlusion_mask: np.ndarray,
    inpaint_radius: int = 5,
    method: str = "telea_texture",
    line_extension: bool = True,
    screentone_synthesis: bool = True,
) -> np.ndarray:
    """
    Executes manga-aware background inpainting:
    1. Pre-inpaint line extension.
    2. Screentone synthesis or Fast-Marching Navier-Stokes inpaint.
    3. Edge-preserving blending.
    """
    h, w = image_bgr.shape[:2]
    hole_bool = occlusion_mask > 127

    if not np.any(hole_bool):
        return image_bgr.copy()

    # Step 1: Extend structural lines
    guided_image = extend_structural_lines(image_bgr, occlusion_mask) if line_extension else image_bgr.copy()

    # Step 2: Screentone pattern fill if tone field detected
    if screentone_synthesis:
        tone_filled, has_tone = synthesize_screentone_fill(guided_image, occlusion_mask)
        if has_tone:
            # Subtle edge blend along hole boundary
            hole_boundary = cv2.Canny((hole_bool).astype(np.uint8) * 255, 50, 150)
            dilated_boundary = cv2.dilate(hole_boundary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            blended = cv2.inpaint(tone_filled, dilated_boundary, 2, cv2.INPAINT_TELEA)
            return blended

    # Step 3: Fast-Marching Telea / Navier-Stokes deep inpaint
    cv2_flag = cv2.INPAINT_NS if "ns" in method.lower() else cv2.INPAINT_TELEA
    inpainted = cv2.inpaint(guided_image, (hole_bool).astype(np.uint8) * 255, inpaint_radius, cv2_flag)

    return inpainted
