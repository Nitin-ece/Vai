"""
Reading order determination for manga panels.

Defaults to traditional Japanese Manga reading order:
Right-to-Left, Top-to-Bottom. Also supports Western (Left-to-Right).
"""
from __future__ import annotations

from typing import Literal

from .classical_detector import PanelBox


def sort_reading_order(
    panels: list[PanelBox],
    direction: Literal["rtl", "ltr"] = "rtl",
    vertical_overlap_threshold: float = 0.40,
) -> list[PanelBox]:
    """
    Sorts *panels* into visual reading order.
    Assigns sequential 0-indexed `reading_order` and formatted `id` ("panel_001", etc.).
    """
    if not panels:
        return []

    # Make a copy
    candidates = list(panels)

    # Helper: vertical overlap ratio between two panels
    def vertical_overlap(p1: PanelBox, p2: PanelBox) -> float:
        y1_top, y1_bot = p1.bbox[1], p1.bbox[1] + p1.bbox[3]
        y2_top, y2_bot = p2.bbox[1], p2.bbox[1] + p2.bbox[3]
        intersection = max(0, min(y1_bot, y2_bot) - max(y1_top, y2_top))
        min_h = min(p1.bbox[3], p2.bbox[3])
        return intersection / float(min_h) if min_h > 0 else 0.0

    # Cluster panels into horizontal tiers
    # Sort first by y coordinate to seed clusters
    candidates.sort(key=lambda p: (p.bbox[1], -p.bbox[0] if direction == "rtl" else p.bbox[0]))

    tiers: list[list[PanelBox]] = []
    for panel in candidates:
        placed = False
        for tier in tiers:
            # Check if panel overlaps vertically with any member of this tier
            if any(vertical_overlap(panel, member) > vertical_overlap_threshold for member in tier):
                tier.append(panel)
                placed = True
                break
        if not placed:
            tiers.append([panel])

    # Sort each tier internally
    # For RTL: rightmost panel first (x center descending)
    # For LTR: leftmost panel first (x center ascending)
    reverse_x = (direction == "rtl")
    sorted_panels: list[PanelBox] = []

    # Sort tiers by their minimum y coordinate
    tiers.sort(key=lambda tier: min(p.bbox[1] for p in tier))

    for tier in tiers:
        tier.sort(key=lambda p: p.bbox[0] + (p.bbox[2] / 2.0), reverse=reverse_x)
        sorted_panels.extend(tier)

    # Assign reading order index and deterministic panel id
    for idx, panel in enumerate(sorted_panels):
        panel.reading_order = idx
        panel.id = f"panel_{idx + 1:03d}"

    return sorted_panels
