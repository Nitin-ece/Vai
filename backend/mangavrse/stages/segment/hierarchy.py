"""
Hierarchy Builder: resolves parent-child nesting trees and assigns depth/zOrder
directly conforming to the scene.json schema in docs/CONTEXT.md.
"""
from __future__ import annotations

from typing import Any
import numpy as np


DEPTH_PRIORS: dict[str, float] = {
    "background": 0.05,
    "environment": 0.20,
    "prop": 0.40,
    "character": 0.65,
    "clothing": 0.68,
    "body": 0.70,
    "head": 0.73,
    "hair": 0.76,
    "eyes": 0.78,
    "effect": 0.86,
    "speech_bubble": 0.94,
    "sfx": 0.98,
}


def build_segment_hierarchy(
    segments: list[dict[str, Any]],
    panel_w: int,
    panel_h: int,
) -> list[dict[str, Any]]:
    """
    Resolves parent-child containment relationships:
    - If Mask A is mostly inside Mask B (IoU intersection / Area(A) >= 0.70)
      and Area(B) > Area(A), Mask A's parentId is set to Mask B's id.
    - Character subparts (head, hair, eyes, body, clothing) are linked to their parent character.
    - Assigns suggested_depth and suggested_z_order.
    """
    n = len(segments)
    if n == 0:
        return []

    # Sort by area descending so parent containers appear before children
    segments_by_area = sorted(segments, key=lambda s: s["area"], reverse=True)

    # Establish parentId relationships
    for i in range(len(segments_by_area)):
        child = segments_by_area[i]
        child_mask = child.get("_mask")
        if child_mask is None:
            continue

        child_area = child["area"]
        best_parent_id = None
        min_parent_area = float("inf")

        # Search preceding larger masks for containment
        for j in range(i):
            parent = segments_by_area[j]
            parent_mask = parent.get("_mask")
            if parent_mask is None:
                continue

            # Check overlap
            intersection = np.logical_and(child_mask > 0, parent_mask > 0).sum()
            overlap_ratio = float(intersection) / max(1.0, float(child_area))

            # Must contain >= 70% of child area
            if overlap_ratio >= 0.70 and parent["area"] < min_parent_area:
                # Don't make speech bubbles or sfx children of character clothing
                if child["label"] in ("speech_bubble", "sfx") and parent["label"] == "character":
                    continue

                min_parent_area = parent["area"]
                best_parent_id = parent["id"]

        child["parentId"] = best_parent_id

        # If child is a part and has a parent, ensure its label is character_part
        if best_parent_id is not None and child.get("part") is not None:
            child["label"] = "character_part"

    # Assign suggested depth and suggested zOrder
    for s in segments_by_area:
        label = s["label"]
        part = s.get("part")
        lookup_key = part if (label == "character_part" and part in DEPTH_PRIORS) else label
        base_depth = DEPTH_PRIORS.get(lookup_key, 0.50)
        
        # Small depth jitter based on layer area to preserve separation
        area_offset = min(0.04, float(s["area"]) / float(panel_w * panel_h) * 0.04)
        s["suggested_depth"] = round(base_depth + area_offset, 3)

    # Sort final segments by suggested_depth ascending
    final_segments = sorted(segments_by_area, key=lambda s: s["suggested_depth"])
    for z_idx, s in enumerate(final_segments):
        s["suggested_z_order"] = z_idx

    return final_segments
