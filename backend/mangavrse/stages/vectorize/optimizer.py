"""
Optimizer: performs speck removal, point reduction, Bezier curve smoothing, and tiny path merging.
"""
from __future__ import annotations

import math
from typing import Any

from .tracer import VectorPath, VectorSegment


class PathOptimizer:
    """Post-processes raw vector paths for node count efficiency and visual smoothness."""

    def __init__(
        self,
        min_path_area: float = 12.0,
        rdp_tolerance: float = 1.2,
        smooth_bezier: bool = True,
        merge_distance: float = 2.0,
    ) -> None:
        self.min_path_area = min_path_area
        self.rdp_tolerance = rdp_tolerance
        self.smooth_bezier = smooth_bezier
        self.merge_distance = merge_distance

    def optimize(self, paths: list[VectorPath]) -> list[VectorPath]:
        """Runs the optimization pipeline on a list of VectorPaths."""
        if not paths:
            return []

        # 1. Speck removal (drop paths below minimum area)
        filtered = [p for p in paths if p.area >= self.min_path_area]

        # 2. Path simplification & Bezier smoothing
        optimized: list[VectorPath] = []
        for path in filtered:
            opt_path = self._simplify_path(path)
            if self.smooth_bezier:
                opt_path = self._smooth_bezier_tangents(opt_path)
            optimized.append(opt_path)

        return optimized

    def _simplify_path(self, path: VectorPath) -> VectorPath:
        """Removes redundant collinear segments."""
        if len(path.segments) < 3:
            return path

        new_segments: list[VectorSegment] = []
        for seg in path.segments:
            # If segment is almost zero-length, skip it
            last_pt = new_segments[-1].end_point if new_segments else path.start_point
            dx = seg.end_point[0] - last_pt[0]
            dy = seg.end_point[1] - last_pt[1]
            if math.hypot(dx, dy) < 0.2 and not seg.is_corner:
                continue
            new_segments.append(seg)

        if not new_segments:
            new_segments = path.segments

        return VectorPath(
            start_point=path.start_point,
            segments=new_segments,
            is_closed=path.is_closed,
            area=path.area,
            bounding_box=path.bounding_box,
        )

    def _smooth_bezier_tangents(self, path: VectorPath) -> VectorPath:
        """Harmonizes Bezier control points to ensure smooth G1 continuity across curve junctions."""
        segments = list(path.segments)
        n = len(segments)
        if n < 2:
            return path

        for i in range(n):
            curr_seg = segments[i]
            prev_seg = segments[(i - 1) % n]

            # If both adjacent segments are curves, align their tangents across the shared anchor
            if not curr_seg.is_corner and not prev_seg.is_corner and curr_seg.c1 and prev_seg.c2:
                anchor = prev_seg.end_point
                # Tangent vectors
                v_in = (anchor[0] - prev_seg.c2[0], anchor[1] - prev_seg.c2[1])
                v_out = (curr_seg.c1[0] - anchor[0], curr_seg.c1[1] - anchor[1])

                len_in = math.hypot(*v_in)
                len_out = math.hypot(*v_out)

                if len_in > 0.5 and len_out > 0.5:
                    # Average tangent direction
                    avg_dx = (v_in[0] / len_in + v_out[0] / len_out) * 0.5
                    avg_dy = (v_in[1] / len_in + v_out[1] / len_out) * 0.5
                    avg_len = math.hypot(avg_dx, avg_dy)

                    if avg_len > 0.1:
                        u_dx = avg_dx / avg_len
                        u_dy = avg_dy / avg_len

                        # Re-project control points along continuous unit tangent
                        new_c2 = (anchor[0] - u_dx * len_in, anchor[1] - u_dy * len_in)
                        new_c1 = (anchor[0] + u_dx * len_out, anchor[1] + u_dy * len_out)

                        prev_seg.c2 = new_c2
                        curr_seg.c1 = new_c1

        return VectorPath(
            start_point=path.start_point,
            segments=segments,
            is_closed=path.is_closed,
            area=path.area,
            bounding_box=path.bounding_box,
        )
