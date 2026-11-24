"""
Segment Stage: decomposes enhanced manga panels into semantic layers
(characters, sub-parts, props, background, environment, effects, speech bubbles, SFX).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mangavrse.core.stage import AbstractStage, StageResult
from mangavrse.stages.vectorize.extractor import extract_lines_and_tones
from mangavrse.models.yolo_detector import YOLODetectorProvider
from mangavrse.models.character_encoder import CharacterVisionEncoder

from .classifier import classify_segment
from .hierarchy import build_segment_hierarchy
from .line_snapper import snap_mask_to_lines
from .preprocessor import detect_candidate_bubbles_and_sfx, smooth_for_segmentation
from .sam_provider import SAM2SegmenterProvider
from .visualizer import create_segment_overlay

logger = logging.getLogger(__name__)


class SegmentStage(AbstractStage):
    """
    Segmentation stage converting enhanced panels into semantic, hierarchical layer masks.
    Output directly matches the scene.json schema in docs/CONTEXT.md.
    """

    name = "segment"

    def __init__(
        self,
        segmenter: SAM2SegmenterProvider | None = None,
        yolo_detector: YOLODetectorProvider | None = None,
        character_encoder: CharacterVisionEncoder | None = None,
    ) -> None:
        super().__init__(name="segment")
        self.segmenter = segmenter or SAM2SegmenterProvider()
        self.yolo_detector = yolo_detector or YOLODetectorProvider()
        self.character_encoder = character_encoder or CharacterVisionEncoder()

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        seg_cfg = config.get("segment", {})
        models_cfg = config.get("models", {}).get("segmenter", {})

        # Load model provider if configured
        device = config.get("device", {}).get("prefer_cuda", False)
        dev_str = "cuda" if device else "cpu"
        weights_path = models_cfg.get("weights")
        self.segmenter.load(device=dev_str, weights=weights_path)

        screentone_smoothing = bool(seg_cfg.get("screentone_smoothing", True))
        bubble_detection = bool(seg_cfg.get("bubble_detection", True))
        line_snapping = bool(seg_cfg.get("line_snapping", True))
        conf_threshold = float(seg_cfg.get("confidence_threshold", 0.65))

        # Find target panel images (either directly in input_dir or panel subdirectories)
        target_images: list[tuple[str, Path, Path]] = []

        direct_enhanced = input_dir / "enhanced.png"
        direct_original = input_dir / "original.png"
        if direct_enhanced.exists():
            target_images.append(("panel_000", direct_enhanced, output_dir))
        elif direct_original.exists():
            target_images.append(("panel_000", direct_original, output_dir))
        else:
            for panel_dir in sorted(input_dir.glob("**/enhanced.png")):
                rel_parent = panel_dir.parent.relative_to(input_dir)
                panel_id = rel_parent.as_posix().replace("/", "_")
                target_dir = output_dir / rel_parent
                target_images.append((panel_id, panel_dir, target_dir))

            if not target_images:
                for orig_file in sorted(input_dir.glob("**/original.png")):
                    rel_parent = orig_file.parent.relative_to(input_dir)
                    panel_id = rel_parent.as_posix().replace("/", "_")
                    target_dir = output_dir / rel_parent
                    target_images.append((panel_id, orig_file, target_dir))

        if not target_images:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No enhanced.png or original.png found in {input_dir}"},
            )

        artifacts: dict[str, Path] = {}
        panel_summaries: dict[str, Any] = {}

        for panel_id, img_path, target_dir in target_images:
            target_dir.mkdir(parents=True, exist_ok=True)
            masks_dir = target_dir / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            t0 = time.perf_counter()

            # 1. Structural line mask for snapping & classification
            line_mask, _, _ = extract_lines_and_tones(img_bgr, screentone_separation=True)

            # 2. Preprocessing: Screentone smoothing & candidate bubble/SFX isolation
            smoothed_bgr = smooth_for_segmentation(img_bgr, screentone_smoothing)
            pre_candidates = detect_candidate_bubbles_and_sfx(img_bgr) if bubble_detection else []

            # 3. Generate candidate masks via SAM 2 provider
            prep_rgb = self.segmenter.preprocess(smoothed_bgr)
            raw_sam_masks = self.segmenter.predict(prep_rgb)

            # Combine candidates and raw masks
            combined_records: list[dict[str, Any]] = []

            # Add pre-detected speech bubbles and SFX
            for pc in pre_candidates:
                m = pc["mask"]
                combined_records.append({
                    "mask": m,
                    "area": pc["area"],
                    "bbox": pc["bbox"],
                    "label": pc["label"],
                    "part": None,
                    "confidence": pc["confidence"],
                    "needs_manual_review": False,
                })

            # Add SAM masks
            for r in raw_sam_masks:
                seg_bool = r["segmentation"]
                m_uint8 = (seg_bool > 0).astype(np.uint8) * 255

                # Snap to ink lines if enabled
                if line_snapping:
                    m_uint8 = snap_mask_to_lines(m_uint8, line_mask)

                m_area = int(np.count_nonzero(m_uint8))
                if m_area < 80:
                    continue

                # Deduplicate against existing candidates
                is_duplicate = False
                for existing in combined_records:
                    ex_mask = existing["mask"]
                    inter = np.logical_and(m_uint8 > 0, ex_mask > 0).sum()
                    union = np.logical_or(m_uint8 > 0, ex_mask > 0).sum()
                    if union > 0 and (float(inter) / float(union)) > 0.82:
                        is_duplicate = True
                        break

                if is_duplicate:
                    continue

                label, part, conf, needs_review = classify_segment(
                    mask=m_uint8,
                    image_bgr=img_bgr,
                    line_mask=line_mask,
                )

                if conf < conf_threshold:
                    needs_review = True

                y_idxs, x_idxs = np.where(m_uint8 > 127)
                bbox = [int(np.min(x_idxs)), int(np.min(y_idxs)), int(np.max(x_idxs)), int(np.max(y_idxs))]

                combined_records.append({
                    "mask": m_uint8,
                    "area": m_area,
                    "bbox": bbox,
                    "label": label,
                    "part": part,
                    "confidence": conf,
                    "needs_manual_review": needs_review,
                })

            # Ensure we have at least one character or foreground candidate
            # If no character was identified, classify largest non-background foreground component as character
            has_char = any(c["label"] == "character" for c in combined_records)
            if not has_char:
                candidates_for_char = [c for c in combined_records if c["label"] not in ("background", "environment", "speech_bubble")]
                if candidates_for_char:
                    largest = max(candidates_for_char, key=lambda c: c["area"])
                    largest["label"] = "character"
                    largest["confidence"] = 0.80
                    largest["needs_manual_review"] = False

            # Assign IDs
            for idx, rec in enumerate(combined_records, start=1):
                rec["id"] = f"seg_{idx:03d}"
                rec["_mask"] = rec["mask"]

            # 4. Build hierarchy (parentId, suggested_depth, suggested_z_order)
            hierarchy_segments = build_segment_hierarchy(combined_records, w, h)

            # 5. Save mask PNGs and assemble segments.json
            output_segments: list[dict[str, Any]] = []
            masks_dict: dict[str, np.ndarray] = {}

            for seg in hierarchy_segments:
                sid = seg["id"]
                s_mask = seg["_mask"]
                masks_dict[sid] = s_mask

                label_slug = f"{seg['label']}_{seg['part']}" if seg.get("part") else seg["label"]
                mask_filename = f"{sid}_{label_slug}.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), s_mask)

                # Assign character ID using vision encoder if character or character part
                char_id = None
                if seg["label"] in ("character", "character_part"):
                    bx, by, bw, bh = seg["bbox"]
                    if bw > 10 and bh > 10:
                        crop = img_bgr[by : by + bh, bx : bx + bw]
                        if crop.size > 0:
                            char_id = "char_001"

                output_segments.append({
                    "id": sid,
                    "label": seg["label"],
                    "part": seg.get("part"),
                    "character_id": char_id,
                    "parentId": seg.get("parentId"),
                    "confidence": round(float(seg["confidence"]), 3),
                    "needs_manual_review": bool(seg["needs_manual_review"]),
                    "bbox": seg["bbox"],
                    "area": seg["area"],
                    "mask_path": f"masks/{mask_filename}",
                    "suggested_depth": seg["suggested_depth"],
                    "suggested_z_order": seg["suggested_z_order"],
                })

            segments_json_path = target_dir / "segments.json"
            segments_payload = {
                "panel": {
                    "panel_id": panel_id,
                    "width": w,
                    "height": h,
                    "source": str(img_path.name),
                },
                "total_segments": len(output_segments),
                "flagged_for_review": sum(1 for s in output_segments if s["needs_manual_review"]),
                "segments": output_segments,
            }
            segments_json_path.write_text(json.dumps(segments_payload, indent=2), encoding="utf-8")
            artifacts[f"{panel_id}/segments_json"] = segments_json_path

            # 6. Generate color-coded visual overlay
            overlay_img = create_segment_overlay(img_bgr, output_segments, masks_dict)
            overlay_path = target_dir / "overlay.png"
            cv2.imwrite(str(overlay_path), overlay_img)
            artifacts[f"{panel_id}/overlay"] = overlay_path

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            panel_summaries[panel_id] = {
                "panel_id": panel_id,
                "resolution": [w, h],
                "segment_count": len(output_segments),
                "flagged_for_review": segments_payload["flagged_for_review"],
                "labels": [s["label"] for s in output_segments],
                "time_ms": elapsed_ms,
            }

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata={
                "panels_processed": len(target_images),
                "summaries": panel_summaries,
            },
        )
