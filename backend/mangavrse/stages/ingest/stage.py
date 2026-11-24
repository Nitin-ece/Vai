"""
Ingest Stage — orchestrates PDF extraction, panel detection, reading order, and studio tools.

Input directory:
  Contains a manga PDF (*.pdf) or pre-extracted page images (page_*.png).

Output layout per panel:
  data/<project>/page_<NNN>/page_<NNN>.png
  data/<project>/page_<NNN>/debug_panels.png
  data/<project>/page_<NNN>/panels.json
  data/<project>/page_<NNN>/panel_<NNN>/original.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

from mangavrse.core.stage import AbstractStage, StageResult

from .classical_detector import ClassicalPanelDetector, PanelBox
from .page_classifier import PageClassifier
from .pdf_extractor import PDFExtractor
from .reading_order import sort_reading_order
from .studio_overlay import StudioOverlayHelper


class IngestStage(AbstractStage):
    """
    Ingests a manga PDF or image folder, detects panels per page,
    orders them by reading sequence, and produces panel crops and debug overlays.
    """

    def __init__(self) -> None:
        super().__init__(name="ingest")
        self.pdf_extractor = PDFExtractor(target_dpi=300)
        self.page_classifier = PageClassifier()
        self.detector = ClassicalPanelDetector()

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        artifacts: dict[str, Path] = {}
        metadata: dict[str, Any] = {
            "pages_processed": 0,
            "total_panels": 0,
            "non_story_pages": 0,
            "pages": [],
        }

        # 1. Discover inputs: look for PDF or direct images
        pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
        extracted_page_paths: list[Path] = []

        if pdf_files:
            pdf_path = pdf_files[0]
            pages_output_dir = output_dir / "pages"
            extracted = self.pdf_extractor.extract_pages(pdf_path, pages_output_dir)
            extracted_page_paths = [ep.image_path for ep in extracted]
        else:
            # Check for direct images (e.g. sample pages)
            image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
            for ext in image_extensions:
                extracted_page_paths.extend(input_dir.glob(ext))
            extracted_page_paths = sorted(
                [p for p in extracted_page_paths if not p.name.startswith("debug_")]
            )

        if not extracted_page_paths:
            return StageResult(
                success=False,
                output_dir=output_dir,
                artifacts={},
                metadata={"error": f"No PDF or image files found in {input_dir}"},
            )

        direction = config.get("reading_order", "rtl")

        # 2. Process each page
        for page_idx, page_img_path in enumerate(extracted_page_paths, start=1):
            page_name = f"page_{page_idx:03d}"
            page_out_dir = output_dir / page_name
            page_out_dir.mkdir(parents=True, exist_ok=True)

            # Copy full resolution page PNG
            target_page_png = page_out_dir / f"{page_name}.png"
            if page_img_path.resolve() != target_page_png.resolve():
                img_bgr = cv2.imread(str(page_img_path))
                if img_bgr is not None:
                    cv2.imwrite(str(target_page_png), img_bgr)
            else:
                img_bgr = cv2.imread(str(target_page_png))

            if img_bgr is None:
                continue

            artifacts[f"{page_name}/original"] = target_page_png

            # 3. Classify page (detect covers, color pages, blanks)
            classification = self.page_classifier.classify(target_page_png)
            page_meta: dict[str, Any] = {
                "page": page_idx,
                "is_story_page": classification.is_story_page,
                "page_type": classification.page_type,
                "flags": classification.flags,
            }

            if not classification.is_story_page:
                metadata["non_story_pages"] += 1

            # 4. Panel detection (or manual override)
            detected_panels = self.detector.detect_panels(img_bgr)
            ordered_panels = sort_reading_order(detected_panels, direction=direction)

            # Check for studio manual override
            final_panels, is_overridden = StudioOverlayHelper.check_and_apply_override(
                page_out_dir, ordered_panels
            )
            page_meta["is_overridden"] = is_overridden

            # 5. Crop panels and save
            panel_metadata_list: list[dict[str, Any]] = []
            for panel in final_panels:
                panel_dir = page_out_dir / panel.id
                panel_crop_path = panel_dir / "original.png"
                StudioOverlayHelper.crop_and_save_panel(img_bgr, panel, panel_crop_path)
                artifacts[f"{page_name}/{panel.id}"] = panel_crop_path
                panel_metadata_list.append(panel.to_dict())

            # 6. Save panels.json
            panels_json_path = page_out_dir / StudioOverlayHelper.PANELS_FILENAME
            panels_data = {
                "page": page_idx,
                "direction": direction,
                "is_story_page": classification.is_story_page,
                "page_type": classification.page_type,
                "panel_count": len(final_panels),
                "panels": panel_metadata_list,
            }
            panels_json_path.write_text(json.dumps(panels_data, indent=2), encoding="utf-8")
            artifacts[f"{page_name}/panels_json"] = panels_json_path

            # 7. Generate debug overlay
            debug_overlay_img = StudioOverlayHelper.draw_debug_overlay(img_bgr, final_panels)
            debug_img_path = page_out_dir / StudioOverlayHelper.DEBUG_IMAGE_FILENAME
            cv2.imwrite(str(debug_img_path), debug_overlay_img)
            artifacts[f"{page_name}/debug_overlay"] = debug_img_path

            page_meta["panel_count"] = len(final_panels)
            metadata["pages"].append(page_meta)
            metadata["total_panels"] += len(final_panels)
            metadata["pages_processed"] += 1

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata=metadata,
        )
