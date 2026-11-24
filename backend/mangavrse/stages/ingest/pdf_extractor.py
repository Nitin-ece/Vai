"""
PDF page extractor using PyMuPDF (fitz).

Extracts pages at 300 DPI. If a page contains a single embedded image
covering the full page (standard for scanned manga), it extracts the image
losslessly directly from the PDF stream to prevent moire artifacts.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf as fitz
from PIL import Image


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed
    image_path: Path
    width: int
    height: int
    dpi: int
    is_lossless_extraction: bool


class PDFExtractor:
    """Extracts pages from a manga PDF with lossless scan extraction and 300 DPI fallback."""

    def __init__(self, target_dpi: int = 300) -> None:
        self.target_dpi = target_dpi

    def extract_pages(self, pdf_path: Path, output_dir: Path) -> list[ExtractedPage]:
        """
        Extract all pages from *pdf_path* and save them as PNGs in *output_dir*.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(pdf_path))
        extracted_pages: list[ExtractedPage] = []

        try:
            for page_idx in range(len(doc)):
                page_num = page_idx + 1
                page = doc[page_idx]
                page_png_path = output_dir / f"page_{page_num:03d}.png"

                # Check if page is a single large embedded image
                lossless_extracted = self._try_extract_single_image(doc, page, page_png_path)

                if lossless_extracted:
                    with Image.open(page_png_path) as img:
                        w, h = img.size
                    extracted_pages.append(
                        ExtractedPage(
                            page_number=page_num,
                            image_path=page_png_path,
                            width=w,
                            height=h,
                            dpi=self.target_dpi,
                            is_lossless_extraction=True,
                        )
                    )
                else:
                    # Render at 300 DPI
                    zoom = self.target_dpi / 72.0
                    matrix = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    pix.save(str(page_png_path))

                    extracted_pages.append(
                        ExtractedPage(
                            page_number=page_num,
                            image_path=page_png_path,
                            width=pix.width,
                            height=pix.height,
                            dpi=self.target_dpi,
                            is_lossless_extraction=False,
                        )
                    )
        finally:
            doc.close()

        return extracted_pages

    def _try_extract_single_image(self, doc: fitz.Document, page: fitz.Page, out_path: Path) -> bool:
        """
        If the page consists of a single raster image covering the whole page area,
        extract it losslessly without re-sampling.
        """
        images = page.get_images(full=True)
        if len(images) != 1:
            return False

        xref = images[0][0]
        base_image = doc.extract_image(xref)
        if not base_image or "image" not in base_image:
            return False

        image_bytes = base_image["image"]
        img_width = base_image.get("width", 0)
        img_height = base_image.get("height", 0)

        # Check if the image resolution is sufficiently high (at least 600px dimension)
        if img_width < 600 or img_height < 600:
            return False

        # Save to out_path as PNG
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Convert CMYK or indexed to RGB if necessary
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            pil_img.save(out_path, format="PNG")
            return True
        except Exception:
            return False
