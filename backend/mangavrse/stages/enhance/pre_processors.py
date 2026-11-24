"""
Pre-processing modules for the Enhance stage:
- Contrast correction (paper white & ink black calibration, CLAHE)
- Screentone-aware denoising (prevents dot smearing and moiré)
- Deblurring / optical restoration
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_contrast(
    image_bgr: np.ndarray,
    method: str = "percentile_stretch",
    white_percentile: float = 99.0,
    black_percentile: float = 1.0,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """
    Calibrates contrast by stretching paper background to pure white (255)
    and linework to deep black (0).
    """
    if method == "clahe":
        # CLAHE in LAB color space on Luminance channel
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Default: Percentile stretch (standard manga white-balance)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    p_low = np.percentile(gray, black_percentile)
    p_high = np.percentile(gray, white_percentile)

    if p_high <= p_low:
        return image_bgr.copy()

    # Stretch range [p_low, p_high] -> [0, 255]
    stretched = np.clip((image_bgr.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)), 0, 255)
    return stretched.astype(np.uint8)


def detect_screentone_mask(gray: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Detects periodic halftone / screentone dot fields.
    Computes local Laplacian energy in a 5x5 window; screentones exhibit
    high-frequency alternating gradients unlike flat paper or solid ink.
    Returns a float mask [0.0, 1.0] where 1.0 indicates screentone dots.
    """
    # Laplacian high-frequency detector
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    abs_lap = np.abs(lap)

    # Local average of high frequency magnitude
    kernel_size = 9
    local_energy = cv2.boxFilter(abs_lap, ddepth=-1, ksize=(kernel_size, kernel_size))

    # Normalized energy
    norm_energy = local_energy / 255.0

    # Intermediate energy is screentone; very high is solid line borders; very low is paper
    screentone_mask = np.clip(
        (norm_energy - threshold) / (threshold * 2.0),
        0.0,
        1.0,
    )
    # Smooth the mask to avoid harsh cutoffs
    screentone_mask = cv2.GaussianBlur(screentone_mask, (7, 7), 2.0)
    return screentone_mask[:, :, np.newaxis]


def apply_denoise(
    image_bgr: np.ndarray,
    method: str = "nlmeans",
    h: float = 3.0,
    preserve_screentone: bool = True,
    screentone_threshold: float = 0.12,
) -> np.ndarray:
    """
    Applies denoising while protecting periodic screentone dots from smearing.
    """
    if method == "bilateral":
        denoised = cv2.bilateralFilter(image_bgr, d=7, sigmaColor=int(h * 10), sigmaSpace=7)
    else:
        # Non-Local Means (NLM)
        if len(image_bgr.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image_bgr,
                None,
                h=h,
                hColor=h,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image_bgr,
                None,
                h=h,
                templateWindowSize=7,
                searchWindowSize=21,
            )

    if not preserve_screentone:
        return denoised

    # Screentone protection: blend original over denoised where halftone dots exist
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    dot_mask = detect_screentone_mask(gray, threshold=screentone_threshold)

    # Blend: original in screentone zones, denoised in flat paper / smooth ink
    blended = (image_bgr.astype(np.float32) * dot_mask) + (denoised.astype(np.float32) * (1.0 - dot_mask))
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def apply_deblur(
    image_bgr: np.ndarray,
    radius: float = 1.0,
    amount: float = 0.6,
) -> np.ndarray:
    """
    Deblurs soft optical scanner focus using unsharp deconvolution filter.
    """
    ksize = int(math.ceil(radius * 3)) * 2 + 1
    blurred = cv2.GaussianBlur(image_bgr, (ksize, ksize), radius)

    # Unsharp mask formula
    deblurred = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(deblurred, 0, 255).astype(np.uint8)


import math
