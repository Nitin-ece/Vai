"""
Typed configuration loaded from config.yaml.

Uses Pydantic v2 models so every value is validated and can be
overridden via environment variables (MANGAVRSE__SECTION__KEY).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ----- Sub-models ------------------------------------------------

class DeviceConfig(BaseModel):
    prefer_cuda: bool = True
    fallback_cpu: bool = True


class ModelEntry(BaseModel):
    name: str
    provider: str
    weights: str | None = None
    threshold: float | None = None
    output_format: str | None = None
    points_per_side: int | None = None

    model_config = {"extra": "allow"}  # forward-compat for new keys


class ModelsConfig(BaseModel):
    panel_detector: ModelEntry = Field(default_factory=lambda: ModelEntry(name="yolov8", provider="ultralytics"))
    depth_estimator: ModelEntry = Field(default_factory=lambda: ModelEntry(name="midas_v3_dpt_large", provider="intel"))
    segmenter: ModelEntry = Field(default_factory=lambda: ModelEntry(name="sam2_hiera_large", provider="meta"))
    vectorizer: ModelEntry = Field(default_factory=lambda: ModelEntry(name="potrace", provider="builtin"))

    model_config = {"extra": "allow"}


class DepthConfig(BaseModel):
    near: float = 0.1
    far: float = 10.0
    default_layers: int = 5
    model_weight: float = 0.50
    bilateral_d: int = 7
    bilateral_sigma: float = 50.0
    smooth_sigma: float = 1.5


class PathsConfig(BaseModel):
    uploads: str = "data/uploads"
    output: str = "data/output"
    models_cache: str = "data/models"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


class ProjectConfig(BaseModel):
    name: str = "mangavrse"
    version: str = "0.1.0"
    data_dir: str = "data/"


class EnhanceConfig(BaseModel):
    enabled_steps: dict[str, bool] = Field(
        default_factory=lambda: {
            "contrast": True,
            "denoise": True,
            "deblur": True,
            "upscale": True,
            "sharpen": True,
        }
    )
    contrast: dict[str, Any] = Field(
        default_factory=lambda: {
            "method": "percentile_stretch",
            "paper_white_percentile": 99.0,
            "ink_black_percentile": 1.0,
        }
    )
    denoise: dict[str, Any] = Field(
        default_factory=lambda: {
            "method": "nlmeans",
            "h": 3.0,
            "preserve_screentone": True,
            "screentone_threshold": 0.12,
        }
    )
    deblur: dict[str, Any] = Field(
        default_factory=lambda: {
            "radius": 1.0,
            "amount": 0.6,
        }
    )
    upscale: dict[str, Any] = Field(
        default_factory=lambda: {
            "scale": 4,
            "model_name": "realesrgan_anime6b",
            "tile_size": 512,
            "tile_pad": 32,
        }
    )
    sharpen: dict[str, Any] = Field(
        default_factory=lambda: {
            "amount": 0.7,
            "radius": 0.8,
        }
    )

    model_config = {"extra": "allow"}


class VectorizeConfig(BaseModel):
    extractor: dict[str, Any] = Field(
        default_factory=lambda: {
            "adaptive_block_size": 15,
            "adaptive_c": 8,
            "screentone_separation": True,
            "min_tone_frequency": 0.12,
        }
    )
    tracer: dict[str, Any] = Field(
        default_factory=lambda: {
            "provider": "potrace",
            "turdsize": 2,
            "alphamax": 1.0,
            "opttolerance": 0.8,
        }
    )
    optimizer: dict[str, Any] = Field(
        default_factory=lambda: {
            "min_path_area": 12.0,
            "rdp_tolerance": 1.2,
            "smooth_bezier": True,
            "merge_distance": 2.0,
        }
    )
    screentone: dict[str, Any] = Field(
        default_factory=lambda: {
            "mode": "pattern_fill",
            "default_dot_size": 2.0,
            "default_spacing": 5.0,
        }
    )

    model_config = {"extra": "allow"}


class SegmentConfig(BaseModel):
    model_name: str = "sam2_hiera_large"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.75
    stability_score_thresh: float = 0.85
    min_mask_region_area: int = 120
    screentone_smoothing: bool = True
    bubble_detection: bool = True
    line_snapping: bool = True
    confidence_threshold: float = 0.65

    model_config = {"extra": "allow"}


class LayersConfig(BaseModel):
    mask_padding: int = 2
    feather_radius: float = 2.0
    inpainting: dict[str, Any] = Field(
        default_factory=lambda: {
            "provider": "default",
            "method": "telea_texture",
            "radius": 5,
            "line_extension": True,
            "screentone_synthesis": True,
        }
    )
    parallax_test: dict[str, Any] = Field(
        default_factory=lambda: {
            "shift_scale": 25.0,
            "angle_degrees": -30.0,
        }
    )

    model_config = {"extra": "allow"}


class ColorizeConfig(BaseModel):
    character_id: str = "char_001"
    palette_path: str | None = None
    overrides: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# ----- Root model ------------------------------------------------

class MangaVrseConfig(BaseModel):
    """Root configuration — mirrors config.yaml exactly."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    enhance: EnhanceConfig = Field(default_factory=EnhanceConfig)
    vectorize: VectorizeConfig = Field(default_factory=VectorizeConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    layers: LayersConfig = Field(default_factory=LayersConfig)
    colorize: ColorizeConfig = Field(default_factory=ColorizeConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    def get_device_string(self) -> str:
        """Return 'cuda' if available and preferred, else 'cpu'."""
        if self.device.prefer_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    def stage_config(self, stage_name: str) -> dict[str, Any]:
        """Extract the config dict relevant to a given stage name."""
        return self.model_dump()


# ----- Loader ----------------------------------------------------

_CONFIG_SEARCH_PATHS = [
    Path("config.yaml"),
    Path("config.yml"),
    Path(__file__).resolve().parents[3] / "config.yaml",   # repo root
]


def load_config(path: Path | str | None = None) -> MangaVrseConfig:
    """Load and validate config from YAML.  Falls back to defaults."""
    if path is not None:
        p = Path(path)
    else:
        p = next((c for c in _CONFIG_SEARCH_PATHS if c.exists()), None)

    if p is not None and p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        raw = {}

    return MangaVrseConfig(**raw)
