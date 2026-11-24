"""
Layers stage package.
"""
from .stage import LayersStage
from .inpainter import InpainterProvider
from .cutter import extract_layer_rgba, pad_and_feather_mask, create_clipped_svg
from .manga_inpaint import inpaint_manga_background
from .parallax_tester import render_parallax_test

__all__ = [
    "LayersStage",
    "InpainterProvider",
    "extract_layer_rgba",
    "pad_and_feather_mask",
    "create_clipped_svg",
    "inpaint_manga_background",
    "render_parallax_test",
]
