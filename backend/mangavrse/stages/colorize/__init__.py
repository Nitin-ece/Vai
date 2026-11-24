"""
Colorize stage package.
"""
from .stage import ColorizeStage
from .colorizer import ColorizerProvider
from .palette_manager import PaletteManager
from .renderer import render_colored_panel
from .vector_styler import assemble_colored_panel_svg, style_layer_svg

__all__ = [
    "ColorizeStage",
    "ColorizerProvider",
    "PaletteManager",
    "render_colored_panel",
    "assemble_colored_panel_svg",
    "style_layer_svg",
]
