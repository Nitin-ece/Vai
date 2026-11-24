"""
Palette Manager: manages character palettes, scene color schemes,
and cross-panel palette inheritance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_ANIME_PALETTES: dict[str, dict[str, str]] = {
    "default_character": {
        "skin": "#FCE5D8",
        "hair": "#4A2E1B",
        "eyes": "#1E6091",
        "clothing_primary": "#2B4C7E",
        "clothing_secondary": "#D96B27",
        "accessories": "#D4AF37",
        "body": "#2B4C7E",
    },
    "environment": {
        "sky": "#90CAF9",
        "ground": "#8D6E63",
        "foliage": "#4CAF50",
        "props": "#A1887F",
        "background": "#F5F5F5",
    },
    "speech_bubble": {
        "fill": "#FFFFFF",
        "stroke": "#000000",
    },
    "sfx": {
        "fill": "#FF5722",
        "stroke": "#000000",
    },
    "effect": {
        "fill": "#FFD54F",
        "stroke": "#FF9800",
    },
}


class PaletteManager:
    """Handles character color palettes and cross-panel visual continuity."""

    def __init__(self, palette_data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = palette_data or {
            "characters": {
                "char_001": {
                    "name": "Main Character",
                    "palette": DEFAULT_ANIME_PALETTES["default_character"].copy(),
                }
            },
            "environment": DEFAULT_ANIME_PALETTES["environment"].copy(),
            "speech_bubble": DEFAULT_ANIME_PALETTES["speech_bubble"].copy(),
            "sfx": DEFAULT_ANIME_PALETTES["sfx"].copy(),
            "effect": DEFAULT_ANIME_PALETTES["effect"].copy(),
        }

    @classmethod
    def load(cls, path: Path | str) -> PaletteManager:
        p = Path(path)
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                return cls(raw)
            except Exception:
                pass
        return cls()

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get_character_palette(self, character_id: str = "char_001") -> dict[str, str]:
        chars = self.data.setdefault("characters", {})
        char_entry = chars.setdefault(character_id, {
            "name": character_id,
            "palette": DEFAULT_ANIME_PALETTES["default_character"].copy(),
        })
        return char_entry.setdefault("palette", DEFAULT_ANIME_PALETTES["default_character"].copy())

    def set_character_color(self, character_id: str, part: str, color_hex: str) -> None:
        palette = self.get_character_palette(character_id)
        palette[part] = color_hex

    def resolve_layer_color(
        self,
        label: str,
        part: str | None = None,
        character_id: str = "char_001",
    ) -> str:
        """Determines the hex color code for a specific layer label and part."""
        if label in ("speech_bubble",):
            return self.data.get("speech_bubble", {}).get("fill", "#FFFFFF")
        if label in ("sfx",):
            return self.data.get("sfx", {}).get("fill", "#FF5722")
        if label in ("effect",):
            return self.data.get("effect", {}).get("fill", "#FFD54F")
        if label in ("background",):
            return self.data.get("environment", {}).get("background", "#F5F5F5")
        if label in ("environment", "prop"):
            return self.data.get("environment", {}).get(label, "#8D6E63")

        # Character and Character Parts
        char_palette = self.get_character_palette(character_id)
        if part and part in char_palette:
            return char_palette[part]

        # Part heuristics
        if part in ("head", "face", "skin"):
            return char_palette.get("skin", "#FCE5D8")
        if part in ("hair",):
            return char_palette.get("hair", "#4A2E1B")
        if part in ("eyes",):
            return char_palette.get("eyes", "#1E6091")
        if part in ("clothing", "clothing_primary"):
            return char_palette.get("clothing_primary", "#2B4C7E")
        if part in ("body",):
            return char_palette.get("body", "#2B4C7E")

        # Default fallback for character
        return char_palette.get("clothing_primary", "#2B4C7E")
