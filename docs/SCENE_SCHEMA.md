# MangaVrse `scene.json` Specification (v1.0)

This document specifies the schema and conventions for `scene.json`, the unified interchange format produced by the **Assembly** stage and consumed by the **3D Viewer** (Three.js), game engines, and video rendering pipelines.

The formal JSON Schema is located at [docs/scene_schema.json](file:///c:/Users/mishr/Downloads/mangavrse/docs/scene_schema.json).

---

## 1. Top-Level Structure

A valid `scene.json` file contains four required top-level properties:

```jsonc
{
  "version": "1.0",
  "panel": { ... },
  "camera": { ... },
  "layers": [ ... ]
}
```

### 1.1 `panel` Object
Defines source dimensions and project hierarchy:
- `project` (`string`): Project or manga series identifier (e.g. `"benchmark"` or `"one_piece_ch1042"`).
- `page` (`integer`): Zero-indexed or one-indexed page number.
- `panel` (`integer`): Panel index on the page.
- `source` (`string`): Path or filename of the source panel image.
- `width` (`integer`): Pixel width of the panel canvas.
- `height` (`integer`): Pixel height of the panel canvas.

---

## 2. `camera` Specification & Parallax Limits

Configures the default 3D camera and clamps parallax motion to ensure background gaps and inpaint seams remain concealed:

- `fov` (`number`): Vertical field of view in degrees (default: `45.0`).
- `near` (`number`): Near clipping plane (default: `0.1`).
- `far` (`number`): Far clipping plane (default: `100.0`).
- `position` (`[number, number, number]`): Default camera coordinates `[X, Y, Z]` (default: `[0.0, 0.0, 5.0]`).
- `limits` (`object`):
  - `minDistance` (`number`): Closest allowed camera distance during zoom/push-in (default: `3.2`).
  - `maxDistance` (`number`): Furthest allowed camera distance (default: `5.5`).
  - `maxPitch` (`number`): Maximum vertical tilt angle in radians (default: `0.045` rad / ~`2.57°`).
  - `maxYaw` (`number`): Maximum horizontal orbit angle in radians (default: `0.060` rad / ~`3.44°`).

---

## 3. `layers` Array Specification

Each entry in `layers` represents a discrete 2.5D visual plane:

| Property | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique layer identifier (e.g. `"seg_001"`, `"layer_bg"`). |
| `type` | `string` | Semantic classification: `background`, `character`, `character_part`, `speech_bubble`, `sfx`, `foreground`, `effect`, `prop`, `environment`. |
| `parentId` | `string \| null` | Identifier of parent layer for hierarchical grouping, or `null`. |
| `svgPath` | `string` | Path to vector layer asset (relative to panel root). |
| `pngPath` | `string` | Path to transparent RGBA raster cutout (relative to panel root). |
| `color` | `string` | Hex color code assigned by the colorize stage (e.g. `"#E63946"`). |
| `depth` | `number` | Normalized depth in $[0.0, 1.0]$ where **$0.0 = \text{far}$** (background) and **$1.0 = \text{near}$** (foreground). |
| `zOrder` | `integer` | Integer rendering order ($0 = \text{backplate}$, $N-1 = \text{frontmost}$). |
| `transform` | `object` | 2D offset and scale: `{ "x": 0, "y": 0, "scaleX": 1.0, "scaleY": 1.0, "rotation": 0.0 }`. |
| `metadata` | `object` | Extended stage properties: `part`, `bbox`, confidence scores, and `animations`. |

### 3.1 Layer Animation Hooks (`metadata.animations`)

Animation hooks allow the viewer to apply subtle, organic 2.5D ambient movement. All hooks are **disabled by default** (`enabled: false`) to preserve clean static reading:

- **`sway`**: Continuous subtle rotation for hair and clothing layers.
  - `enabled` (`boolean`): `false`
  - `speed` (`number`): Angular frequency multiplier (e.g. `1.0`).
  - `amplitude` (`number`): Angular oscillation in radians (e.g. `0.02`).
- **`float`**: Subtle vertical bobbing for speech balloons, sound effects, and aura effects.
  - `enabled` (`boolean`): `false`
  - `speed` (`number`): Frequency multiplier (e.g. `1.2`).
  - `amplitude` (`number`): Vertical oscillation in world units (e.g. `0.03`).
- **`blink`**: Cyclic opacity or visibility pulse for eye layers.
  - `enabled` (`boolean`): `false`
  - `interval` (`number`): Period between blinks in seconds (e.g. `4.0`).

---

## 4. 3D World Space Registration

To ensure pixel-perfect alignment with the original 2D panel when viewed from the resting camera position $(0, 0, Z_{\text{cam}})$:

1. **Z-Position**:
   $$Z_{\text{layer}} = (\text{depth} - 0.5) \times D_{\text{scale}}$$
   where $D_{\text{scale}} \approx 1.5$ world units.

2. **Perspective Scale Compensation**:
   $$\text{scale}(Z) = \frac{Z_{\text{cam}} - Z_{\text{layer}}}{Z_{\text{cam}}}$$
   This scale adjustment ensures that at $Z_{\text{cam}} = 5.0$, all layers project to their exact 2D pixel coordinates, avoiding spatial distortion.
