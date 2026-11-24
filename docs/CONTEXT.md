# VAI (Volumetric Art Intelligence) — CONTEXT.md

> **VAI: Manga to 2.5D Volumetric Art Intelligence Engine**
>
> Takes a real manga PDF, extracts panels, and turns ONE panel at a time
> into a layered, vectorized, colored, depth-aware 2.5D scene viewable
> in a browser.

---

## 1  Project Summary

VAI (Volumetric Art Intelligence) is a monorepo containing a **Python backend** (FastAPI +
PyTorch + OpenCV) and a **Next.js frontend** (Three.js + GSAP).  The
backend exposes a *stage-based pipeline*: each stage reads files from
an input directory, writes files to an output directory, and caches
its results.  The frontend loads the final `scene.json` and renders an
interactive 2.5D scene using `@react-three/fiber`.

**MVP target**: one manga panel → one interactive 2.5D browser scene.

---

## 2  Pipeline Stages

```
PDF ──▶ ① Ingest ──▶ ② Enhance ──▶ ③ Segmentation
                                          │
        ⑧ Scene Assembly ◀── ⑦ Colorization ◀── ⑥ Vectorization
              │                                       │
              │                              ⑤ Depth Estimation
              ▼
         scene.json  ──▶  Three.js viewer
```

| #   | Stage              | Input                     | Output                          |
| --- | ------------------ | ------------------------- | ------------------------------- |
| 1   | Ingest             | `manga.pdf`               | `page_NNN.png`, `panels.json`, `panel_NNN/original.png` |
| 2   | Enhance            | `original.png`            | `enhanced.png`, `comparison.png`, `metrics.json` |
| 3   | Segmentation       | enhanced panel crop       | layer masks (PNG per layer)     |
| 4   | Depth Estimation   | enhanced panel crop       | depth map (PFM / PNG16)         |
| 5   | Vectorization      | layer masks               | SVG traces per layer            |
| 6   | Colorization       | SVGs + reference panel    | colored SVGs / palette JSON     |
| 7   | Scene Assembly     | all of the above          | `scene.json`                    |

Each stage is a subclass of `AbstractStage` with a typed
`run(input_dir, output_dir, config) -> StageResult` contract.

---

## 3  Folder Layout

```
mangavrse/
├── backend/
│   ├── mangavrse/
│   │   ├── core/           # Stage, Cache, Config, ModelProvider, Registry
│   │   ├── stages/         # Concrete stage implementations
│   │   ├── api/            # FastAPI app + routes
│   │   └── cli/            # Click-based CLI
│   └── pyproject.toml
├── frontend/               # Next.js + Tailwind + Three.js
│   ├── src/app/            # App Router pages
│   └── package.json
├── data/                   # ⛔ gitignored — all runtime data
├── docs/
│   └── CONTEXT.md          # ← you are here
├── tests/                  # pytest test suite
├── scripts/
│   └── gpu_check.py
├── config.yaml             # single source of truth
├── Makefile                # make setup / make test / make gpu-check
└── .gitignore
```

---

## 4  Per-Panel Data Layout

All runtime data lives under `data/`.  Within it, every panel gets a
deterministic directory tree:

```
data/
└── <project>/                  # e.g. "one_piece_ch1042"
    └── page_<NNN>/             # zero-padded page number
        └── panel_<NNN>/        # zero-padded panel index
            ├── extraction/     # stage 1 outputs
            ├── detection/      # stage 2 outputs
            ├── segmentation/   # stage 3 outputs
            ├── depth/          # stage 4 outputs
            ├── vectorization/  # stage 5 outputs
            ├── colorization/   # stage 6 outputs
            ├── assembly/       # stage 7 outputs
            │   └── scene.json  # ★ final output
            └── thumbnails/     # optional previews
```

Each stage directory contains:
- The stage's output files (PNGs, SVGs, JSONs, etc.)
- `.cache_manifest.json` — cache metadata (auto-generated)

This layout maps directly to object-storage prefixes (`project/page/panel/stage/`)
when migrating away from the local filesystem.

---

## 5  `scene.json` Schema (v1.0)

```jsonc
{
  "version": "1.0",
  "panel": {
    "project": "my_manga",
    "page": 1,
    "panel": 3,
    "source": "extraction/page_001.png",
    "width": 800,
    "height": 1200
  },
  "layers": [
    {
      "id": "layer_001",
      "type": "background | character | speech_bubble | sfx | foreground",
      "parentId": null,            // for grouped / nested layers
      "svgPath": "vectorization/layer_001.svg",
      "pngPath": "segmentation/layer_001.png",
      "color": "#FFFFFF",          // dominant or assigned color
      "depth": 0.0,               // normalized 0.0 (far) → 1.0 (near)
      "zOrder": 0,                // integer draw order
      "transform": {
        "x": 0, "y": 0,           // position offset (px)
        "scaleX": 1.0, "scaleY": 1.0,
        "rotation": 0.0           // degrees
      },
      "metadata": {}              // stage-specific extras
    }
  ],
  "camera": {
    "fov": 45,
    "near": 0.1,
    "far": 100.0,
    "position": [0, 0, 5]
  }
}
```

### Layer types

| Type            | Description                                    |
| --------------- | ---------------------------------------------- |
| `background`    | Full-panel background art                      |
| `character`     | A segmented character / figure                 |
| `speech_bubble` | Dialogue balloon (with text removed or kept)   |
| `sfx`           | Sound-effect text / onomatopoeia               |
| `foreground`    | Foreground overlay elements                    |

---

## 6  Stage Interface Contract

```python
class AbstractStage(ABC):
    name: str

    def execute(input_dir, output_dir, config) -> StageResult:
        # 1. compute cache key = SHA-256(input files + config)
        # 2. if cache valid → return cached StageResult
        # 3. else → call self.run(...), save manifest, return result

    @abstractmethod
    def run(input_dir, output_dir, config) -> StageResult:
        ...
```

`StageResult` carries: `success`, `output_dir`, `artifacts` dict,
`cached` flag, `elapsed_seconds`, and free-form `metadata`.

---

## 7  ModelProvider Protocol

```python
class ModelProvider(Protocol):
    name: str
    def load(device, weights) -> None
    def preprocess(image) -> Any
    def predict(preprocessed) -> Any
    def postprocess(raw_output) -> Any
```

Implementations are registered in `ModelRegistry` and resolved by the
`name` key in `config.yaml → models.<stage>.name`.

---

## 8  Configuration

All tunables live in `config.yaml` (project root).  Loaded via Pydantic
into `MangaVrseConfig`.  Sections: `project`, `device`, `models`,
`depth`, `paths`, `server`.

---

## 9  Running

```bash
make setup          # venv + pip install + npm install
make test           # pytest + (vitest when added)
make gpu-check      # CUDA / VRAM diagnostics

# CLI
mangavrse list
mangavrse run noop -i ./data/in -o ./data/out

# API
make serve          # uvicorn on :8000
POST /api/v1/stages/noop/run { input_dir, output_dir }
```
