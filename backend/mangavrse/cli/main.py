"""
CLI for running pipeline stages.

Usage:
    mangavrse run noop --input-dir ./in --output-dir ./out
    mangavrse list
    mangavrse gpu-check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from mangavrse.core.config import load_config
from mangavrse.core.stage import AbstractStage
from mangavrse.stages.assembly import AssemblyStage
from mangavrse.stages.colorize import ColorizeStage
from mangavrse.stages.depth import DepthStage
from mangavrse.stages.enhance import EnhanceStage
from mangavrse.stages.ingest import IngestStage
from mangavrse.stages.layers import LayersStage
from mangavrse.stages.noop import NoOpStage
from mangavrse.stages.segment import SegmentStage
from mangavrse.stages.vectorize import VectorizeStage

# ----- Stage registry (mirrors api/routes.py) ---------------------
_STAGES: dict[str, AbstractStage] = {
    "noop": NoOpStage(),
    "ingest": IngestStage(),
    "enhance": EnhanceStage(),
    "vectorize": VectorizeStage(),
    "segment": SegmentStage(),
    "layers": LayersStage(),
    "colorize": ColorizeStage(),
    "depth": DepthStage(),
    "assembly": AssemblyStage(),
}


@click.group()
def cli() -> None:
    """MangaVrse — Manga to 2.5D Vector Engine CLI."""


@cli.command("list")
def list_stages() -> None:
    """List all registered pipeline stages."""
    for name in sorted(_STAGES):
        click.echo(f"  • {name}")


@cli.command("run")
@click.argument("stage_name")
@click.option("--input-dir", "-i", required=True, type=click.Path(), help="Input directory")
@click.option("--output-dir", "-o", required=True, type=click.Path(), help="Output directory")
@click.option("--config", "-c", type=click.Path(exists=True), default=None, help="Config YAML path")
def run_stage(stage_name: str, input_dir: str, output_dir: str, config: str | None) -> None:
    """Execute a pipeline stage by name."""
    stage = _STAGES.get(stage_name)
    if stage is None:
        click.echo(f"Error: unknown stage '{stage_name}'. Use 'mangavrse list'.", err=True)
        sys.exit(1)

    cfg = load_config(config)
    merged = cfg.stage_config(stage_name)

    inp = Path(input_dir)
    out = Path(output_dir)
    inp.mkdir(parents=True, exist_ok=True)

    result = stage.execute(inp, out, merged)

    output = {
        "stage": stage_name,
        "success": result.success,
        "cached": result.cached,
        "elapsed_seconds": result.elapsed_seconds,
        "artifacts": {k: str(v) for k, v in result.artifacts.items()},
        "metadata": result.metadata,
    }
    click.echo(json.dumps(output, indent=2))

    if not result.success:
        sys.exit(1)


@cli.command("gpu-check")
def gpu_check() -> None:
    """Print GPU / CUDA diagnostic info."""
    # Reuse the standalone script logic
    from scripts import gpu_check as _gc  # noqa: delayed import

    _gc.main()


if __name__ == "__main__":
    cli()
