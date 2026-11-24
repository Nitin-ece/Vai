"""
NoOp stage — a passthrough used for testing the stage interface,
caching, CLI, and API wiring.

It copies every file from input_dir → output_dir unchanged and
records them as artifacts.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mangavrse.core.stage import AbstractStage, StageResult


class NoOpStage(AbstractStage):
    """Does nothing except copy inputs to outputs (identity stage)."""

    def __init__(self) -> None:
        super().__init__(name="noop")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        artifacts: dict[str, Path] = {}

        if input_dir.exists():
            for src in sorted(input_dir.iterdir()):
                if src.is_file() and not src.name.startswith("."):
                    dst = output_dir / src.name
                    shutil.copy2(src, dst)
                    artifacts[src.name] = dst

        return StageResult(
            success=True,
            output_dir=output_dir,
            artifacts=artifacts,
            metadata={"copied_files": len(artifacts)},
        )
