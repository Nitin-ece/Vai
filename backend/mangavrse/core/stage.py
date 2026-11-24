"""
Abstract stage interface.

Every pipeline step (PDF extraction, panel detection, segmentation, …)
subclasses `AbstractStage` and implements `run()`.  The public entry
point is `execute()`, which wraps `run()` with per-stage caching.

Contracts
---------
* **Input**:  a directory of files produced by the previous stage.
* **Output**: a directory of files consumed by the next stage.
* **Config**: a plain dict (subset of the global config.yaml).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheManager


@dataclass
class StageResult:
    """Returned by every stage execution."""

    success: bool
    output_dir: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    cached: bool = False
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class AbstractStage(ABC):
    """
    Base class for all pipeline stages.

    Subclasses MUST:
      1. Call ``super().__init__(name)`` with a unique stage name.
      2. Implement ``run(input_dir, output_dir, config) -> StageResult``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._cache = CacheManager()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------
    @abstractmethod
    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        """Execute the stage's core logic.  Implemented by each stage."""
        ...

    # ------------------------------------------------------------------
    # Public entry point (adds caching)
    # ------------------------------------------------------------------
    def execute(
        self,
        input_dir: Path,
        output_dir: Path,
        config: dict[str, Any],
    ) -> StageResult:
        """Run the stage with caching.  Call this, not ``run()``."""
        cache_key = self._cache.compute_key(input_dir, config)

        if self._cache.is_valid(output_dir, cache_key):
            return StageResult(
                success=True,
                output_dir=output_dir,
                cached=True,
                artifacts=self._cache.load_artifacts(output_dir),
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.monotonic()
        result = self.run(input_dir, output_dir, config)
        result.elapsed_seconds = round(time.monotonic() - start, 4)

        if result.success:
            self._cache.save_manifest(
                output_dir=output_dir,
                cache_key=cache_key,
                stage_name=self.name,
                artifacts=result.artifacts,
                config=config,
                input_dir=input_dir,
            )

        return result
