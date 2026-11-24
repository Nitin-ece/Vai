"""Tests for the AbstractStage interface and NoOpStage."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mangavrse.core.stage import AbstractStage, StageResult
from mangavrse.stages.noop import NoOpStage


# ------------------------------------------------------------------
# AbstractStage contract tests
# ------------------------------------------------------------------

class _FailStage(AbstractStage):
    """A stage that always fails (for testing)."""

    def __init__(self):
        super().__init__("fail_stage")

    def run(self, input_dir: Path, output_dir: Path, config: dict[str, Any]) -> StageResult:
        return StageResult(success=False, output_dir=output_dir)


class TestAbstractStage:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AbstractStage("test")  # type: ignore[abstract]

    def test_subclass_has_name(self, noop_stage: NoOpStage):
        assert noop_stage.name == "noop"

    def test_failed_stage_does_not_cache(self, tmp_dirs):
        inp, out = tmp_dirs
        stage = _FailStage()
        result = stage.execute(inp, out, {})
        assert not result.success
        assert not result.cached
        # No manifest should be written
        assert not (out / ".cache_manifest.json").exists()


# ------------------------------------------------------------------
# NoOpStage tests
# ------------------------------------------------------------------

class TestNoOpStage:
    def test_empty_input(self, noop_stage: NoOpStage, tmp_dirs):
        inp, out = tmp_dirs
        result = noop_stage.execute(inp, out, {})
        assert result.success
        assert result.artifacts == {}
        assert result.metadata["copied_files"] == 0

    def test_copies_files(self, noop_stage: NoOpStage, tmp_dirs, sample_file):
        inp, out = tmp_dirs
        result = noop_stage.execute(inp, out, {})
        assert result.success
        assert "sample.txt" in result.artifacts
        copied = out / "sample.txt"
        assert copied.exists()
        assert copied.read_text(encoding="utf-8") == "hello mangavrse"

    def test_result_has_timing(self, noop_stage: NoOpStage, tmp_dirs):
        inp, out = tmp_dirs
        result = noop_stage.execute(inp, out, {})
        assert result.elapsed_seconds >= 0

    def test_ignores_dotfiles(self, noop_stage: NoOpStage, tmp_dirs):
        inp, out = tmp_dirs
        (inp / ".hidden").write_text("secret")
        (inp / "visible.txt").write_text("hello")
        result = noop_stage.execute(inp, out, {})
        assert "visible.txt" in result.artifacts
        assert ".hidden" not in result.artifacts
