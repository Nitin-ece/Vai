"""Shared fixtures for MangaVrse tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from mangavrse.core.stage import AbstractStage, StageResult
from mangavrse.stages.noop import NoOpStage


@pytest.fixture
def tmp_dirs(tmp_path: Path):
    """Create input_dir / output_dir under pytest's tmp_path."""
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir()
    return inp, out


@pytest.fixture
def noop_stage() -> NoOpStage:
    return NoOpStage()


@pytest.fixture
def sample_file(tmp_dirs):
    """Create a small sample file in the input directory."""
    inp, out = tmp_dirs
    sample = inp / "sample.txt"
    sample.write_text("hello mangavrse", encoding="utf-8")
    return sample
