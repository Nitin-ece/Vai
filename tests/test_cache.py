"""Tests for the CacheManager and stage-level caching behaviour."""
from __future__ import annotations

from pathlib import Path

import pytest

from mangavrse.core.cache import CacheManager
from mangavrse.stages.noop import NoOpStage


@pytest.fixture
def cache() -> CacheManager:
    return CacheManager()


class TestCacheManager:
    # -- Key computation -------------------------------------------

    def test_same_inputs_same_key(self, cache: CacheManager, tmp_path: Path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        cfg = {"foo": "bar"}
        assert cache.compute_key(d, cfg) == cache.compute_key(d, cfg)

    def test_different_config_different_key(self, cache: CacheManager, tmp_path: Path):
        d = tmp_path / "in"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        k1 = cache.compute_key(d, {"x": 1})
        k2 = cache.compute_key(d, {"x": 2})
        assert k1 != k2

    def test_different_file_content_different_key(self, cache: CacheManager, tmp_path: Path):
        d = tmp_path / "in"
        d.mkdir()
        f = d / "a.txt"
        f.write_text("v1")
        k1 = cache.compute_key(d, {})
        f.write_text("v2")
        k2 = cache.compute_key(d, {})
        assert k1 != k2

    def test_empty_dir_key_is_stable(self, cache: CacheManager, tmp_path: Path):
        d = tmp_path / "empty"
        d.mkdir()
        k1 = cache.compute_key(d, {})
        k2 = cache.compute_key(d, {})
        assert k1 == k2

    def test_nonexistent_input_dir_key_is_stable(self, cache: CacheManager, tmp_path: Path):
        d = tmp_path / "does_not_exist"
        k1 = cache.compute_key(d, {})
        k2 = cache.compute_key(d, {})
        assert k1 == k2

    # -- Manifest I/O ----------------------------------------------

    def test_no_manifest_means_invalid(self, cache: CacheManager, tmp_path: Path):
        assert not cache.is_valid(tmp_path, "anykey")

    def test_save_then_valid(self, cache: CacheManager, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        cache.save_manifest(
            output_dir=out,
            cache_key="abc123",
            stage_name="test",
            artifacts={},
            config={},
            input_dir=tmp_path,
        )
        assert cache.is_valid(out, "abc123")
        assert not cache.is_valid(out, "wrong_key")

    def test_invalidate(self, cache: CacheManager, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir()
        cache.save_manifest(
            output_dir=out,
            cache_key="k",
            stage_name="t",
            artifacts={},
            config={},
            input_dir=tmp_path,
        )
        assert cache.is_valid(out, "k")
        assert cache.invalidate(out)
        assert not cache.is_valid(out, "k")

    def test_invalidate_missing_is_noop(self, cache: CacheManager, tmp_path: Path):
        assert not cache.invalidate(tmp_path)


# ------------------------------------------------------------------
# Integration: stage caching via execute()
# ------------------------------------------------------------------

class TestStageCaching:
    def test_second_run_is_cached(self, tmp_path: Path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        (inp / "data.txt").write_text("hello")

        stage = NoOpStage()
        r1 = stage.execute(inp, out, {"key": "val"})
        assert r1.success
        assert not r1.cached

        r2 = stage.execute(inp, out, {"key": "val"})
        assert r2.success
        assert r2.cached

    def test_cache_invalidated_by_config_change(self, tmp_path: Path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        (inp / "data.txt").write_text("hello")

        stage = NoOpStage()
        stage.execute(inp, out, {"v": 1})
        r2 = stage.execute(inp, out, {"v": 2})
        assert not r2.cached

    def test_cache_invalidated_by_input_change(self, tmp_path: Path):
        inp = tmp_path / "in"
        out = tmp_path / "out"
        inp.mkdir()
        f = inp / "data.txt"
        f.write_text("v1")

        stage = NoOpStage()
        stage.execute(inp, out, {})

        f.write_text("v2")
        r2 = stage.execute(inp, out, {})
        assert not r2.cached
