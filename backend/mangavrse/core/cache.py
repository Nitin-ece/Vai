"""
Per-stage caching based on SHA-256 of (input files + config).

The cache manifest is stored as `.cache_manifest.json` in the stage's
output directory.  It records enough information to decide on a cache
hit *and* to audit what produced the outputs.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_MANIFEST_NAME = ".cache_manifest.json"


class CacheManager:
    """Filesystem-based, per-stage output cache."""

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------
    @staticmethod
    def _hash_file(path: Path, *, block_size: int = 1 << 16) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(block_size):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _hash_config(config: dict[str, Any]) -> str:
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def compute_key(self, input_dir: Path, config: dict[str, Any]) -> str:
        """SHA-256( sorted input-file hashes ∥ config hash )."""
        file_hashes: list[str] = []
        if input_dir.exists():
            for root, _dirs, files in os.walk(input_dir):
                for fname in sorted(files):
                    fp = Path(root) / fname
                    file_hashes.append(f"{fp.relative_to(input_dir)}:{self._hash_file(fp)}")

        combined = "\n".join(sorted(file_hashes)) + "\n" + self._hash_config(config)
        return hashlib.sha256(combined.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------
    def _manifest_path(self, output_dir: Path) -> Path:
        return output_dir / _MANIFEST_NAME

    def is_valid(self, output_dir: Path, expected_key: str) -> bool:
        """Return True when *output_dir* has a manifest matching *expected_key*."""
        mp = self._manifest_path(output_dir)
        if not mp.exists():
            return False
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            return manifest.get("cache_key") == expected_key
        except (json.JSONDecodeError, OSError):
            return False

    def save_manifest(
        self,
        *,
        output_dir: Path,
        cache_key: str,
        stage_name: str,
        artifacts: dict[str, Path],
        config: dict[str, Any],
        input_dir: Path,
    ) -> Path:
        manifest = {
            "stage": stage_name,
            "cache_key": cache_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": self._hash_config(config),
            "input_dir": str(input_dir),
            "artifacts": {k: str(v) for k, v in artifacts.items()},
        }
        mp = self._manifest_path(output_dir)
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return mp

    def load_artifacts(self, output_dir: Path) -> dict[str, Path]:
        """Read artifact paths from an existing manifest."""
        mp = self._manifest_path(output_dir)
        if not mp.exists():
            return {}
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        return {k: Path(v) for k, v in manifest.get("artifacts", {}).items()}

    def invalidate(self, output_dir: Path) -> bool:
        """Delete the cache manifest (forces re-run)."""
        mp = self._manifest_path(output_dir)
        if mp.exists():
            mp.unlink()
            return True
        return False
