"""
Tests for src/config.py — centralized path configuration.

Covers:
- resolve_project_root() with different environments
- Path constants are valid Path objects
- Environment variable overrides (PDM_PROJECT_DIR)
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestResolveProjectRoot:
    """Tests for resolve_project_root()."""

    def test_env_var_override(self, tmp_path, monkeypatch):
        """PDM_PROJECT_DIR env var takes highest priority."""
        monkeypatch.setenv("PDM_PROJECT_DIR", str(tmp_path))

        # Re-import to trigger fresh resolution
        import importlib
        import config

        importlib.reload(config)
        result = config.resolve_project_root()

        assert result == tmp_path

        # Clean up: remove env var and reload to restore defaults
        monkeypatch.delenv("PDM_PROJECT_DIR", raising=False)
        importlib.reload(config)

    def test_cwd_with_data_signals(self, tmp_path, monkeypatch):
        """If CWD contains data/signals/, use CWD as root."""
        monkeypatch.delenv("PDM_PROJECT_DIR", raising=False)
        (tmp_path / "data" / "signals").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        import importlib
        import config

        importlib.reload(config)
        result = config.resolve_project_root()

        assert result == tmp_path
        importlib.reload(config)

    def test_fallback_to_cwd(self, tmp_path, monkeypatch):
        """Without env var or data/signals in CWD or file-based, falls back to CWD."""
        monkeypatch.delenv("PDM_PROJECT_DIR", raising=False)
        # Use a temp dir that has no data/signals
        empty = tmp_path / "empty_project"
        empty.mkdir()
        monkeypatch.chdir(empty)

        import importlib
        import config

        # Also patch __file__-based lookup so it doesn't find the real repo
        original_file = config.__file__
        fake_file = str(empty / "src" / "config.py")
        monkeypatch.setattr(config, "__file__", fake_file)

        result = config.resolve_project_root()
        # Should fall back to CWD since neither env var, CWD/data/signals,
        # nor file-based/data/signals exist
        assert result == empty

        monkeypatch.setattr(config, "__file__", original_file)
        importlib.reload(config)


class TestPathConstants:
    """Tests for module-level path constants."""

    def test_all_constants_are_path_instances(self):
        import config

        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.MODELS_DIR, Path)
        assert isinstance(config.REPORTS_DIR, Path)
        assert isinstance(config.RESOURCES_DIR, Path)
        assert isinstance(config.CACHE_DIR, Path)

    def test_data_dir_is_under_project_root(self):
        import config

        assert str(config.DATA_DIR).startswith(str(config.PROJECT_ROOT))

    def test_cache_dir_is_under_resources(self):
        import config

        assert str(config.CACHE_DIR).startswith(str(config.RESOURCES_DIR))
