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


class TestGetLedgerDir:
    """get_ledger_dir() reads PMM_LEDGER_DIR at EVERY call (never frozen at
    import, unlike the path constants) and creates nothing.

    Uses the same top-level ``config`` alias as the classes above (the same
    source file the package imports as ``predictive_maintenance_mcp.config``).
    """

    def test_env_var_is_read_at_each_call(self, tmp_path, monkeypatch):
        import config

        monkeypatch.setenv("PMM_LEDGER_DIR", str(tmp_path / "one"))
        assert config.get_ledger_dir() == tmp_path / "one"
        monkeypatch.setenv("PMM_LEDGER_DIR", str(tmp_path / "two"))
        assert config.get_ledger_dir() == tmp_path / "two"

    def test_unset_defaults_to_data_ledger_under_project_root(self, monkeypatch):
        import config

        monkeypatch.delenv("PMM_LEDGER_DIR", raising=False)
        assert config.get_ledger_dir() == config.PROJECT_ROOT / "data" / "ledger"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_counts_as_unset(self, monkeypatch, blank):
        import config

        monkeypatch.setenv("PMM_LEDGER_DIR", blank)
        assert config.get_ledger_dir() == config.PROJECT_ROOT / "data" / "ledger"

    def test_import_creates_no_directory(self, tmp_path, monkeypatch):
        """No side effect at import: the ledger store creates the directory on
        first append, the server at startup."""
        import importlib
        import config

        target = tmp_path / "fresh_ledger"
        monkeypatch.setenv("PMM_LEDGER_DIR", str(target))
        importlib.reload(config)
        assert config.get_ledger_dir() == target
        assert not target.exists()
        importlib.reload(config)
