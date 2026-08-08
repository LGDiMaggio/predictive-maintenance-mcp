"""
Tests for src/server.py — MCP server orchestrator.

Covers:
- _setup_environment() directory creation
- MCPServer instance creation
- register_all() integration
- main() argument parsing
"""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSetupEnvironment:
    """Tests for _setup_environment()."""

    def test_creates_all_required_directories(self, tmp_path, monkeypatch):
        """_setup_environment must create DATA_DIR, MODELS_DIR, REPORTS_DIR,
        RESOURCES_DIR (with sub-dirs), and CACHE_DIR."""
        data_dir = tmp_path / "data" / "signals"
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        resources_dir = tmp_path / "resources"
        cache_dir = resources_dir / "cache"

        import predictive_maintenance_mcp.server as srv

        monkeypatch.setattr(srv, "DATA_DIR", data_dir)
        monkeypatch.setattr(srv, "MODELS_DIR", models_dir)
        monkeypatch.setattr(srv, "REPORTS_DIR", reports_dir)
        monkeypatch.setattr(srv, "RESOURCES_DIR", resources_dir)
        monkeypatch.setattr(srv, "CACHE_DIR", cache_dir)

        srv._setup_environment()

        assert data_dir.is_dir()
        assert models_dir.is_dir()
        assert reports_dir.is_dir()
        assert resources_dir.is_dir()
        assert cache_dir.is_dir()
        assert (resources_dir / "machine_manuals").is_dir()
        assert (resources_dir / "bearing_catalogs").is_dir()
        assert (resources_dir / "datasheets").is_dir()

    def test_idempotent_when_dirs_exist(self, tmp_path, monkeypatch):
        """Calling _setup_environment twice should not raise."""
        data_dir = tmp_path / "data" / "signals"
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports"
        resources_dir = tmp_path / "resources"
        cache_dir = resources_dir / "cache"

        import predictive_maintenance_mcp.server as srv

        monkeypatch.setattr(srv, "DATA_DIR", data_dir)
        monkeypatch.setattr(srv, "MODELS_DIR", models_dir)
        monkeypatch.setattr(srv, "REPORTS_DIR", reports_dir)
        monkeypatch.setattr(srv, "RESOURCES_DIR", resources_dir)
        monkeypatch.setattr(srv, "CACHE_DIR", cache_dir)

        srv._setup_environment()
        srv._setup_environment()  # second call — no error expected

        assert data_dir.is_dir()


class TestMCPInstance:
    """Tests for the module-level `mcp` MCPServer object."""

    def test_mcp_is_mcpserver_instance(self):
        from mcp.server.mcpserver import MCPServer
        import predictive_maintenance_mcp.server as srv

        assert isinstance(srv.mcp, MCPServer)

    def test_mcp_has_expected_name(self):
        import predictive_maintenance_mcp.server as srv

        # MCPServer stores the name; check it matches what we passed
        assert srv.mcp.name == "Predictive Maintenance"


class TestRegisterAll:
    """Tests for register_all()."""

    def test_register_all_no_exception(self):
        """register_all() should complete without raising on a fresh MCPServer."""
        from mcp.server.mcpserver import MCPServer
        from predictive_maintenance_mcp.mcp_tools import register_all

        fresh_mcp = MCPServer("test-server")
        # Should not raise
        register_all(fresh_mcp)


class TestMainArgParser:
    """Tests for main() CLI argument parsing."""

    def test_default_args_stdio(self, monkeypatch):
        """With no CLI args, the transport is stdio and carries no bind kwargs.

        mcp 2.x routes run(transport='stdio') to run_stdio_async(), which
        accepts no host/port — passing them would raise TypeError, so their
        absence is the contract worth pinning here.
        """
        import predictive_maintenance_mcp.server as srv

        # Prevent mcp.run from actually starting the server
        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)

        # Simulate empty CLI args
        monkeypatch.setattr(sys, "argv", ["server"])

        # Patch _setup_environment to avoid side effects
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)

        # Clear env vars that could override defaults
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)

        srv.main()

        mock_run.assert_called_once_with(transport="stdio")

    def test_sse_transport_args(self, monkeypatch):
        """Passing --transport sse --host 0.0.0.0 --port 9090."""
        import predictive_maintenance_mcp.server as srv

        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)

        monkeypatch.setattr(
            sys, "argv",
            ["server", "--transport", "sse", "--host", "0.0.0.0", "--port", "9090"],
        )

        srv.main()

        mock_run.assert_called_once_with(
            transport="sse", host="0.0.0.0", port=9090
        )

    def test_env_var_overrides(self, monkeypatch):
        """Environment variables MCP_TRANSPORT / MCP_HOST / MCP_PORT override defaults."""
        import predictive_maintenance_mcp.server as srv

        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)

        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("MCP_HOST", "10.0.0.1")
        monkeypatch.setenv("MCP_PORT", "7777")

        monkeypatch.setattr(sys, "argv", ["server"])

        srv.main()

        mock_run.assert_called_once_with(
            transport="streamable-http", host="10.0.0.1", port=7777
        )
