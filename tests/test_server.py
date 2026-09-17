"""
Tests for src/server.py — MCP server orchestrator.

Covers:
- _setup_environment() directory creation
- MCPServer instance creation
- register_all() integration
- main() argument parsing
"""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sandbox_server(tmp_path, monkeypatch):
    """The server module with every directory _setup_environment creates
    redirected under tmp_path (the ledger directory follows PMM_LEDGER_DIR,
    which conftest already points at tmp_path for every test)."""
    import predictive_maintenance_mcp.server as srv

    resources_dir = tmp_path / "resources"
    monkeypatch.setattr(srv, "DATA_DIR", tmp_path / "data" / "signals")
    monkeypatch.setattr(srv, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(srv, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(srv, "RESOURCES_DIR", resources_dir)
    monkeypatch.setattr(srv, "CACHE_DIR", resources_dir / "cache")
    return srv


def _ledger_warnings(caplog) -> list[str]:
    """Warning-or-worse records about the asset ledger directory."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "ledger" in record.getMessage().lower()
    ]


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


class TestLedgerDirectorySetup:
    """_setup_environment creates the asset ledger directory and warns, once
    at startup, about the two ways it can silently fail later: not writable,
    or under a cloud-synced folder. Warnings only: the server still starts."""

    def _silence_sync_env(self, srv, monkeypatch):
        for var in srv.SYNC_ROOT_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_creates_ledger_dir_from_env(self, sandbox_server, tmp_path, monkeypatch):
        target = tmp_path / "custom" / "ledger"
        monkeypatch.setenv("PMM_LEDGER_DIR", str(target))
        sandbox_server._setup_environment()
        assert target.is_dir()
        sandbox_server._setup_environment()  # idempotent
        assert target.is_dir()

    def test_local_unsynced_dir_is_silent(
        self, sandbox_server, tmp_path, monkeypatch, package_caplog
    ):
        self._silence_sync_env(sandbox_server, monkeypatch)
        monkeypatch.setenv("PMM_LEDGER_DIR", str(tmp_path / "ledger"))
        sandbox_server._setup_environment()
        assert _ledger_warnings(package_caplog) == []

    def test_warns_when_dir_cannot_be_created(
        self, sandbox_server, tmp_path, monkeypatch, package_caplog
    ):
        """A regular file where the parent should be: mkdir fails, the server
        still starts, and the warning names the remedy."""
        self._silence_sync_env(sandbox_server, monkeypatch)
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory", encoding="utf-8")
        monkeypatch.setenv("PMM_LEDGER_DIR", str(blocker / "ledger"))

        sandbox_server._setup_environment()  # must not raise

        warnings = _ledger_warnings(package_caplog)
        assert len(warnings) == 1
        assert "cannot be created" in warnings[0]
        assert "PMM_LEDGER_DIR" in warnings[0]

    def test_warns_when_dir_is_not_writable(
        self, sandbox_server, tmp_path, monkeypatch, package_caplog
    ):
        self._silence_sync_env(sandbox_server, monkeypatch)
        monkeypatch.setenv("PMM_LEDGER_DIR", str(tmp_path / "ledger"))
        monkeypatch.setattr(sandbox_server, "_ledger_dir_writable", lambda path: False)

        sandbox_server._setup_environment()

        warnings = _ledger_warnings(package_caplog)
        assert len(warnings) == 1
        assert "not writable" in warnings[0]
        assert "PMM_LEDGER_DIR" in warnings[0]

    def test_write_probe_is_a_real_write(self, sandbox_server, tmp_path):
        """The probe writes and removes a file: a regular file in place of the
        directory fails it, a writable directory passes it and is left clean."""
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        assert sandbox_server._ledger_dir_writable(blocker) is False
        assert sandbox_server._ledger_dir_writable(tmp_path) is True
        assert sorted(p.name for p in tmp_path.iterdir()) == ["file"]

    def test_warns_under_onedrive_env_root(
        self, sandbox_server, tmp_path, monkeypatch, package_caplog
    ):
        """PMM_LEDGER_DIR under the folder named by the OneDrive variable."""
        self._silence_sync_env(sandbox_server, monkeypatch)
        monkeypatch.setenv("OneDrive", str(tmp_path))
        monkeypatch.setenv("PMM_LEDGER_DIR", str(tmp_path / "data" / "ledger"))

        sandbox_server._setup_environment()

        warnings = _ledger_warnings(package_caplog)
        assert len(warnings) == 1
        assert "OneDrive" in warnings[0]
        assert "PMM_LEDGER_DIR" in warnings[0]

    @pytest.mark.parametrize("marker", ["Dropbox", "iCloud Drive"])
    def test_warns_under_named_sync_ancestor(
        self, sandbox_server, tmp_path, monkeypatch, package_caplog, marker
    ):
        self._silence_sync_env(sandbox_server, monkeypatch)
        target = tmp_path / marker / "project" / "data" / "ledger"
        monkeypatch.setenv("PMM_LEDGER_DIR", str(target))

        sandbox_server._setup_environment()

        warnings = _ledger_warnings(package_caplog)
        assert len(warnings) == 1
        assert marker in warnings[0]
        assert "PMM_LEDGER_DIR" in warnings[0]

    def test_find_sync_root_is_none_outside_known_roots(
        self, sandbox_server, tmp_path, monkeypatch
    ):
        self._silence_sync_env(sandbox_server, monkeypatch)
        assert sandbox_server.find_sync_root(tmp_path / "plain" / "ledger") is None
        monkeypatch.setenv("OneDriveCommercial", str(tmp_path / "cloud"))
        assert sandbox_server.find_sync_root(tmp_path / "plain" / "ledger") is None
        found = sandbox_server.find_sync_root(tmp_path / "cloud" / "ledger")
        assert found is not None and found.startswith("OneDriveCommercial=")


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

    def test_sse_defaults_bind_to_loopback(self, monkeypatch):
        """The assertion that would catch a default widening to 0.0.0.0.

        The other transport tests all pass --host/--port explicitly, so none
        of them exercises the argparse defaults on the branch that actually
        opens a socket. This is the one that pins the local-processing
        invariant at the bind boundary.
        """
        import predictive_maintenance_mcp.server as srv

        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)
        monkeypatch.delenv("MCP_HOST", raising=False)
        monkeypatch.delenv("MCP_PORT", raising=False)
        monkeypatch.setattr(sys, "argv", ["server", "--transport", "sse"])

        srv.main()

        mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)

    def test_blank_env_host_does_not_become_a_wildcard_bind(self, monkeypatch):
        """MCP_HOST set-but-empty must not resolve to INADDR_ANY.

        os.environ.get(name, fallback) returns "" for a variable that exists
        with an empty value — trivially produced by blanking it in a compose
        file — and "" binds to every interface. An operator doing that to
        *undo* a wildcard bind would get the opposite.
        """
        import predictive_maintenance_mcp.server as srv

        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)
        monkeypatch.setenv("MCP_HOST", "")
        monkeypatch.delenv("MCP_PORT", raising=False)
        monkeypatch.setattr(sys, "argv", ["server", "--transport", "sse"])

        srv.main()

        mock_run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)

    def test_invalid_env_transport_exits_instead_of_reaching_run(self, monkeypatch):
        """argparse validates `choices` only for command-line values.

        The env var is the configuration channel in every container, so an
        unvalidated default reaches run() and crash-loops behind a log line
        claiming the server is listening.
        """
        import predictive_maintenance_mcp.server as srv

        mock_run = MagicMock()
        monkeypatch.setattr(srv.mcp, "run", mock_run)
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)
        monkeypatch.setenv("MCP_TRANSPORT", "http")
        monkeypatch.setattr(sys, "argv", ["server"])

        with pytest.raises(SystemExit):
            srv.main()
        mock_run.assert_not_called()

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
            sys,
            "argv",
            ["server", "--transport", "sse", "--host", "0.0.0.0", "--port", "9090"],
        )

        srv.main()

        mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=9090)

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
