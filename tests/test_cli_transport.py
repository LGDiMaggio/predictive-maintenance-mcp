"""
Tests for CLI argument parsing and transport configuration.

Covers:
- argparse defaults (transport=stdio, host=127.0.0.1, port=8000)
- Explicit flag values (--transport sse, --host, --port)
- Short flag aliases (-t, -p)
- Environment variable fallbacks (MCP_TRANSPORT, MCP_HOST, MCP_PORT)
- Invalid transport choice → error
- Invalid port (non-numeric) → error
- CLI args override env vars
"""

import os
import pytest
from unittest.mock import patch, MagicMock


def _build_parser(env_transport="stdio", env_host="127.0.0.1", env_port="8000"):
    """Build the REAL parser from src/server.py under the given environment.

    This used to re-implement main()'s argparse setup inline. A copy cannot
    catch drift: changing the real default to 0.0.0.0 would leave every
    assertion here green. Delegating means these tests now fail when the
    shipped parser changes.
    """
    from predictive_maintenance_mcp.server import build_parser

    with patch.dict(
        os.environ,
        {
            "MCP_TRANSPORT": env_transport,
            "MCP_HOST": env_host,
            "MCP_PORT": str(env_port),
        },
    ):
        return build_parser()


# ── Default values ─────────────────────────────────────────────────────────

class TestCLIDefaults:

    def test_default_transport_stdio(self):
        args = _build_parser().parse_args([])
        assert args.transport == "stdio"

    def test_default_host(self):
        args = _build_parser().parse_args([])
        assert args.host == "127.0.0.1"

    def test_default_port(self):
        args = _build_parser().parse_args([])
        assert args.port == 8000


# ── Explicit flags ─────────────────────────────────────────────────────────

class TestCLIExplicitFlags:

    def test_transport_sse(self):
        args = _build_parser().parse_args(["--transport", "sse"])
        assert args.transport == "sse"

    def test_transport_streamable_http(self):
        args = _build_parser().parse_args(["--transport", "streamable-http"])
        assert args.transport == "streamable-http"

    def test_host_custom(self):
        args = _build_parser().parse_args(["--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_port_custom(self):
        args = _build_parser().parse_args(["--port", "9090"])
        assert args.port == 9090

    def test_short_flag_transport(self):
        args = _build_parser().parse_args(["-t", "sse"])
        assert args.transport == "sse"

    def test_short_flag_port(self):
        args = _build_parser().parse_args(["-p", "7777"])
        assert args.port == 7777

    def test_all_flags_combined(self):
        args = _build_parser().parse_args([
            "-t", "sse", "--host", "10.0.0.1", "-p", "3000"
        ])
        assert args.transport == "sse"
        assert args.host == "10.0.0.1"
        assert args.port == 3000


# ── Invalid arguments ──────────────────────────────────────────────────────

class TestCLIInvalidArgs:

    def test_invalid_transport_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--transport", "websocket"])

    def test_non_numeric_port_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--port", "abc"])


# ── Environment variable fallbacks ─────────────────────────────────────────

class TestCLIEnvVars:

    def test_mcp_transport_env(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
        env_t = os.environ.get("MCP_TRANSPORT", "stdio")
        args = _build_parser(env_transport=env_t).parse_args([])
        assert args.transport == "sse"

    def test_mcp_host_env(self, monkeypatch):
        monkeypatch.setenv("MCP_HOST", "192.168.1.100")
        env_h = os.environ.get("MCP_HOST", "127.0.0.1")
        args = _build_parser(env_host=env_h).parse_args([])
        assert args.host == "192.168.1.100"

    def test_mcp_port_env(self, monkeypatch):
        monkeypatch.setenv("MCP_PORT", "5555")
        env_p = os.environ.get("MCP_PORT", "8000")
        args = _build_parser(env_port=env_p).parse_args([])
        assert args.port == 5555

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "sse")
        monkeypatch.setenv("MCP_PORT", "5555")
        env_t = os.environ.get("MCP_TRANSPORT", "stdio")
        env_p = os.environ.get("MCP_PORT", "8000")
        args = _build_parser(env_transport=env_t, env_port=env_p).parse_args(
            ["--transport", "streamable-http", "--port", "9999"]
        )
        assert args.transport == "streamable-http"
        assert args.port == 9999


# ── main() integration (smoke test) ───────────────────────────────────────

def test_main_help_exits_cleanly():
    """Verify --help works without crashing (exercises the real main parser)."""
    import sys
    from unittest.mock import patch as _patch

    with _patch.object(sys, 'argv', ['test', '--help']):
        with pytest.raises(SystemExit) as exc_info:
            from predictive_maintenance_mcp.server import main
            main()
        assert exc_info.value.code == 0
