"""Every registered tool that takes a shaft speed refuses an invalid one.

Issue #83: a zero, negative, NaN or infinite ``rpm`` produced impossible
fault frequencies (a bearing reported healthy, or a false fault), picked an
ISO band silently or dropped harmonic labels without a word. These tests pin
the refusal across the whole registered surface at once:

- a matrix over every registered tool with an ``rpm`` parameter, called
  through the refusal boundary, so the refusal is a ``ToolError`` whose text
  the model receives;
- a guard bound to the registry: the set of tools with an ``rpm`` parameter
  must equal the matrix's set, so a new tool with a shaft speed cannot ship
  without being added here (and therefore guarded);
- a wire-level case: MCP clients can send a literal ``NaN`` token, which the
  SDK's JSON parser accepts as a float. The Python client cannot carry it (it
  serializes NaN to ``null``), so raw JSON-RPC lines go through the same
  stdio transport the server runs on.
"""

import inspect
import io
import json
import math
from unittest.mock import AsyncMock

import anyio
import numpy as np
import pandas as pd
import pytest
from mcp import types
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.stdio import stdio_server

from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

RPM_PHRASE = "rpm must be positive and finite"

#: Every registered tool with an ``rpm`` parameter, with the other arguments
#: that reach its speed check. The check runs before any signal lookup, so an
#: unloaded signal id is enough; a refusal for that id would fail the phrase.
RPM_TOOLS = {
    "assess_severity": {"signal_id": "__not_loaded__"},
    "calculate_bearing_characteristic_frequencies": {
        "num_balls": 9,
        "ball_diameter_mm": 7.94,
        "pitch_diameter_mm": 39.04,
    },
    "check_bearing_faults": {"signal_id": "__not_loaded__", "bearing_id": "6205"},
    "diagnose_vibration": {"signal_id": "__not_loaded__"},
    "generate_diagnostic_report": {"signal_id": "__not_loaded__"},
    "generate_fft_report": {"signal_id": "__not_loaded__"},
    "generate_iso_report": {"signal_id": "__not_loaded__"},
}

BAD_SPEEDS = [0.0, -1.0, math.nan, math.inf, -math.inf]
BAD_SPEED_IDS = ["zero", "negative", "nan", "inf", "-inf"]


def _tools_with_rpm(server: MCPServer) -> set[str]:
    """Names of the registered tools that take a parameter named ``rpm``."""
    return {
        tool.name
        for tool in server._tool_manager._tools.values()
        if "rpm" in inspect.signature(inspect.unwrap(tool.fn)).parameters
    }


@pytest.fixture(scope="module")
def mcp():
    server = MCPServer("test-rpm-refusals")
    register_all(server)
    return server


class TestRegistryGuard:
    def test_every_tool_with_an_rpm_parameter_is_in_the_matrix(self, mcp):
        assert _tools_with_rpm(mcp) == set(RPM_TOOLS), (
            "A registered tool takes rpm but is not in RPM_TOOLS (or the "
            "reverse): add it there so its speed refusal is tested."
        )

    def test_guard_fails_for_an_unlisted_tool_with_rpm(self):
        """The guard can fail: a new tool taking rpm makes the sets differ."""
        server = MCPServer("test-rpm-guard")
        register_all(server)

        async def spin_test(rpm: float) -> str:
            return "ok"

        server.add_tool(spin_test)

        assert _tools_with_rpm(server) != set(RPM_TOOLS)
        assert "spin_test" in _tools_with_rpm(server) - set(RPM_TOOLS)


class TestRefusalMatrix:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpm", BAD_SPEEDS, ids=BAD_SPEED_IDS)
    @pytest.mark.parametrize("tool_name", sorted(RPM_TOOLS))
    async def test_invalid_speed_is_refused(
        self, mcp, sandbox_data_dir, tool_name, rpm
    ):
        tool = mcp._tool_manager._tools[tool_name]
        kwargs = {**RPM_TOOLS[tool_name], "rpm": rpm}
        if "ctx" in inspect.signature(tool.fn).parameters:
            kwargs["ctx"] = AsyncMock()

        with pytest.raises(ToolError) as exc_info:
            result = tool.fn(**kwargs)
            if inspect.isawaitable(result):
                await result

        assert isinstance(exc_info.value, ValueError)
        assert RPM_PHRASE in str(exc_info.value)


def _jsonrpc_lines(arguments_json: str) -> str:
    """An initialize handshake, then one tools/call with raw argument JSON."""
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-rpm-wire", "version": "0"},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = (
        '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": '
        '{"name": "check_bearing_faults", "arguments": ' + arguments_json + "}}"
    )
    return "\n".join([json.dumps(initialize), json.dumps(initialized), call]) + "\n"


async def _serve(server: MCPServer, lines: str) -> dict:
    """Run the server over in-memory stdio and return the tools/call response."""
    out = io.StringIO()
    with anyio.fail_after(60):
        async with stdio_server(
            anyio.wrap_file(io.StringIO(lines)), anyio.wrap_file(out)
        ) as (read, write):
            await server._lowlevel_server.run(
                read, write, server._lowlevel_server.create_initialization_options()
            )
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line]
    return next(r for r in responses if r.get("id") == 2)


class TestOverTheWire:
    @pytest.fixture
    def loaded_signal(self, sandbox_data_dir):
        fs = 10000
        signal = np.sin(2 * np.pi * 50 * np.arange(fs) / fs)
        pd.DataFrame(signal).to_csv(
            sandbox_data_dir / "rpm_wire.csv", index=False, header=False
        )
        repo = get_repository()
        repo.load_signal(
            "rpm_wire.csv", sampling_rate=fs, signal_unit="g", overwrite=True
        )
        yield "rpm_wire"
        repo.clear_signal("rpm_wire")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "0"])
    async def test_literal_token_is_refused_by_the_tool(
        self, mcp, loaded_signal, token
    ):
        arguments = (
            '{"signal_id": "' + loaded_signal + '", "bearing_id": "6205", '
            '"rpm": ' + token + "}"
        )
        response = await _serve(mcp, _jsonrpc_lines(arguments))

        result = response["result"]
        text = " ".join(c["text"] for c in result["content"] if c["type"] == "text")
        assert result["isError"] is True
        assert RPM_PHRASE in text

    @pytest.mark.asyncio
    async def test_the_same_request_with_a_measured_speed_succeeds(
        self, mcp, loaded_signal
    ):
        arguments = (
            '{"signal_id": "' + loaded_signal + '", "bearing_id": "6205", '
            '"rpm": 1797}'
        )
        response = await _serve(mcp, _jsonrpc_lines(arguments))

        assert response["result"].get("isError") is not True
