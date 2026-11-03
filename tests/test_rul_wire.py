"""estimate_rul as an MCP client sees it.

- Raw JSON-RPC: a client can send a literal ``NaN`` or ``Infinity`` token,
  which the SDK's Python client never produces. It must reach the tool and be
  refused with the tool's own text, not computed on. The stdio harness mirrors
  ``tests/test_forecast_wire.py``.
- A model-implied threshold crossing travels as structured content that
  validates against the advertised output model.
"""

import json

import anyio
import pytest
from mcp import Client, types
from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server

from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.models import RULEstimationResult


@pytest.fixture(scope="module")
def mcp():
    server = MCPServer("test-rul-wire")
    register_all(server)
    return server


def _call_lines(arguments_json: str) -> str:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-rul-wire", "version": "0"},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = (
        '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": '
        '{"name": "estimate_rul", "arguments": ' + arguments_json + "}}"
    )
    return "\n".join([json.dumps(initialize), json.dumps(initialized), call]) + "\n"


class _Stdout:
    def __init__(self) -> None:
        self.replies: list[dict] = []
        self.call_replied = anyio.Event()

    async def write(self, text: str) -> None:
        for line in text.splitlines():
            if line:
                reply = json.loads(line)
                self.replies.append(reply)
                if reply.get("id") == 2:
                    self.call_replied.set()

    async def flush(self) -> None:
        pass


class _Stdin:
    """Yields the request lines, then stays open until the call is answered."""

    def __init__(self, lines: str, until: anyio.Event) -> None:
        self._lines = lines.splitlines(keepends=True)
        self._until = until

    def __aiter__(self):
        return self._read()

    async def _read(self):
        for line in self._lines:
            yield line
        await self._until.wait()


async def _serve(server: MCPServer, lines: str) -> dict:
    stdout = _Stdout()
    with anyio.fail_after(60):
        async with stdio_server(_Stdin(lines, stdout.call_replied), stdout) as (
            read,
            write,
        ):
            await server._lowlevel_server.run(
                read, write, server._lowlevel_server.create_initialization_options()
            )
    return next(r for r in stdout.replies if r.get("id") == 2)


def _arguments(values: str = "1.0, 1.2, 1.4, 1.6", threshold: str = "4.5") -> str:
    return (
        '{"failure_threshold": ' + threshold + ', "feature_values": [' + values + "], "
        '"timestamps": [0, 24, 48, 72]}'
    )


class TestOverTheWire:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "arguments, phrase",
        [
            (_arguments(values="1.0, NaN, 1.4, 1.6"), "first at sample index 1"),
            (_arguments(values="1.0, 1.2, 1.4, Infinity"), "first at sample index 3"),
            (_arguments(threshold="NaN"), "failure_threshold is nan, not a finite"),
        ],
    )
    async def test_literal_token_is_refused_by_the_tool(self, mcp, arguments, phrase):
        response = await _serve(mcp, _call_lines(arguments))
        result = response["result"]
        text = " ".join(c["text"] for c in result["content"] if c["type"] == "text")
        assert result["isError"] is True
        assert phrase in text

    @pytest.mark.asyncio
    async def test_the_same_request_with_measured_values_succeeds(self, mcp):
        response = await _serve(mcp, _call_lines(_arguments()))
        assert response["result"].get("isError") is not True

    @pytest.mark.asyncio
    async def test_model_crossing_is_valid_structured_content(self, mcp):
        async with Client(mcp, mode="legacy") as client:
            result = await client.call_tool(
                "estimate_rul",
                {
                    "failure_threshold": 10.0,
                    "feature_values": [6.0, 7.1, 8.0, 9.2, 10.4, 9.7],
                    "timestamps": [0, 1, 2, 3, 4, 5],
                },
            )
        assert not result.is_error, result.content[0].text
        model = RULEstimationResult.model_validate(result.structured_content)
        assert model.status == "model_threshold_crossed"
        assert model.current_value < model.failure_threshold
        assert model.modelled_current_value >= model.failure_threshold
