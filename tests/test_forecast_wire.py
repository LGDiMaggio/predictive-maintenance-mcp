"""forecast_asset_indicator as an MCP client sees it (forecast PR 1, U4).

- Raw JSON-RPC: a client can send a literal ``NaN`` token, which the SDK's
  Python client never produces (it serialises NaN to ``null``). Both must
  reach the tool and be refused with its own text. The stdio harness mirrors
  ``tests/test_rpm_refusals.py``.
- Byte budget: the ``CallToolResult`` the client receives never exceeds
  ``MAX_RESULT_BYTES`` and echoes no input series. The tool measures each
  result the way the SDK builds it, in both encodings the SDK writes, and
  refuses one above the budget, naming a shorter horizon whose result it has
  measured to fit.
"""

import json
import re
from datetime import datetime, timedelta, timezone

import anyio
import httpx2
import numpy as np
import pytest
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.server.stdio import stdio_server

from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.mcp_tools.prognostics_tools import (
    RESULT_METADATA_ALLOWANCE,
    RETRY_RESERVE,
    call_tool_result_bytes,
    client_result_bytes,
    forecast_asset_indicator,
)
from predictive_maintenance_mcp.models import IndicatorForecastResult
from predictive_maintenance_mcp.prognostics import indicator_forecast
from predictive_maintenance_mcp.prognostics.indicator_forecast import MAX_RESULT_BYTES
from predictive_maintenance_mcp.prognostics.indicator_series import (
    MAX_HORIZON_STEPS,
    MAX_INPUT_POINTS,
)
from predictive_maintenance_mcp.signal_acquisition.measurement import (
    MAX_FREE_TEXT_CHARS,
)
from predictive_maintenance_mcp.signal_validation import InvalidInputError


@pytest.fixture(scope="module")
def mcp():
    server = MCPServer("test-forecast-wire")
    register_all(server)
    return server


def _series(n, step=timedelta(minutes=10), level=3.0, spread=0.05):
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0) - step
    rng = np.random.default_rng(4)
    instants = [
        (end - (n - 1 - i) * step).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)
    ]
    values = [
        float(v) for v in np.round(np.cumsum(rng.normal(0, spread, n)) + level, 9)
    ]
    return instants, values


def _call_lines(arguments_json: str) -> str:
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": types.LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test-forecast-wire", "version": "0"},
        },
    }
    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    call = (
        '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": '
        '{"name": "forecast_asset_indicator", "arguments": ' + arguments_json + "}}"
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


def _arguments_with_token(token: str) -> str:
    instants, values = _series(30)
    rendered = [repr(v) for v in values]
    rendered[5] = token
    return (
        '{"observed_at": '
        + json.dumps(instants)
        + ', "values": ['
        + ", ".join(rendered)
        + "], "
        '"indicator": "velocity RMS", "unit": "mm/s", "resolution": "10min", "horizon_steps": 3}'
    )


class TestOverTheWire:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "token, phrase",
        [
            ("NaN", "sample index 5"),
            ("Infinity", "sample index 5"),
            ("null", "values[5] is null"),
        ],
    )
    async def test_literal_token_is_refused_by_the_tool(self, mcp, token, phrase):
        response = await _serve(mcp, _call_lines(_arguments_with_token(token)))
        result = response["result"]
        text = " ".join(c["text"] for c in result["content"] if c["type"] == "text")
        assert result["isError"] is True
        assert phrase in text

    @pytest.mark.asyncio
    async def test_the_same_request_with_a_measured_value_succeeds(self, mcp):
        response = await _serve(mcp, _call_lines(_arguments_with_token("3.0")))
        assert response["result"].get("isError") is not True


class TestResultSize:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level, spread",
        # The second series has a band so narrow against its level (a ratio
        # near 8e6, just inside the 1e7 the output can carry) that the
        # quantiles need the most significant digits the output allows.
        [(3.0, 0.05), (123456.7, 0.012)],
    )
    async def test_result_at_the_caps_stays_within_the_budget(self, mcp, level, spread):
        instants, values = _series(MAX_INPUT_POINTS, level=level, spread=spread)
        async with Client(mcp, mode="legacy") as client:
            result = await client.call_tool(
                "forecast_asset_indicator",
                {
                    "observed_at": instants,
                    "values": values,
                    "indicator": "velocity RMS",
                    "unit": "mm/s",
                    "resolution": "10min",
                    "horizon_steps": MAX_HORIZON_STEPS,
                },
                read_timeout_seconds=30,
            )
        assert not result.is_error
        wire = result.model_dump_json(by_alias=True, exclude_none=True)
        size = len(wire.encode("utf-8"))
        assert size <= MAX_RESULT_BYTES, size
        # No input series is echoed: only the history block's first and last instants.
        assert sum(wire.count(t) for t in instants[1:-1]) == 0


#: Value shapes whose quantiles print long: a sign, an exponent and the most
#: digits the band needs. Unrounded, so the noise survives.
LONG_FORMS = [
    (2.5e-5, 2.5e-11),
    (-1.2345678e-5, 1.2e-12),
    (-1.2345678e25, 1e18),
    (-9.87654321e15, 9.6e8),
]
LABEL = "x" * MAX_FREE_TEXT_CHARS  # the longest indicator and unit a call may send


def _raw_series(level, spread, n=MAX_INPUT_POINTS):
    instants, _ = _series(n)
    rng = np.random.default_rng(4)
    return instants, [float(v) for v in level + np.cumsum(rng.normal(0, spread, n))]


def _arguments(instants, values, horizon, indicator="velocity RMS", unit="mm/s"):
    return {
        "observed_at": instants,
        "values": values,
        "indicator": indicator,
        "unit": unit,
        "resolution": "10min",
        "horizon_steps": horizon,
    }


async def _client_call(mcp, arguments, mode="legacy"):
    async with Client(mcp, mode=mode) as client:
        return await client.call_tool(
            "forecast_asset_indicator", arguments, read_timeout_seconds=60
        )


def _wire_bytes(result) -> int:
    """The received result in the larger of the two encodings the SDK writes:
    pydantic's (stdio, SSE, legacy HTTP) and the standard library's (modern
    HTTP, with non-ASCII escaped and floats in Python's form)."""
    pydantic_form = result.model_dump_json(by_alias=True, exclude_none=True)
    stdlib_form = json.dumps(
        result.model_dump(mode="json", by_alias=True, exclude_none=True),
        separators=(",", ":"),
    )
    return max(len(pydantic_form.encode()), len(stdlib_form.encode()))


async def _http_call(server, arguments):
    """Call over the modern streamable HTTP transport, in process.

    Returns the result and the bytes of the JSON-RPC ``result`` member exactly
    as the transport wrote them.
    """
    bodies: list[bytes] = []
    app = server.streamable_http_app(json_response=True)

    async def recording(scope, receive, send):
        chunks: list[bytes] = []

        async def record(message):
            if message["type"] == "http.response.body":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    bodies.append(b"".join(chunks))
            await send(message)

        await app(scope, receive, record)

    async with server.session_manager.run():
        transport = httpx2.ASGITransport(app=recording)
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://127.0.0.1:8000"
        ) as http:
            streams = streamable_http_client(
                "http://127.0.0.1:8000/mcp", http_client=http
            )
            async with Client(streams, mode="auto") as client:
                result = await client.call_tool(
                    "forecast_asset_indicator", arguments, read_timeout_seconds=60
                )
    body = [b for b in bodies if b'"isError"' in b][-1]  # the tools/call reply
    member = b'"result":'
    return result, len(body) - body.index(member) - len(member) - 1


def _named_horizon(text: str) -> tuple[int, int]:
    m = re.search(
        r"Call again with horizon_steps of (\d+): that result was recomputed and "
        r"checked to fit, at a conservative size of ([\d,]+) bytes",
        text,
    )
    assert m, text
    return int(m.group(1)), int(m.group(2).replace(",", ""))


class TestResultBudgetEnforcement:
    """The tool measures each result as the client receives it and refuses one
    above MAX_RESULT_BYTES, naming a shorter horizon it has measured to fit."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["legacy", "auto"])
    @pytest.mark.parametrize(
        "level, spread, horizon",
        [
            (3.0, 0.05, MAX_HORIZON_STEPS),
            (3.0, 0.05, 24),
            (123456.7, 0.012, MAX_HORIZON_STEPS),
            (-123456.7, 0.012, 128),
            (2.5e-5, 2.5e-11, 128),
            (-1.2345678e-5, 1.2e-12, 64),
            (-9.87654321e15, 9.6e8, 1),
        ],
    )
    async def test_measured_bytes_equal_what_the_client_receives(
        self, mcp, mode, level, spread, horizon
    ):
        instants, values = _raw_series(level, spread)
        result = await _client_call(mcp, _arguments(instants, values, horizon), mode)
        assert not result.is_error, result.content[0].text
        model = IndicatorForecastResult.model_validate(result.structured_content)
        core = call_tool_result_bytes(model)
        if mode == "legacy":
            assert _wire_bytes(result) == core
        else:  # the newer protocol adds the server's _meta serverInfo block
            assert core < _wire_bytes(result) <= core + RESULT_METADATA_ALLOWANCE
        assert _wire_bytes(result) <= client_result_bytes(model) <= MAX_RESULT_BYTES

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["legacy", "auto"])
    async def test_the_production_server_fits_within_the_allowance(self, mode):
        """The real server's name and version, as the newer protocol adds them."""
        from predictive_maintenance_mcp.server import mcp as production

        instants, values = _raw_series(3.0, 0.05, n=200)
        result = await _client_call(production, _arguments(instants, values, 24), mode)
        assert not result.is_error
        model = IndicatorForecastResult.model_validate(result.structured_content)
        assert _wire_bytes(result) <= client_result_bytes(model)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("level, spread", LONG_FORMS)
    async def test_long_number_forms_at_the_caps_are_refused_with_a_horizon_that_fits(
        self, mcp, level, spread
    ):
        instants, values = _raw_series(level, spread)
        refused = await _client_call(
            mcp, _arguments(instants, values, MAX_HORIZON_STEPS)
        )
        text = refused.content[0].text
        assert refused.is_error
        assert f"over the server's {MAX_RESULT_BYTES:,}-byte output-size limit" in text
        assert "No quantile values are truncated, further rounded or omitted" in text
        assert (
            "Its size depends on the number of forecast steps, the digits needed to "
            "represent the quantile bands, and the length of the indicator and unit "
            "labels repeated in the response." in text
        )
        assert _wire_bytes(refused) <= MAX_RESULT_BYTES
        steps, size = _named_horizon(text)
        assert steps < MAX_HORIZON_STEPS
        assert size <= MAX_RESULT_BYTES - RETRY_RESERVE
        # The retry runs at a later clock, so its size may differ; it must fit.
        retry = await _client_call(mcp, _arguments(instants, values, steps))
        assert not retry.is_error
        assert _wire_bytes(retry) <= MAX_RESULT_BYTES

    @pytest.mark.asyncio
    @pytest.mark.parametrize("horizon", [32, 96, 160, 224, MAX_HORIZON_STEPS])
    async def test_every_horizon_is_either_within_the_budget_or_refused(
        self, mcp, horizon
    ):
        instants, values = _raw_series(-9.87654321e15, 9.6e8)
        result = await _client_call(mcp, _arguments(instants, values, horizon))
        if result.is_error:
            steps, size = _named_horizon(result.content[0].text)
            assert steps < horizon and size <= MAX_RESULT_BYTES
        else:
            assert len(result.structured_content["p50"]) == horizon
        assert _wire_bytes(result) <= MAX_RESULT_BYTES

    @pytest.mark.asyncio
    @pytest.mark.parametrize("level, spread", [(3.0, 0.05), (-9.87654321e15, 9.6e8)])
    async def test_the_longest_labels_stay_within_the_budget(self, mcp, level, spread):
        instants, values = _raw_series(level, spread)
        arguments = _arguments(
            instants, values, MAX_HORIZON_STEPS, indicator=LABEL, unit=LABEL
        )
        result = await _client_call(mcp, arguments)
        assert _wire_bytes(result) <= MAX_RESULT_BYTES
        if result.is_error:
            steps, size = _named_horizon(result.content[0].text)
            assert size <= MAX_RESULT_BYTES - RETRY_RESERVE
            retry = await _client_call(mcp, {**arguments, "horizon_steps": steps})
            assert not retry.is_error
            assert _wire_bytes(retry) <= MAX_RESULT_BYTES
        else:
            assert result.structured_content["indicator"] == LABEL

    @pytest.mark.asyncio
    async def test_labels_too_long_for_any_horizon_are_refused_without_one(self, mcp):
        # The standard library escapes a character outside the basic plane to
        # twelve bytes, so with the longest such labels even one step is too much.
        label = "\U0001f4c8" * MAX_FREE_TEXT_CHARS
        instants, values = _raw_series(3.0, 0.05)
        result = await _client_call(mcp, _arguments(instants, values, 24, label, label))
        text = result.content[0].text
        assert result.is_error
        assert (
            "No shorter horizon was verified to fit within the output-size limit "
            "for these inputs. "
            "If the indicator or unit labels are long, shorten them before retrying."
            in text
        )
        assert "Call again with horizon_steps" not in text
        assert _wire_bytes(result) <= MAX_RESULT_BYTES

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "level, spread, indicator, unit",
        [
            # Floats written as 'e+25' by the standard library, 'e25' by pydantic.
            (-1.2345678e25, 1e18, "velocity RMS", "mm/s"),
            # Each non-ASCII character escaped to six bytes by the standard library.
            (3.0, 0.05, "é" * MAX_FREE_TEXT_CHARS, "°" * MAX_FREE_TEXT_CHARS),
        ],
    )
    async def test_the_modern_http_transport_receives_no_result_above_the_budget(
        self, mcp, level, spread, indicator, unit
    ):
        instants, values = _raw_series(level, spread)
        arguments = _arguments(instants, values, MAX_HORIZON_STEPS, indicator, unit)
        refused, refused_bytes = await _http_call(mcp, arguments)
        assert refused.is_error
        assert refused_bytes <= MAX_RESULT_BYTES
        steps, size = _named_horizon(refused.content[0].text)
        retry, retry_bytes = await _http_call(
            mcp, {**arguments, "horizon_steps": steps}
        )
        assert not retry.is_error
        model = IndicatorForecastResult.model_validate(retry.structured_content)
        core = call_tool_result_bytes(model)
        # The measured encoding plus the _meta serverInfo block of this protocol.
        assert core < retry_bytes <= client_result_bytes(model) <= MAX_RESULT_BYTES


class TestRetryReserve:
    """A later call of the same request grows by less than RETRY_RESERVE, so
    the horizon a refusal names still fits when the call is made again."""

    @pytest.mark.parametrize(
        "resolution, step, horizon",
        [
            ("1s", timedelta(seconds=1), MAX_HORIZON_STEPS),
            ("1min", timedelta(minutes=1), MAX_HORIZON_STEPS),
            ("10min", timedelta(minutes=10), MAX_HORIZON_STEPS),
            ("1d", timedelta(days=1), 30),
        ],
    )
    def test_a_later_clock_adds_less_than_the_reserve(self, resolution, step, horizon):
        # A clock on a whole second, so issued_at is at its shortest; the last
        # point opens the last bin and the clock is one step later: no step past.
        now = datetime(2026, 10, 26, tzinfo=timezone.utc)
        last = now - step
        instants = [
            (last - (MAX_INPUT_POINTS - 1 - i) * step).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(MAX_INPUT_POINTS)
        ]
        _, values = _raw_series(-9.87654321e15, 9.6e8)

        def size_at(clock):
            payload = indicator_forecast.forecast_indicator(
                observed_at=instants,
                values=values,
                indicator="velocity RMS",
                unit="mm/s",
                resolution=resolution,
                horizon_steps=horizon,
                backend="naive",
                aggregation="mean",
                season_length_steps=None,
                now=clock,
                server_version="0.13.0",
            )
            result = IndicatorForecastResult.model_validate(payload)
            return client_result_bytes(result), {w.code for w in result.warnings}

        fresh, codes = size_at(now)
        assert "stale_history" not in codes
        later = [step, (horizon + 1) * step, timedelta(days=36_525)]
        for offset in later:
            size, codes = size_at(now + offset + timedelta(microseconds=123_456))
            assert "stale_history" in codes
            assert size - fresh < RETRY_RESERVE, (offset, size - fresh)


class TestResultBudgetBoundary:
    """The budget is compared with the exact measured size."""

    def _call(self, horizon=24):
        instants, values = _series(400)
        return lambda: forecast_asset_indicator(
            None, **_arguments(instants, values, horizon)
        )

    @pytest.mark.asyncio
    async def test_a_result_exactly_at_the_budget_is_returned(self, monkeypatch):
        call = self._call()
        size = client_result_bytes(await call())
        monkeypatch.setattr(indicator_forecast, "MAX_RESULT_BYTES", size)
        assert client_result_bytes(await call()) == size

    @pytest.mark.asyncio
    async def test_one_byte_over_the_budget_is_refused_with_a_shorter_horizon(
        self, monkeypatch
    ):
        call = self._call()
        size = client_result_bytes(await call())
        monkeypatch.setattr(indicator_forecast, "MAX_RESULT_BYTES", size - 1)
        with pytest.raises(InvalidInputError) as exc:
            await call()
        msg = str(exc.value)
        assert f"has a conservative serialized size of {size:,} bytes" in msg
        assert f"over the server's {size - 1:,}-byte output-size limit" in msg
        steps, fitted = _named_horizon(msg)
        assert steps < 24 and fitted <= size - 1

    @pytest.mark.asyncio
    async def test_no_horizon_fits_a_budget_below_any_result(self, monkeypatch):
        monkeypatch.setattr(indicator_forecast, "MAX_RESULT_BYTES", 1_000)
        with pytest.raises(InvalidInputError) as exc:
            await self._call()()
        assert (
            "No shorter horizon was verified to fit within the output-size limit"
            in str(exc.value)
        )
