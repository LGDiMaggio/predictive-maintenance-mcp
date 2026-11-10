"""Local signal files and callers that may be on another machine.

The server does not authenticate its clients. Over HTTP, unless the call is
direct and local, signal files are closed; an operator's explicit opt-in opens
the data directory and nothing else. Stdio, in-process clients and direct
loopback connections keep every path they had.
"""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from predictive_maintenance_mcp.asset_ledger import service as ledger_service
from predictive_maintenance_mcp.mcp_tools import (
    acquisition_tools,
    import_tools,
    register_all,
    report_tools,
)
from predictive_maintenance_mcp.mcp_tools._call_origin import (
    NETWORK_DATA_DIR_OPT_IN,
    NETWORK_FILES_CLOSED,
    _SERVER,
    is_direct_local_call,
    signal_file_access,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository

RATE = 1000
LOOPBACK_PEER = ("127.0.0.1", 50000)
REMOTE_PEER = ("203.0.113.9", 50000)


def _ctx(headers: Optional[dict[str, str]], peer: Optional[str]) -> Any:
    client = None if peer is None else SimpleNamespace(host=peer)
    return SimpleNamespace(
        headers=headers,
        request_context=SimpleNamespace(request=SimpleNamespace(client=client)),
    )


class TestIsDirectLocalCall:
    LOCAL = {"host": "127.0.0.1:8000"}

    @pytest.mark.parametrize(
        "transport, bind, headers, peer, expected",
        [
            pytest.param(None, None, None, None, True, id="no-http-nothing-recorded"),
            pytest.param("stdio", "127.0.0.1", None, None, True, id="stdio"),
            pytest.param("sse", "0.0.0.0", None, None, False, id="reachable-no-http"),
            pytest.param("sse", "127.0.0.1", LOCAL, "127.0.0.1", True, id="loopback"),
            pytest.param(
                "streamable-http",
                "::1",
                {"host": "[::1]:8000"},
                "::1",
                True,
                id="ipv6-loopback",
            ),
            pytest.param(None, None, LOCAL, "127.0.0.1", False, id="bypassed-main"),
            pytest.param(
                "stdio", "127.0.0.1", LOCAL, "127.0.0.1", False, id="http-on-stdio"
            ),
            pytest.param("sse", "0.0.0.0", LOCAL, "127.0.0.1", False, id="reachable"),
            pytest.param(
                "sse", "127.0.0.1", LOCAL, "203.0.113.9", False, id="remote-peer"
            ),
            pytest.param("sse", "127.0.0.1", LOCAL, None, False, id="unknown-peer"),
            pytest.param(
                "sse",
                "127.0.0.1",
                {"host": "mcp.example.com"},
                "127.0.0.1",
                False,
                id="host-not-loopback",
            ),
            pytest.param("sse", "127.0.0.1", {}, "127.0.0.1", False, id="no-host"),
            pytest.param(
                "sse",
                "127.0.0.1",
                {**LOCAL, "x-forwarded-for": "127.0.0.1"},
                "127.0.0.1",
                False,
                id="forwarded",
            ),
        ],
    )
    def test_decision(self, monkeypatch, transport, bind, headers, peer, expected):
        monkeypatch.setitem(_SERVER, "transport", transport)
        monkeypatch.setitem(_SERVER, "host", bind)
        assert is_direct_local_call(_ctx(headers, peer)) is expected

    def test_a_missing_context_is_local_unless_the_bind_is_reachable(self, monkeypatch):
        assert is_direct_local_call(None) is True
        monkeypatch.setitem(_SERVER, "transport", "sse")
        monkeypatch.setitem(_SERVER, "host", "0.0.0.0")
        assert is_direct_local_call(None) is False


def _write_signal(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.sin(2 * np.pi * 50 * np.arange(RATE) / RATE)
    pd.DataFrame(samples).to_csv(path, index=False, header=False)
    return path


async def _over_http(
    tool: str,
    arguments: dict[str, Any],
    *,
    peer: tuple[str, int] = LOOPBACK_PEER,
    headers: Optional[dict[str, str]] = None,
):
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.server.mcpserver import MCPServer

    app_server = MCPServer("test-network-files")
    register_all(app_server)
    app = app_server.streamable_http_app(json_response=True)
    async with app_server.session_manager.run():
        transport = httpx2.ASGITransport(app=app, client=peer)
        async with httpx2.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8000",
            headers=headers or {},
        ) as http:
            streams = streamable_http_client(
                "http://127.0.0.1:8000/mcp", http_client=http
            )
            async with Client(streams, mode="auto") as client:
                return await client.call_tool(tool, arguments, read_timeout_seconds=120)


@pytest.fixture
def store():
    repository = get_repository()
    repository.clear_all()
    yield repository
    repository.clear_all()


@pytest.fixture
def inside(sandbox_data_dir) -> Path:
    return _write_signal(sandbox_data_dir / "run.csv")


@pytest.fixture
def outside(tmp_path) -> Path:
    return _write_signal(tmp_path / "elsewhere ZQ" / "secret.csv")


def _load(filepath: Any) -> dict[str, Any]:
    return {"filepath": filepath, "sampling_rate": float(RATE), "signal_unit": "g"}


def _assert_refused_silently(result: Any, store: Any) -> None:
    assert result.is_error
    text = result.content[0].text
    assert "data directory" in text
    assert "ZQ" not in text and "secret" not in text
    assert store.list_signals() == []


@pytest.fixture
def opted_in(monkeypatch) -> None:
    monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, "1")


@pytest.mark.usefixtures("opted_in")
class TestLoadSignalOverTheNetwork:
    @pytest.mark.anyio
    async def test_an_absolute_path_outside_the_data_directory_is_refused(
        self, store, inside, outside
    ):
        result = await _over_http("load_signal", _load(str(outside)))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_a_relative_path_leaving_the_data_directory_is_refused(
        self, store, sandbox_data_dir, outside
    ):
        relative = os.path.relpath(outside, sandbox_data_dir)
        result = await _over_http("load_signal", _load(relative))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_one_outside_path_refuses_the_whole_batch(
        self, store, inside, outside
    ):
        result = await _over_http("load_signal", _load(["run.csv", str(outside)]))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_a_network_share_is_refused_before_the_repository_runs(
        self, store, inside, monkeypatch
    ):
        def never(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise AssertionError("a refused path reached the repository")

        resolved: list[Any] = []
        monkeypatch.setattr(store, "load_signal", never)
        monkeypatch.setattr(
            acquisition_tools, "safe_resolve", lambda *a: resolved.append(a)
        )
        result = await _over_http("load_signal", _load("//server/share/ZQ.csv"))
        _assert_refused_silently(result, store)
        assert resolved == []

    @pytest.mark.anyio
    async def test_a_link_leaving_the_data_directory_is_refused(
        self, store, sandbox_data_dir, outside
    ):
        link = sandbox_data_dir / "linked"
        try:
            os.symlink(outside.parent, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            winapi = pytest.importorskip("_winapi")
            winapi.CreateJunction(str(outside.parent), str(link))
        result = await _over_http("load_signal", _load("linked/secret.csv"))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_a_file_linking_out_of_the_data_directory_is_refused(
        self, store, sandbox_data_dir, outside
    ):
        try:
            os.symlink(outside, sandbox_data_dir / "linked.csv")
        except (OSError, NotImplementedError):
            pytest.skip("this account cannot create file links")
        result = await _over_http("load_signal", _load("linked.csv"))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_the_disk_listing_answers(self, store, inside):
        result = await _over_http("list_signals", {"scope": "disk"})
        assert not result.is_error, result.content
        assert "run.csv" in result.content[0].text

    @pytest.mark.anyio
    async def test_a_companion_linking_out_of_the_data_directory_is_refused(
        self, store, inside, tmp_path
    ):
        secret = tmp_path / "elsewhere ZQ" / "secret_metadata.json"
        secret.parent.mkdir()
        secret.write_text('{"secret": "ZQ"}', encoding="utf-8")
        try:
            os.symlink(secret, inside.with_name("run_metadata.json"))
        except (OSError, NotImplementedError):
            pytest.skip("this account cannot create file links")
        result = await _over_http("load_signal", _load("run.csv"))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    @pytest.mark.skipif(os.name == "nt", reason="the kernel walks '..' after a link")
    async def test_a_path_climbing_out_through_a_link_is_refused(
        self, store, sandbox_data_dir, outside
    ):
        target = outside.parent / "deeper"
        target.mkdir()
        os.symlink(target, sandbox_data_dir / "linked", target_is_directory=True)
        result = await _over_http("load_signal", _load("linked/../secret.csv"))
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "peer, headers",
        [
            pytest.param(REMOTE_PEER, {}, id="remote-peer-forged-host"),
            pytest.param(LOOPBACK_PEER, {"X-Forwarded-For": "1.2.3.4"}, id="proxied"),
        ],
    )
    async def test_a_loopback_bind_does_not_vouch_for_the_caller(
        self, store, outside, monkeypatch, peer, headers
    ):
        monkeypatch.setitem(_SERVER, "transport", "streamable-http")
        monkeypatch.setitem(_SERVER, "host", "127.0.0.1")
        result = await _over_http(
            "load_signal", _load(str(outside)), peer=peer, headers=headers
        )
        _assert_refused_silently(result, store)

    @pytest.mark.anyio
    async def test_a_file_under_the_data_directory_still_loads(self, store, inside):
        result = await _over_http("load_signal", _load("run.csv"))
        assert not result.is_error, result.content
        assert len(store.list_signals()) == 1


class TestDirectLocalCallersKeepEveryPath:
    @pytest.mark.anyio
    async def test_an_in_process_client_loads_an_absolute_path(self, store, outside):
        from predictive_maintenance_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            result = await client.call_tool("load_signal", _load(str(outside)))
        assert not result.is_error, result.content
        assert len(store.list_signals()) == 1

    @pytest.mark.anyio
    async def test_a_direct_loopback_call_loads_an_absolute_path(
        self, store, outside, monkeypatch
    ):
        monkeypatch.setitem(_SERVER, "transport", "streamable-http")
        monkeypatch.setitem(_SERVER, "host", "127.0.0.1")
        result = await _over_http("load_signal", _load(str(outside)))
        assert not result.is_error, result.content
        assert len(store.list_signals()) == 1


class TestReprocessOverTheNetwork:
    @pytest.fixture
    def seen(self, monkeypatch) -> dict[str, Any]:
        calls: dict[str, Any] = {}
        real = ledger_service.reprocess_for_assessment

        def spy(*args: Any, **kwargs: Any):
            calls.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(ledger_service, "reprocess_for_assessment", spy)
        return calls

    ARGUMENTS = {
        "asset_id": "P-101",
        "measurement_point_id": "motor_de",
        "reprocess": True,
    }
    POINT = {
        "asset_id": "P-101",
        "measurement_point_id": "motor_de",
        "machine_group": 2,
        "support_type": "rigid",
        "declared_by": "test",
    }

    @pytest.mark.anyio
    async def test_a_network_call_is_refused_by_default(
        self, sandbox_data_dir, seen, store
    ):
        result = await _over_http("assess_asset_change", self.ARGUMENTS)
        _assert_closed(result, store)
        assert seen == {}

    @pytest.mark.anyio
    async def test_an_assessment_without_reprocess_still_answers(
        self, sandbox_data_dir, seen
    ):
        from predictive_maintenance_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            await client.call_tool("declare_measurement_point", self.POINT)
        arguments = {**self.ARGUMENTS, "reprocess": False}
        result = await _over_http("assess_asset_change", arguments)
        assert not result.is_error, result.content

    @pytest.mark.anyio
    async def test_a_network_call_reads_the_data_directory_only(
        self, sandbox_data_dir, seen, opted_in
    ):
        from predictive_maintenance_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            await client.call_tool("declare_measurement_point", self.POINT)
        result = await _over_http("assess_asset_change", self.ARGUMENTS)
        assert not result.is_error, result.content
        assert seen["data_dir_only"] is True
        assert seen["import_roots"] is None

    @pytest.mark.anyio
    async def test_a_local_call_keeps_every_recorded_location(
        self, sandbox_data_dir, seen
    ):
        from predictive_maintenance_mcp.server import mcp

        async with Client(mcp, mode="legacy") as client:
            await client.call_tool("declare_measurement_point", self.POINT)
            result = await client.call_tool("assess_asset_change", self.ARGUMENTS)
        assert not result.is_error, result.content
        assert seen["data_dir_only"] is False


class TestSignalFileAccess:
    REMOTE = {"host": "127.0.0.1:8000"}

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, "none"),
            ("", "none"),
            ("0", "none"),
            ("false", "none"),
            ("no", "none"),
            ("off", "none"),
            ("maybe", "none"),
            ("1", "data_dir"),
            ("true", "data_dir"),
            ("YES", "data_dir"),
            (" on ", "data_dir"),
        ],
    )
    def test_the_opt_in_is_explicit(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv(NETWORK_DATA_DIR_OPT_IN, raising=False)
        else:
            monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, value)
        assert signal_file_access(_ctx(self.REMOTE, "203.0.113.9")) == expected

    @pytest.mark.parametrize("value", [None, "1"])
    def test_a_direct_local_call_does_not_need_it(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(NETWORK_DATA_DIR_OPT_IN, raising=False)
        else:
            monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, value)
        assert signal_file_access(None) == "all"
        monkeypatch.setitem(_SERVER, "transport", "sse")
        monkeypatch.setitem(_SERVER, "host", "127.0.0.1")
        assert signal_file_access(_ctx(self.REMOTE, "127.0.0.1")) == "all"


def _assert_closed(result: Any, store: Any) -> None:
    assert result.is_error
    text = result.content[0].text
    assert NETWORK_FILES_CLOSED in text
    assert "not authentication" in text
    assert store.list_signals() == []


class TestSignalFilesAreClosedByDefault:
    @pytest.mark.anyio
    async def test_a_file_under_the_data_directory_is_refused(self, store, inside):
        _assert_closed(await _over_http("load_signal", _load("run.csv")), store)

    @pytest.mark.anyio
    async def test_a_batch_is_refused(self, store, inside):
        _assert_closed(await _over_http("load_signal", _load(["run.csv"])), store)

    @pytest.mark.anyio
    async def test_an_absolute_path_is_refused(self, store, outside):
        _assert_closed(await _over_http("load_signal", _load(str(outside))), store)

    @pytest.mark.anyio
    async def test_a_reachable_bind_is_refused(self, store, inside, monkeypatch):
        monkeypatch.setitem(_SERVER, "transport", "sse")
        monkeypatch.setitem(_SERVER, "host", "0.0.0.0")
        _assert_closed(await _over_http("load_signal", _load("run.csv")), store)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "arguments", [{}, {"scope": "disk"}], ids=["default", "disk"]
    )
    async def test_the_disk_listing_is_refused(self, store, inside, arguments):
        _assert_closed(await _over_http("list_signals", arguments), store)

    @pytest.mark.anyio
    async def test_the_memory_listing_still_answers(self, store):
        result = await _over_http("list_signals", {"scope": "memory"})
        assert not result.is_error, result.content

    @pytest.mark.anyio
    async def test_a_generated_signal_is_still_available(self, store, sandbox_data_dir):
        result = await _over_http(
            "generate_test_signal", {"signal_type": "normal", "duration": 1.0}
        )
        assert not result.is_error, result.content
        assert len(store.list_signals()) == 1

    @pytest.mark.anyio
    async def test_the_companion_of_a_stored_signal_is_not_read(
        self, store, inside, monkeypatch
    ):
        seen: list[str] = []

        def spy(info: Any, access: str = "all") -> dict[str, Any]:
            seen.append(access)
            return {}

        from predictive_maintenance_mcp.server import mcp

        monkeypatch.setattr(report_tools, "_companion_metadata", spy)
        async with Client(mcp, mode="legacy") as client:
            await client.call_tool("load_signal", _load("run.csv"))
            signal_id = store.list_signals()[0]["signal_id"]
            arguments = {
                "signal_id": signal_id,
                "filter_low": 100.0,
                "filter_high": 400.0,
            }
            await client.call_tool("generate_envelope_report", arguments)
        await _over_http("generate_envelope_report", arguments)
        monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, "1")
        await _over_http("generate_envelope_report", arguments)
        assert seen == ["all", "none", "data_dir"]


class TestTheOptInStopsAtTheDataDirectory:
    @pytest.mark.anyio
    async def test_it_does_not_open_the_import_tools(
        self, store, inside, sandbox_data_dir, monkeypatch
    ):
        monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, "1")
        monkeypatch.setenv("PMM_IMPORT_ROOTS", str(sandbox_data_dir))
        result = await _over_http("inspect_signal_file", {"filepath": str(inside)})
        assert result.is_error
        assert "File import is refused" in result.content[0].text

    def test_it_does_not_change_the_import_startup_refusal(self):
        environ = {NETWORK_DATA_DIR_OPT_IN: "1"}
        assert import_tools.remote_opt_in_refusal("sse", "0.0.0.0", environ) is None
        environ["PMM_IMPORT_ALLOW_REMOTE"] = "1"
        assert import_tools.remote_opt_in_refusal("sse", "0.0.0.0", environ)


class TestStartupMessages:
    @staticmethod
    def _main(monkeypatch, *arguments: str) -> None:
        import predictive_maintenance_mcp.server as srv

        monkeypatch.setattr(srv.mcp, "run", MagicMock())
        monkeypatch.setattr(srv, "_setup_environment", lambda: None)
        for name in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(sys, "argv", ["server", *arguments])
        srv.main()

    @staticmethod
    def _warnings(caplog: Any) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelname == "WARNING" and NETWORK_DATA_DIR_OPT_IN in r.getMessage()
        ]

    def test_the_opt_in_warns_that_it_is_not_authentication(
        self, monkeypatch, package_caplog
    ):
        monkeypatch.setenv(NETWORK_DATA_DIR_OPT_IN, "1")
        self._main(monkeypatch, "--transport", "sse", "--host", "0.0.0.0")
        [warning] = self._warnings(package_caplog)
        assert "can read signal files" in warning
        assert "not authentication" in warning

    def test_a_reachable_bind_says_files_are_closed(self, monkeypatch, package_caplog):
        monkeypatch.delenv(NETWORK_DATA_DIR_OPT_IN, raising=False)
        self._main(monkeypatch, "--transport", "sse", "--host", "0.0.0.0")
        [warning] = self._warnings(package_caplog)
        assert "not read for network clients" in warning

    @pytest.mark.parametrize(
        "arguments",
        [("--transport", "stdio"), ("--transport", "sse", "--host", "127.0.0.1")],
        ids=["stdio", "loopback"],
    )
    def test_nothing_is_said_where_nothing_changes(
        self, monkeypatch, package_caplog, arguments
    ):
        monkeypatch.delenv(NETWORK_DATA_DIR_OPT_IN, raising=False)
        self._main(monkeypatch, *arguments)
        assert self._warnings(package_caplog) == []


class TestDeploymentProxyExamples:
    """Examples for a server bound to loopback send what the policy relies on.

    A proxy on this machine that adds no forwarding header and presents a
    loopback ``Host`` cannot be told from a direct local call.
    """

    @staticmethod
    def _blocks(upstream: str) -> list[str]:
        guide = Path(__file__).resolve().parent.parent / "docs" / "DEPLOYMENT.md"
        text = guide.read_text(encoding="utf-8")
        return [
            b for b in re.findall(r"```[a-z]*\n(.*?)```", text, re.S) if upstream in b
        ]

    def test_every_nginx_example_marks_the_call_and_sends_a_loopback_host(self):
        blocks = self._blocks("proxy_pass http://127.0.0.1:8000")
        assert len(blocks) == 2
        for block in blocks:
            assert re.search(
                r"^\s*proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;",
                block,
                re.M,
            )
            # With no Host line, nginx sends $proxy_host by default.
            assert set(
                re.findall(r"^\s*proxy_set_header Host (\S+);", block, re.M)
            ) <= {"$proxy_host"}

    def test_every_caddy_example_sends_a_loopback_host(self):
        blocks = self._blocks("reverse_proxy 127.0.0.1:8000")
        assert len(blocks) == 3
        for block in blocks:
            assert re.search(
                r"^\s*header_up Host \{upstream_hostport\}\s*$", block, re.M
            )
