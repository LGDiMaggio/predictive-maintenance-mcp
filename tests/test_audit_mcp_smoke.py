"""Guards for the stdio protocol smoke harness (``python -m tools.audit smoke``).

Every other protocol test in this suite talks to the server in-process
(``Client(mcp, mode="legacy")``), so nothing proved that the thing a desktop
host actually launches -- a subprocess speaking newline-delimited JSON-RPC on
its stdout -- completes a handshake, lists the frozen tools, runs the
documented chain and refuses an unknown signal with project wording. The
harness in ``tools/audit/mcp_smoke.py`` does that over real stdio, and this
module proves the harness itself:

* the pure parts first (inventory diff, schema check, server command
  parsing), because a guard that silently stopped detecting would look
  identical to a clean tree (the result contract and CLI dispatch shared by
  every audit check are pinned in ``tests/test_audit_results.py``);
* then the harness against this checkout's server in both handshake eras,
  with ``conftest.SUBPROCESS_PIN`` prepended so a worktree tests its own
  source rather than whatever the editable install points at;
* then the failure paths it builds: a server that dies at startup, one that
  corrupts stdout, one that never answers, and an inventory naming a tool
  the server does not register. Each must end as ``error`` or ``fail``,
  never ``pass``.

The stdout-corruption case matters more than it looks: under mcp 2.x a
non-JSON line does NOT end the session. The transport hands it to the
client's message handler, whose default discards it, so a server printing
to fd 1 passes every in-process test and breaks hosts that are stricter.
"""

import json
import sys
from pathlib import Path

import pytest

from tools.audit import mcp_smoke, results

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = REPO_ROOT / "tests" / "fixtures" / "tool_inventory.json"


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


class TestInventoryDiff:
    def test_a_phantom_inventory_tool_is_reported_missing(self):
        diff = mcp_smoke.diff_inventory(
            discovered=["analyze_fft", "load_signal"],
            expected=["analyze_fft", "load_signal", "phantom_tool"],
        )
        assert diff == {"missing": ["phantom_tool"], "unexpected": []}

    def test_a_server_tool_absent_from_the_inventory_is_reported_unexpected(self):
        diff = mcp_smoke.diff_inventory(
            discovered=["analyze_fft", "load_signal", "phantom_tool"],
            expected=["analyze_fft", "load_signal"],
        )
        assert diff == {"missing": [], "unexpected": ["phantom_tool"]}

    def test_equal_sets_report_nothing(self):
        diff = mcp_smoke.diff_inventory(["b", "a"], ["a", "b"])
        assert diff == {"missing": [], "unexpected": []}

    def test_the_frozen_inventory_loads_as_tool_names(self):
        names = mcp_smoke.load_inventory(INVENTORY)
        reference = json.loads(INVENTORY.read_text(encoding="utf-8"))
        assert names == set(reference["tools"])


class TestSchemaCheck:
    VALID_INPUT = {
        "type": "object",
        "properties": {"signal_id": {"type": "string"}},
        "required": ["signal_id"],
    }

    def _tool(self, name="analyze_fft", input_schema=None, output_schema=None):
        return {
            "name": name,
            "inputSchema": self.VALID_INPUT if input_schema is None else input_schema,
            "outputSchema": output_schema,
        }

    def test_valid_schemas_produce_no_violation(self):
        tool = self._tool(output_schema={"type": "object", "properties": {}})
        assert mcp_smoke.check_tool_schemas([tool]) == []

    def test_an_input_schema_without_type_object_is_a_violation(self):
        schema = {"properties": {"signal_id": {"type": "string"}}}
        violations = mcp_smoke.check_tool_schemas([self._tool(input_schema=schema)])
        assert [(v.tool, v.field) for v in violations] == [
            ("analyze_fft", "inputSchema")
        ]
        assert "object" in violations[0].problem

    def test_an_invalid_keyword_type_in_an_input_schema_is_a_violation(self):
        schema = {"type": "object", "required": "signal_id"}
        violations = mcp_smoke.check_tool_schemas([self._tool(input_schema=schema)])
        assert [(v.tool, v.field) for v in violations] == [
            ("analyze_fft", "inputSchema")
        ]

    def test_an_invalid_keyword_type_in_an_output_schema_is_a_violation(self):
        schema = {"type": "object", "properties": {"zone": {"type": 5}}}
        violations = mcp_smoke.check_tool_schemas([self._tool(output_schema=schema)])
        assert [(v.tool, v.field) for v in violations] == [
            ("analyze_fft", "outputSchema")
        ]


class TestServerCommand:
    def test_split_server_cmd_accepts_one_string_or_repeated_elements(self):
        assert mcp_smoke.split_server_cmd(["python -m predictive_maintenance_mcp"]) == [
            "python",
            "-m",
            "predictive_maintenance_mcp",
        ]
        assert mcp_smoke.split_server_cmd(["C:/a b/python.exe", "-m", "pkg"]) == [
            "C:/a b/python.exe",
            "-m",
            "pkg",
        ]


# ---------------------------------------------------------------------------
# Integration: this checkout's server over real stdio, both eras
# ---------------------------------------------------------------------------


def _checkout_listing() -> dict[str, list[str]]:
    """Files under the checkout's data/ and reports/, listed independently."""
    listing = {}
    for tree in ("data", "reports"):
        base = REPO_ROOT / tree
        listing[tree] = sorted(
            path.relative_to(base).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
    return listing


@pytest.fixture(scope="module")
def checkout_run(tmp_path_factory):
    """One harness run against this checkout, pinned like every subprocess test.

    Without the pin the child would import whatever the editable install
    points at, which in a worktree is another checkout's source.
    """
    from conftest import SUBPROCESS_PIN

    server_cmd = [
        sys.executable,
        "-c",
        SUBPROCESS_PIN + "import runpy; runpy.run_module("
        "'predictive_maintenance_mcp', run_name='__main__')",
    ]
    before = _checkout_listing()
    result = mcp_smoke.run_smoke(
        mcp_smoke.SmokeConfig(
            server_cmd=server_cmd,
            inventory=INVENTORY,
            expect_version=mcp_smoke.default_expected_version(),
            probe_prelude=SUBPROCESS_PIN,
            variant="pytest-checkout",
        )
    )
    after = _checkout_listing()
    target = tmp_path_factory.mktemp("smoke") / "smoke.json"
    doc = json.loads(results.write_json(result, target).read_text(encoding="utf-8"))
    return {"result": result, "doc": doc, "before": before, "after": after}


class TestCheckoutOverStdio:
    def test_passes_in_both_eras(self, checkout_run):
        result = checkout_run["result"]
        gated = [f.to_dict() for f in result.findings if f.severity != "info"]
        assert result.status == "pass", gated
        assert result.exit_code == 0

    def test_legacy_era_negotiates_a_handshake_version(self, checkout_run):
        from mcp.types.version import (
            HANDSHAKE_PROTOCOL_VERSIONS,
            KNOWN_PROTOCOL_VERSIONS,
        )

        negotiated = checkout_run["doc"]["facts"]["negotiated_protocol"]
        assert negotiated["legacy"] in HANDSHAKE_PROTOCOL_VERSIONS
        assert negotiated["auto"] in KNOWN_PROTOCOL_VERSIONS

    def test_discovered_tools_equal_the_inventory_in_both_eras(self, checkout_run):
        frozen = len(json.loads(INVENTORY.read_text(encoding="utf-8"))["tools"])
        assert checkout_run["doc"]["facts"]["tool_count"] == {
            "legacy": frozen,
            "auto": frozen,
        }
        assert frozen >= 30, "anti-rot: the inventory shrank to a near-empty surface"

    def test_every_chain_step_returned_a_structured_result(self, checkout_run):
        chain = checkout_run["doc"]["facts"]["chain"]
        assert list(chain) == sorted(mcp_smoke.CHAIN_TOOLS)
        assert set(chain.values()) == {"ok"}

    def test_chain_artifacts_landed_in_the_sandbox(self, checkout_run):
        files = checkout_run["doc"]["facts"]["sandbox_files"]["legacy"]
        assert any(f.startswith("data/signals/") and f.endswith(".csv") for f in files)
        assert any(f.startswith("reports/") and f.endswith(".html") for f in files)

    def test_the_checkout_data_and_reports_trees_are_unchanged(self, checkout_run):
        assert checkout_run["after"] == checkout_run["before"]

    def test_an_unknown_signal_id_is_refused_with_the_load_signal_remedy(
        self, checkout_run
    ):
        expected = {"is_error": True, "remedy_named": True}
        assert checkout_run["doc"]["facts"]["refusal"] == {
            "legacy": expected,
            "auto": expected,
        }

    def test_the_import_probe_resolved_this_checkout(self, checkout_run):
        facts = checkout_run["doc"]["facts"]
        source = (REPO_ROOT / "src" / "__init__.py").resolve()
        assert Path(facts["import_location"]).resolve() == source
        assert facts["package_version"] == facts["expected_version"]

    def test_the_written_result_validates(self, checkout_run):
        results.validate_result(checkout_run["doc"])
        assert checkout_run["doc"]["check"] == "smoke"


# ---------------------------------------------------------------------------
# Failure paths: never a pass
# ---------------------------------------------------------------------------

#: A minimal real MCP server. ``clean`` serves; ``noisy`` first writes a
#: non-JSON line on stdout; ``silent`` never answers, and exits once the
#: client closes its stdin. Its one tool refuses every id with the project's
#: remedy, so an ``auto``-era run against it has nothing to report except
#: what the mode injects.
STUB_SERVER = '''\
import sys

mode = sys.argv[1]
if mode == "noisy":
    sys.stdout.write("pmm-audit-stub: this line is not JSON-RPC\\n")
    sys.stdout.flush()
elif mode == "silent":
    sys.stdin.read()
    sys.exit(0)

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

app = MCPServer("audit-stub")


def analyze_fft(signal_id: str) -> str:
    """Refuse every id the way the real server does."""
    raise ToolError(f"Signal {signal_id!r} is not loaded. Load it with load_signal.")


app.tool()(analyze_fft)
app.run(transport="stdio")
'''


@pytest.fixture
def stub(tmp_path):
    """Command factory for the stub server, plus an inventory matching it."""
    script = tmp_path / "stub_server.py"
    script.write_text(STUB_SERVER, encoding="utf-8")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"tools": {"analyze_fft": {}}}), encoding="utf-8")

    def command(mode):
        return [sys.executable, str(script), mode]

    command.inventory = inventory
    return command


def _gated(result, era):
    """Non-info findings recorded for ``era``."""
    return [
        f.rule
        for f in result.findings
        if f.severity != "info" and era in f.evidence.get("eras", [])
    ]


class TestFailurePaths:
    def test_a_server_exiting_with_code_3_is_an_error_with_its_stderr(self, tmp_path):
        script = tmp_path / "exit3.py"
        script.write_text(
            "import sys\nsys.stderr.write('pmm-boom-marker\\n')\nsys.exit(3)\n",
            encoding="utf-8",
        )
        result = mcp_smoke.run_smoke(
            mcp_smoke.SmokeConfig(
                server_cmd=[sys.executable, str(script)], inventory=INVENTORY
            )
        )
        assert (result.status, result.exit_code) == ("error", 2)
        errors = [f for f in result.findings if f.rule == "session-error"]
        assert sorted(f.subject for f in errors) == [
            "auto era startup",
            "legacy era startup",
        ]
        assert all("pmm-boom-marker" in f.evidence["excerpt"] for f in errors)

    def test_a_non_json_stdout_line_is_an_error_not_a_pass(self, stub):
        result = mcp_smoke.run_smoke(
            mcp_smoke.SmokeConfig(server_cmd=stub("noisy"), inventory=stub.inventory)
        )
        assert (result.status, result.exit_code) == ("error", 2)
        noise = {
            f.subject: f.evidence["excerpt"]
            for f in result.findings
            if f.rule == "stdout-not-json-rpc"
        }
        assert set(noise) == {"legacy era server stdout", "auto era server stdout"}
        assert all("this line is not JSON-RPC" in text for text in noise.values())
        # The auto era has no chain, so the noise is its only problem: without
        # the recording handler this era would have passed.
        assert _gated(result, "auto") == ["stdout-not-json-rpc"]

    def test_a_phantom_inventory_tool_fails_the_run(self, stub, tmp_path):
        inventory = tmp_path / "phantom.json"
        inventory.write_text(
            json.dumps({"tools": {"analyze_fft": {}, "phantom_tool": {}}}),
            encoding="utf-8",
        )
        result = mcp_smoke.run_smoke(
            mcp_smoke.SmokeConfig(
                server_cmd=stub("clean"), inventory=inventory, eras=("auto",)
            )
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        assert _gated(result, "auto") == ["tool-missing"]
        assert [f.subject for f in result.findings if f.rule == "tool-missing"] == [
            "phantom_tool"
        ]

    def test_a_server_that_never_answers_times_out_as_an_error(self, stub):
        result = mcp_smoke.run_smoke(
            mcp_smoke.SmokeConfig(
                server_cmd=stub("silent"),
                inventory=stub.inventory,
                eras=("legacy",),
                timeout_startup=1.0,
            )
        )
        assert (result.status, result.exit_code) == ("error", 2)
        assert [f.subject for f in result.findings if f.rule == "timeout"] == [
            "legacy era startup"
        ]
