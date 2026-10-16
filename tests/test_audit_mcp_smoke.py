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
  never ``pass``;
* then one deterministic defect per remaining rule, against a stub of the
  documented chain (``CHAIN_STUB``, whose clean mode is the control run): a
  refusal that is not one or lacks its remedy, a chain step that errors or
  returns nothing usable, an artifact outside the sandbox or missing, a write
  into a stand-in checkout, an unexpected tool, an invalid published schema.
  Where no real server can produce the defect, the input is synthetic: the
  SDK's handshake version set, and a fake package for the import probe
  (wrong location, wrong version, unimportable, no interpreter). Each test
  asserts the rule, the run status and the evidence a maintainer acts on,
  and fails when its rule is disabled.

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

    def test_an_unreadable_inventory_is_an_error_naming_the_file(self, stub, tmp_path):
        missing = tmp_path / "no-such-inventory.json"
        result = mcp_smoke.run_smoke(
            mcp_smoke.SmokeConfig(
                server_cmd=stub("clean"), inventory=missing, eras=("auto",)
            )
        )
        assert (result.status, result.exit_code) == ("error", 2)
        (finding,) = _rule(result, "inventory-unreadable")
        assert finding.classification == "not-checked"
        assert str(missing) in finding.evidence["excerpt"]


# ---------------------------------------------------------------------------
# Contract failures: one deterministic defect per rule
# ---------------------------------------------------------------------------

#: A stub of the documented chain with every tool the harness calls, all of
#: them returning structured results inside the sandbox. ``argv[1]`` injects
#: one defect, ``argv[2]`` is a directory some defects write to. ``clean``
#: injects nothing; TestChainStubControl proves it passes, so each failure
#: below comes from its defect alone.
CHAIN_STUB = """\
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

MODE = sys.argv[1]
ELSEWHERE = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SIGNAL_ID = "stub-signal-1"
app = MCPServer("audit-chain-stub")


def _root() -> Path:
    return Path(os.environ["PDM_PROJECT_DIR"])


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def generate_test_signal(
    signal_type: str,
    duration: float,
    sampling_rate: float,
    noise_level: float,
    random_seed: int,
) -> dict[str, Any]:
    if MODE == "signal-without-id":
        return {"filepath": _write(_root() / "data" / "stub.csv", "t,x")}
    if MODE == "signal-escapes":
        return {"signal_id": SIGNAL_ID, "filepath": _write(ELSEWHERE / "escaped.csv", "t,x")}
    return {"signal_id": SIGNAL_ID, "filepath": _write(_root() / "data" / "stub.csv", "t,x")}


def analyze_fft(signal_id: str) -> dict[str, Any]:
    if signal_id == SIGNAL_ID:
        if MODE == "writes-checkout":
            _write(ELSEWHERE / "reports" / "stub-leak.html", "<html></html>")
        return {"peaks": [10.0]}
    if MODE == "accepts-unknown":
        return {"peaks": []}
    if MODE == "refuses-without-remedy":
        raise ToolError("pmm-stub: that signal does not exist")
    raise ToolError(f"Signal {signal_id!r} is not loaded. Load it with load_signal.")


def analyze_envelope(signal_id: str) -> dict[str, Any]:
    if MODE == "envelope-fails":
        raise ToolError("pmm-stub-marker: the envelope step exploded")
    return {"peaks": [10.0]}


def check_bearing_faults(
    signal_id: str, rpm: float, frequencies: dict[str, float]
) -> dict[str, Any]:
    return {"matches": sorted(frequencies)}


def assess_severity(
    signal_id: str, rpm: float, machine_group: int, support_type: str
) -> dict[str, Any]:
    return {"zone": "A"}


def generate_envelope_report(signal_id: str) -> dict[str, Any]:
    target = _root() / "reports" / "stub-report.html"
    if MODE == "report-missing":
        return {"file_path": str(target)}
    return {"file_path": _write(target, "<html></html>")}


def rogue_tool() -> dict[str, Any]:
    return {}


for function in (
    generate_test_signal,
    analyze_fft,
    analyze_envelope,
    check_bearing_faults,
    assess_severity,
    generate_envelope_report,
):
    app.tool()(function)
if MODE == "extra-tool":
    app.tool()(rogue_tool)
if MODE == "schema-required-not-a-list":
    app._tool_manager.get_tool("analyze_envelope").parameters = {
        "type": "object",
        "properties": {"signal_id": {"type": "string"}},
        "required": "signal_id",
    }
if MODE == "schema-not-an-object":
    app._tool_manager.get_tool("analyze_envelope").parameters = {"type": "array"}
app.run(transport="stdio")
"""


@pytest.fixture
def chain_stub(tmp_path):
    """Command factory for the chain stub, plus an inventory matching it."""
    script = tmp_path / "chain_stub.py"
    script.write_text(CHAIN_STUB, encoding="utf-8")
    inventory = tmp_path / "chain-inventory.json"
    inventory.write_text(
        json.dumps({"tools": {name: {} for name in mcp_smoke.CHAIN_TOOLS}}),
        encoding="utf-8",
    )

    def command(mode, elsewhere=None):
        argv = [sys.executable, str(script), mode]
        return argv + ([str(elsewhere)] if elsewhere is not None else [])

    command.inventory = inventory
    return command


def _rule(result, rule):
    return [f for f in result.findings if f.rule == rule]


def _run(server_cmd, inventory, eras=("legacy",), **config):
    return mcp_smoke.run_smoke(
        mcp_smoke.SmokeConfig(
            server_cmd=server_cmd, inventory=inventory, eras=eras, **config
        )
    )


class TestChainStubControl:
    def test_the_clean_chain_stub_passes_in_both_eras(self, chain_stub):
        """The control for every defect below: without one, nothing is gated."""
        result = _run(chain_stub("clean"), chain_stub.inventory, eras=mcp_smoke.ERAS)
        gated = [f.rule for f in result.findings if f.severity != "info"]
        assert (result.status, gated) == ("pass", [])
        assert set(result.facts["chain"].values()) == {"ok"}


class TestRefusalContract:
    def test_an_unknown_signal_that_is_accepted_fails(self, chain_stub):
        result = _run(
            chain_stub("accepts-unknown"), chain_stub.inventory, eras=("auto",)
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "refusal-contract")
        assert finding.evidence["arguments"] == {
            "signal_id": mcp_smoke.UNKNOWN_SIGNAL_ID
        }
        assert finding.evidence["is_error"] is False
        assert finding.evidence["eras"] == ["auto"]

    def test_a_refusal_without_the_load_signal_remedy_fails(self, chain_stub):
        result = _run(
            chain_stub("refuses-without-remedy"), chain_stub.inventory, eras=("auto",)
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "refusal-contract")
        assert finding.evidence["is_error"] is True
        # The server's own words are the evidence a maintainer acts on.
        assert "that signal does not exist" in finding.evidence["excerpt"]


class TestChainFailures:
    def test_a_step_that_errors_fails_and_stops_the_chain(self, chain_stub):
        result = _run(chain_stub("envelope-fails"), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "chain-step-failed")
        assert finding.subject == "analyze_envelope"
        assert finding.evidence["is_error"] is True
        assert finding.evidence["arguments"] == {"signal_id": "stub-signal-1"}
        assert "pmm-stub-marker" in finding.evidence["excerpt"]
        chain = result.facts["chain"]
        assert (chain["analyze_fft"], chain["analyze_envelope"]) == ("ok", "failed")
        assert chain["generate_envelope_report"] == "not-run"

    def test_a_step_without_a_usable_structured_result_fails(self, chain_stub):
        """No error flag, but no signal_id either: the chain cannot go on."""
        result = _run(chain_stub("signal-without-id"), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "chain-step-failed")
        assert finding.subject == "generate_test_signal"
        assert finding.evidence["is_error"] is False
        assert finding.evidence["arguments"] == mcp_smoke.TEST_SIGNAL

    def test_an_artifact_written_outside_the_sandbox_fails(self, chain_stub, tmp_path):
        elsewhere = tmp_path / "outside-the-sandbox"
        result = _run(chain_stub("signal-escapes", elsewhere), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "chain-artifact-outside-sandbox")
        assert finding.subject == "generate_test_signal.filepath"
        assert finding.evidence["paths"] == [str(elsewhere / "escaped.csv")]
        assert set(result.facts["chain"].values()) == {"ok"}

    def test_an_artifact_that_does_not_exist_fails(self, chain_stub):
        result = _run(chain_stub("report-missing"), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "chain-artifact-outside-sandbox")
        assert finding.subject == "generate_envelope_report.file_path"
        assert finding.evidence["paths"][0].endswith("stub-report.html")


class TestSandboxEscape:
    def test_a_write_into_the_checkout_fails_naming_the_file(
        self, chain_stub, tmp_path, monkeypatch
    ):
        """A stand-in checkout, so the real one is never written: the harness
        lists its data/ and reports/ before and after the run."""
        checkout = tmp_path / "checkout"
        for tree in mcp_smoke.CHECKOUT_TREES:
            (checkout / tree).mkdir(parents=True)
        (checkout / "pyproject.toml").write_text("", encoding="utf-8")
        monkeypatch.setattr(mcp_smoke, "REPO_ROOT", checkout)
        result = _run(chain_stub("writes-checkout", checkout), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "checkout-written")
        assert finding.subject == "checkout reports/"
        assert finding.evidence["paths"] == ["+stub-leak.html"]


class TestDiscoveryFailures:
    def test_a_tool_the_inventory_does_not_list_fails(self, chain_stub):
        result = _run(chain_stub("extra-tool"), chain_stub.inventory, eras=("auto",))
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "tool-unexpected")
        assert finding.subject == "rogue_tool"
        assert "rogue_tool" in finding.title

    def test_an_invalid_published_schema_fails(self, chain_stub):
        """``required`` must be a list: the SDK sends the schema as is, and only
        the meta-schema check catches it."""
        result = _run(
            chain_stub("schema-required-not-a-list"),
            chain_stub.inventory,
            eras=("auto",),
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "schema-invalid")
        assert finding.subject == "analyze_envelope inputSchema"
        assert "not a valid JSON Schema 2020-12 document" in finding.evidence["excerpt"]
        assert "required" in finding.evidence["excerpt"]

    def test_a_server_whose_input_schema_is_not_an_object_never_passes(
        self, chain_stub
    ):
        """The SDK's server refuses to send a non-object input schema, so the
        session fails at tools/list (an error) before the schema check could
        report it; TestSchemaCheck covers that rule on wire-shaped data."""
        result = _run(
            chain_stub("schema-not-an-object"), chain_stub.inventory, eras=("auto",)
        )
        assert (result.status, result.exit_code) == ("error", 2)
        (finding,) = _rule(result, "session-error")
        assert finding.subject == "auto era tools/list"


class TestProtocolVersion:
    """No real server can make the SDK client accept a legacy handshake
    outside the SDK's own handshake set, so the set is the synthetic input."""

    def test_a_version_outside_the_handshake_set_fails(self, chain_stub, monkeypatch):
        monkeypatch.setattr(
            mcp_smoke, "sdk_protocol_versions", lambda: {"handshake": ["1999-01-01"]}
        )
        result = _run(chain_stub("clean"), chain_stub.inventory)
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "protocol-version-unexpected")
        versions = finding.evidence["versions"]
        assert versions["handshake"] == ["1999-01-01"]
        assert versions["negotiated"] == result.facts["negotiated_protocol"]["legacy"]

    def test_an_sdk_without_a_handshake_list_is_recorded_as_not_checked(
        self, chain_stub, monkeypatch
    ):
        monkeypatch.setattr(mcp_smoke, "sdk_protocol_versions", lambda: {})
        result = _run(chain_stub("clean"), chain_stub.inventory)
        (finding,) = _rule(result, "protocol-version-unverified")
        assert finding.classification == "not-checked"
        assert finding.evidence["versions"]["negotiated"]
        # An observation, not a verdict: the run still passes.
        assert result.status == "pass"


class TestInstallationProbe:
    """The import probe runs a fake package from a temporary directory, so the
    location and the version are fixed by the test, not by this environment."""

    VERSION = "9.9.9-stub"

    @pytest.fixture
    def fake_package(self, tmp_path):
        root = tmp_path / "fake-site"
        package = root / mcp_smoke.PACKAGE
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"__version__ = {self.VERSION!r}\n", encoding="utf-8"
        )
        return root

    def _probe(self, chain_stub, prelude, **config):
        return _run(
            chain_stub("clean"),
            chain_stub.inventory,
            eras=("auto",),
            python=sys.executable,
            probe_prelude=prelude,
            **config,
        )

    def _prelude(self, root):
        return f"import sys; sys.path.insert(0, {str(root)!r})\n"

    def test_the_fake_package_passes_when_the_expectations_hold(
        self, chain_stub, fake_package
    ):
        result = self._probe(
            chain_stub,
            self._prelude(fake_package),
            expect_installed_under=fake_package,
            expect_version=self.VERSION,
        )
        assert result.status == "pass"
        assert Path(result.facts["import_location"]).parent.parent == fake_package

    def test_an_import_from_outside_the_install_prefix_fails(
        self, chain_stub, fake_package, tmp_path
    ):
        prefix = tmp_path / "expected-prefix"
        result = self._probe(
            chain_stub, self._prelude(fake_package), expect_installed_under=prefix
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "import-location-unexpected")
        location, expected = finding.evidence["paths"]
        assert Path(location).parent.parent == fake_package
        assert expected == str(prefix)

    def test_a_version_other_than_the_expected_one_fails(
        self, chain_stub, fake_package
    ):
        result = self._probe(
            chain_stub, self._prelude(fake_package), expect_version="0.13.0"
        )
        assert (result.status, result.exit_code) == ("fail", 1)
        (finding,) = _rule(result, "package-version-mismatch")
        assert finding.evidence["versions"] == {
            "expected": "0.13.0",
            "found": self.VERSION,
        }

    def test_a_package_that_cannot_be_imported_is_an_error(self, chain_stub):
        blocked = f"import sys; sys.modules[{mcp_smoke.PACKAGE!r}] = None\n"
        result = self._probe(chain_stub, blocked, expect_version="0.13.0")
        assert (result.status, result.exit_code) == ("error", 2)
        (finding,) = _rule(result, "import-probe-failed")
        assert "ModuleNotFoundError" in finding.evidence["excerpt"]
        assert finding.evidence["command"][0] == sys.executable

    def test_a_stated_expectation_without_an_interpreter_is_an_error(self, chain_stub):
        """The stub runs as a script, so the probe has no interpreter to use."""
        result = _run(
            chain_stub("clean"),
            chain_stub.inventory,
            eras=("auto",),
            expect_version="0.13.0",
        )
        assert (result.status, result.exit_code) == ("error", 2)
        (finding,) = _rule(result, "import-probe-not-run")
        assert finding.classification == "not-checked"
        assert "--python" in finding.title
