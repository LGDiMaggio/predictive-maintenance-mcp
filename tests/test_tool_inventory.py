"""Characterization test: the registered MCP inventory is frozen.

The U6 refactor converted every tool from a closure inside ``register(mcp)``
to a module-level function registered by reference. The conversion must be
protocol-invisible: tool names, input schemas, async-ness, context wiring,
resource URIs, and prompt argument lists must match the snapshot taken
BEFORE the conversion (tests/fixtures/tool_inventory.json).

Output models are deliberately NOT frozen here: the U6 error-contract work
intentionally replaces error-shaped dict returns with raises/typed results,
and U9 consolidates the surface. When a unit intentionally changes tool
signatures, regenerate the fixture on purpose (see the snapshot recipe
below) — never edit it by hand.

Fixture history:
- U6: original pre-conversion snapshot (closure -> module-level parity).
- U8: regenerated intentionally — every analysis/diagnosis/report/
  prognostics tool switched its handle to signal_id (filename-era
  parameters removed) and load_signal gained batch + overwrite. Tool/
  resource/prompt COUNTS were unchanged (46/4/4).
- U9a: regenerated intentionally — the four MAJOR merges landed:
  assess_severity (<- evaluate_iso_20816 + assess_vibration_severity +
  check_vibration_alert + check_custom_vibration_alert), analyze_envelope
  (<- compute_envelope_spectrum_tool), check_bearing_faults
  (<- check_bearing_fault_peak_tool + check_bearing_faults_direct +
  lookup_bearing_and_compute_tool), analyze_signal_trend
  (<- detect_signal_degradation_onset). Counts: 46 -> 39 tools;
  resources/prompts unchanged (4/4). U9b brings the minor merges,
  resource/prompt drops, and the 33/0/3 target surface.

Snapshot recipe (run from the repo root):
    python -c "from tests.test_tool_inventory import build_inventory, \\
               FIXTURE; import json; FIXTURE.write_text(\\
               json.dumps(build_inventory(), indent=2, sort_keys=True))"
"""

import json
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from predictive_maintenance_mcp.mcp_tools import register_all

FIXTURE = Path(__file__).parent / "fixtures" / "tool_inventory.json"


def build_inventory() -> dict:
    """Introspect a freshly registered server into a comparable dict."""
    mcp = FastMCP("inventory-characterization")
    register_all(mcp)
    return {
        "tools": {
            t.name: {
                "parameters": t.parameters,
                "is_async": t.is_async,
                "context_kwarg": t.context_kwarg,
            }
            for t in mcp._tool_manager._tools.values()
        },
        "resources": sorted(
            [str(uri) for uri in mcp._resource_manager._resources.keys()]
            + [
                str(t.uri_template)
                for t in mcp._resource_manager._templates.values()
            ]
        ),
        "prompts": {
            p.name: sorted(a.name for a in (p.arguments or []))
            for p in mcp._prompt_manager._prompts.values()
        },
    }


@pytest.fixture(scope="module")
def inventory() -> dict:
    return build_inventory()


@pytest.fixture(scope="module")
def reference() -> dict:
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


class TestInventoryCharacterization:
    """Registered surface == pre-conversion snapshot, member by member."""

    def test_tool_names_identical(self, inventory, reference):
        assert sorted(inventory["tools"]) == sorted(reference["tools"])

    def test_tool_schemas_identical(self, inventory, reference):
        for name, ref_tool in reference["tools"].items():
            assert inventory["tools"][name] == ref_tool, (
                f"Tool '{name}' drifted from the frozen inventory "
                f"(input schema / async-ness / ctx wiring changed)"
            )

    def test_resources_identical(self, inventory, reference):
        assert inventory["resources"] == reference["resources"]

    def test_prompts_identical(self, inventory, reference):
        assert inventory["prompts"] == reference["prompts"]

    def test_expected_counts(self, inventory):
        """47 endpoints after U9a: 39 tools + 4 resources + 4 prompts.

        U9a executed the four major merges (46 - 9 old + 2 new = 39
        tools). U9b consolidates further to the target surface of
        33 tools + 0 resources + 3 prompts.
        """
        assert len(inventory["tools"]) == 39
        assert len(inventory["resources"]) == 4
        assert len(inventory["prompts"]) == 4

    def test_merged_tools_absent_new_tools_present(self, inventory):
        """U9a clean cut: absorbed tool names gone, unified names present."""
        absorbed = {
            "evaluate_iso_20816",
            "assess_vibration_severity",
            "check_vibration_alert",
            "check_custom_vibration_alert",
            "compute_envelope_spectrum_tool",
            "check_bearing_fault_peak_tool",
            "check_bearing_faults_direct",
            "lookup_bearing_and_compute_tool",
            "detect_signal_degradation_onset",
        }
        assert absorbed.isdisjoint(inventory["tools"])
        for unified in (
            "assess_severity",
            "analyze_envelope",
            "check_bearing_faults",
            "analyze_signal_trend",
        ):
            assert unified in inventory["tools"]
