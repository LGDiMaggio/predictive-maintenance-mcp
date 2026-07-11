"""Characterization test: the registered MCP inventory is frozen.

The U6 refactor converted every tool from a closure inside ``register(mcp)``
to a module-level function registered by reference. The conversion must be
protocol-invisible: tool names, input schemas, async-ness, context wiring,
resource URIs, and prompt argument lists must match the snapshot taken
BEFORE the conversion (tests/fixtures/tool_inventory.json).

Output models are deliberately NOT frozen here: the U6 error-contract work
intentionally replaces error-shaped dict returns with raises/typed results,
and U9 consolidates the surface. When U9 lands, regenerate the fixture on
purpose (see the snapshot recipe below) — never edit it by hand.

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
        """54 endpoints: 46 tools + 4 resources + 4 prompts (audit inventory).

        Updated intentionally in U9 when the surface consolidates to
        33 tools + 3 prompts + 0 resources.
        """
        assert len(inventory["tools"]) == 46
        assert len(inventory["resources"]) == 4
        assert len(inventory["prompts"]) == 4
