"""The capability ledger stays closed over the registered MCP surface.

``tests/fixtures/capability_claims.json`` holds one entry per registered tool
and prompt. A new tool or prompt fails here until it has an entry, and an
entry whose name is no longer registered fails too, so every capability that
reaches a client has a recorded release state and, once decided, the kind of
statement it may support (``claim_class``, null until then).

``tests/fixtures/released_surface_v0.13.0.json`` freezes the surface of the
last published release: tool and prompt names, input parameter names and
output property names (top level and ``$defs``). The ledger's ``release``
field is checked against it, and other semantic tests use it to find names a
client of the published server does not have.

Regenerating the released surface after a new release, from the repository
root, with the project's environment (replace the tag):

    git archive --format=tar v0.13.0 src | tar -x -C <export_dir>

then run, with ``TAG``, ``EXPORT`` and ``OUT`` set::

    import importlib.util, json, os, subprocess, sys, tempfile
    os.environ["PDM_PROJECT_DIR"] = tempfile.mkdtemp()
    src = os.path.join(EXPORT, "src")
    spec = importlib.util.spec_from_file_location(
        "predictive_maintenance_mcp", os.path.join(src, "__init__.py"),
        submodule_search_locations=[src])
    module = importlib.util.module_from_spec(spec)
    sys.modules["predictive_maintenance_mcp"] = module
    spec.loader.exec_module(module)
    from mcp.server.mcpserver import MCPServer
    from predictive_maintenance_mcp.mcp_tools import register_all
    server = MCPServer("released-surface"); register_all(server)

    def output_properties(schema):
        names = set((schema or {}).get("properties", {}))
        for definition in (schema or {}).get("$defs", {}).values():
            names.update(definition.get("properties", {}))
        return sorted(names)

    tools = {}
    for tool in server._tool_manager._tools.values():
        schema = getattr(tool, "output_schema", None) or getattr(
            getattr(tool, "fn_metadata", None), "output_schema", None)
        tools[tool.name] = {
            "parameters": sorted(tool.parameters.get("properties", {})),
            "output_properties": output_properties(schema)}
    commit = subprocess.run(["git", "rev-parse", TAG + "^{commit}"],
                            capture_output=True, text=True, check=True).stdout.strip()
    fixture = {"tag": TAG, "commit": commit,
               "method": "See the module docstring of "
                         "tests/test_semantic_registry_closure.py.",
               "tools": dict(sorted(tools.items())),
               "prompts": sorted(p.name for p in
                                 server._prompt_manager._prompts.values())}
    json.dump(fixture, open(OUT, "w", encoding="utf-8", newline="\\n"), indent=2)

then regenerate the ledger's ``release`` values against the new fixture.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from _semantic_guards import PLUGIN_DIR, coverage_list, list_differences
from predictive_maintenance_mcp.mcp_tools import register_all

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LEDGER_FILE = FIXTURES / "capability_claims.json"
RELEASED_FILE = FIXTURES / "released_surface_v0.13.0.json"
ENTRY_KEYS = {"kind", "release", "claim_class", "claim_class_decision"}


@functools.lru_cache(maxsize=None)
def registered() -> dict[str, str]:
    """Every registered endpoint as {name: "tool" | "prompt"}."""
    server = MCPServer("semantic-registry-closure")
    register_all(server)
    names = {tool.name: "tool" for tool in server._tool_manager._tools.values()}
    names.update({p.name: "prompt" for p in server._prompt_manager._prompts.values()})
    return names


def ledger() -> dict[str, dict]:
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))["entries"]


def released() -> dict:
    return json.loads(RELEASED_FILE.read_text(encoding="utf-8"))


class TestCapabilityLedger:
    def test_every_registered_name_has_exactly_one_entry(self):
        differences = list_differences(ledger(), registered())
        assert differences == {"missing": [], "unregistered": []}, (
            "capability_claims.json must list every registered tool and prompt and "
            f"nothing else: {differences}"
        )

    def test_entries_record_kind_release_and_claim_class(self):
        for name, entry in ledger().items():
            assert set(entry) == ENTRY_KEYS, f"{name}: keys {sorted(entry)}"
            assert entry["kind"] == registered()[name], name
            assert entry["release"] in {
                "unreleased",
                released()["tag"].lstrip("v"),
            }, name

    def test_release_state_matches_the_released_surface(self):
        surface = released()
        published = set(surface["tools"]) | set(surface["prompts"])
        version = surface["tag"].lstrip("v")
        wrong = sorted(
            name
            for name, entry in ledger().items()
            if (entry["release"] == version) != (name in published)
        )
        assert not wrong, f"release field disagrees with {RELEASED_FILE.name}: {wrong}"


class TestReleasedSurface:
    def test_fixture_records_its_tag_and_commit(self):
        surface = released()
        assert surface["tag"] == "v0.13.0"
        assert re.fullmatch(r"[0-9a-f]{40}", surface["commit"])
        assert surface["tools"] and surface["prompts"]

    def test_every_released_tool_is_still_registered(self):
        removed = sorted(set(released()["tools"]) - set(registered()))
        assert (
            not removed
        ), f"tools of the last release are no longer registered: {removed}"


class TestPublishedToolLists:
    def test_plugin_readme_lists_every_registered_endpoint(self):
        readme = (PLUGIN_DIR / "README.md").read_text(encoding="utf-8")
        differences = list_differences(coverage_list(readme), registered())
        assert differences == {"missing": [], "unregistered": []}, differences
