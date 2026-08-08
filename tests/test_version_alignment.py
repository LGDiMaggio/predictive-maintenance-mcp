"""U10 CI guard: version strings and endpoint counts cannot drift.

Version/count drift is a documented recurring failure here (CITATION.cff
forgotten in past bumps; four different stale tool counts across doc
surfaces at once — audit 4.2/4.7). This guard pins:

- ONE package version across ``pyproject.toml``, ``src/__init__.py``,
  ``server.json`` (both version fields), ``CITATION.cff``, and the README
  citation block;
- the plugin manifest version (``plugin/.claude-plugin/plugin.json``)
  == the plugin entry in ``.claude-plugin/marketplace.json``;
- every "N tools / N prompts / N endpoints / N resources" claim in
  ``README.md`` and ``plugin/README.md`` equals the INTROSPECTED registered
  surface (MCPServer + register_all — the single source of truth);
- every "N skills / N agents / N commands" claim equals the number of
  plugin components actually on disk.

U11 release-bump touchpoints (must change together, this test enforces it):
pyproject.toml, src/__init__.py, server.json (x2), CITATION.cff, and the
README.md BibTeX block. The plugin manifests (plugin.json +
marketplace.json) version independently but must match each other.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import register_all

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Version strings
# ---------------------------------------------------------------------------


def collect_package_versions() -> dict[str, str]:
    """Every place the PACKAGE version is declared."""
    versions = {}
    versions["pyproject.toml"] = tomllib.loads(_read("pyproject.toml"))["project"][
        "version"
    ]
    m = re.search(r'^__version__ = "([^"]+)"', _read("src/__init__.py"), re.M)
    assert m, "src/__init__.py: __version__ not found"
    versions["src/__init__.py"] = m.group(1)

    server_json = json.loads(_read("server.json"))
    versions["server.json (top-level)"] = server_json["version"]
    versions["server.json (packages[0])"] = server_json["packages"][0]["version"]

    # CITATION.cff declares the version TWICE: the top-level ``version:`` and
    # the indented ``preferred-citation.version:``. The old ``^version:``
    # anchor only caught the top-level one, so the indented copy silently
    # drifted (it was stuck at 0.8.1 while everything else moved to 0.9.0).
    # Capture BOTH occurrences (leading whitespace allowed) so the guard
    # covers the preferred-citation version too.
    cff_versions = re.findall(r"^\s*version:\s*(\S+)\s*$", _read("CITATION.cff"), re.M)
    assert len(cff_versions) >= 2, (
        f"CITATION.cff: expected both the top-level and preferred-citation "
        f"version fields, found {cff_versions}"
    )
    versions["CITATION.cff (top-level)"] = cff_versions[0]
    versions["CITATION.cff (preferred-citation)"] = cff_versions[1]

    m = re.search(r"version\s*=\s*\{([^}]+)\}", _read("README.md"))
    assert m, "README.md: BibTeX version field not found"
    versions["README.md (BibTeX)"] = m.group(1).strip()
    return versions


class TestPackageVersionAlignment:
    def test_single_package_version_everywhere(self):
        versions = collect_package_versions()
        distinct = sorted(set(versions.values()))
        assert len(distinct) == 1, f"Version strings diverged: {versions}"

    def test_plugin_manifest_versions_match(self):
        plugin_version = json.loads(_read("plugin/.claude-plugin/plugin.json"))[
            "version"
        ]
        marketplace = json.loads(_read(".claude-plugin/marketplace.json"))
        entries = [
            p for p in marketplace["plugins"] if p["name"] == "predictive-maintenance"
        ]
        assert len(entries) == 1, "marketplace.json: plugin entry missing/duplicated"
        assert entries[0]["version"] == plugin_version, (
            f"plugin.json says {plugin_version}, marketplace.json says "
            f"{entries[0]['version']}"
        )


# ---------------------------------------------------------------------------
# Endpoint counts: docs vs the introspected registered surface
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def surface_counts() -> dict[str, int]:
    mcp = MCPServer("version-alignment-guard")
    register_all(mcp)
    n_tools = len(mcp._tool_manager._tools)
    n_resources = len(mcp._resource_manager._resources) + len(
        mcp._resource_manager._templates
    )
    n_prompts = len(mcp._prompt_manager._prompts)
    return {
        "tools": n_tools,
        "resources": n_resources,
        "prompts": n_prompts,
        "endpoints": n_tools + n_resources + n_prompts,
    }


@pytest.fixture(scope="module")
def component_counts() -> dict[str, int]:
    return {
        "skills": len(list((REPO_ROOT / "plugin" / "skills").glob("*/SKILL.md"))),
        "agents": len(list((REPO_ROOT / "plugin" / "agents").glob("*.md"))),
        "commands": len(list((REPO_ROOT / "plugin" / "commands").glob("*.md"))),
    }


#: Numeric claims recognized in the READMEs, per counted kind.
CLAIM_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "tools": [re.compile(r"\b(\d+)\s+(?:specialized\s+)?(?:MCP\s+)?tools\b", re.I)],
    "endpoints": [
        re.compile(r"\b(\d+)\s+(?:specialized\s+)?(?:MCP\s+)?endpoints\b", re.I)
    ],
    "prompts": [re.compile(r"\b(\d+)\s+(?:MCP\s+|guided\s+)?prompts\b", re.I)],
    "resources": [re.compile(r"\b(\d+)\s+(?:MCP\s+)?resources?\b", re.I)],
    "skills": [
        re.compile(r"\b(\d+)\s+(?:diagnostic\s+|domain\s+)?skills\b", re.I),
        re.compile(r"\bSkills\s*\((\d+)\)", re.I),
    ],
    "agents": [
        re.compile(r"\b(\d+)\s+agents\b", re.I),
        re.compile(r"\bAgents\s*\((\d+)\)", re.I),
    ],
    "commands": [
        re.compile(r"\b(\d+)\s+commands\b", re.I),
        re.compile(r"\bCommands\s*\((\d+)\)", re.I),
    ],
}

README_FILES = ["README.md", "plugin/README.md"]


def _claims(text: str, kind: str) -> list[tuple[int, str]]:
    """All numeric claims of *kind* in *text* as (value, matched snippet)."""
    found = []
    for pattern in CLAIM_PATTERNS[kind]:
        for m in pattern.finditer(text):
            found.append((int(m.group(1)), m.group(0)))
    return found


class TestEndpointCountClaims:
    @pytest.mark.parametrize("readme", README_FILES)
    @pytest.mark.parametrize("kind", ["tools", "endpoints", "prompts", "resources"])
    def test_surface_claims_match_introspection(self, readme, kind, surface_counts):
        text = _read(readme)
        expected = surface_counts[kind]
        wrong = [
            f"{readme}: claims '{snippet}' but the registered surface has "
            f"{expected} {kind}"
            for value, snippet in _claims(text, kind)
            if value != expected
        ]
        assert wrong == [], "\n".join(wrong)

    @pytest.mark.parametrize("readme", README_FILES)
    @pytest.mark.parametrize("kind", ["skills", "agents", "commands"])
    def test_component_claims_match_disk(self, readme, kind, component_counts):
        text = _read(readme)
        expected = component_counts[kind]
        wrong = [
            f"{readme}: claims '{snippet}' but plugin/ contains "
            f"{expected} {kind}"
            for value, snippet in _claims(text, kind)
            if value != expected
        ]
        assert wrong == [], "\n".join(wrong)

    @pytest.mark.parametrize("readme", README_FILES)
    def test_readmes_state_the_tool_count(self, readme):
        """Anti-rot: each README must make at least one tool-count claim the
        guard can check (if the wording drifts out of CLAIM_PATTERNS, this
        fails instead of the guard silently checking nothing)."""
        assert _claims(_read(readme), "tools"), (
            f"{readme}: no recognizable 'N tools' claim found — either the "
            f"count was removed or the wording escaped CLAIM_PATTERNS"
        )

    def test_plugin_readme_states_component_counts(self):
        text = _read("plugin/README.md")
        for kind in ("skills", "agents", "commands"):
            assert _claims(text, kind), (
                f"plugin/README.md: no recognizable '{kind}' count claim"
            )

    def test_expected_final_surface(self, surface_counts):
        """The U9 target surface, restated here so a surface change makes
        BOTH the inventory test and the doc guards go red together."""
        assert surface_counts == {
            "tools": 33,
            "resources": 0,
            "prompts": 3,
            "endpoints": 36,
        }
