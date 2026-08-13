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

U1 (distribution plan) widened the guarded surfaces after "52 endpoints /
7 skills / 0.8.0 / 86%" shipped stale on the public funnel:

- the count-claim guard also scans ``pyproject.toml``'s ``description`` and
  ``docs/_config.yml``;
- ``docs/index.html`` gets a dedicated scanner
  (:func:`scan_landing_page_claims`): the landing page carries numbers in
  meta/OG/Twitter descriptions, nested JSON-LD strings, animated
  ``data-target`` counters (where the number and its noun live in different
  HTML elements), and a BibTeX block. The scanner DERIVES every occurrence
  from the text — never from a hardcoded list of N known copies;
- coverage percentages are compared against the CI-enforced minimum
  (``[tool.coverage.report] fail_under``), never a measured snapshot;
- ``.zenodo.json`` must parse as strict JSON (a trailing comma shipped).

U11 release-bump touchpoints (must change together, this test enforces it):
pyproject.toml, src/__init__.py, server.json (x2), CITATION.cff, the
README.md BibTeX block, and docs/index.html (x2: JSON-LD softwareVersion +
BibTeX block). The plugin manifests (plugin.json + marketplace.json)
version independently but must match each other.

U2 (README funnel) widened the guarded surfaces again:

- ``docs/TOOL_CATALOG.md`` (the tool catalog migrated out of the README)
  joins the count-claim scan, with its own anti-rot check — its stated
  totals must equal the introspected surface;
- the ``<!-- mcp-name: ... -->`` marker in README.md must equal the
  ``name`` field of ``server.json``. The MCP registry validates PyPI
  package ownership by finding this marker in the long description (the
  README), and losing it fails ``publish-mcp.yml`` only AFTER the
  immutable PyPI upload — forcing a patch release.
"""

import json
import re
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import register_all

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


#: docs/index.html version markers, shared by the package-version alignment
#: and the landing-page scanner below.
_SOFTWARE_VERSION_RE = re.compile(r'"softwareVersion"\s*:\s*"([^"]+)"')
_BIBTEX_VERSION_RE = re.compile(r"\bversion\s*=\s*\{([^}]+)\}")


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

    m = _BIBTEX_VERSION_RE.search(_read("README.md"))
    assert m, "README.md: BibTeX version field not found"
    versions["README.md (BibTeX)"] = m.group(1).strip()

    # docs/index.html declares the version TWICE: the JSON-LD
    # ``softwareVersion`` and the copyable BibTeX block. Both sat at 0.8.0
    # while the package moved to 0.12.0 — exactly the drift class this
    # module exists for — so both join the alignment set.
    index_html = _read("docs/index.html")
    m = _SOFTWARE_VERSION_RE.search(index_html)
    assert m, "docs/index.html: JSON-LD softwareVersion not found"
    versions["docs/index.html (JSON-LD softwareVersion)"] = m.group(1)
    m = _BIBTEX_VERSION_RE.search(index_html)
    assert m, "docs/index.html: BibTeX version field not found"
    versions["docs/index.html (BibTeX)"] = m.group(1).strip()
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
# MCP registry ownership marker (U2)
# ---------------------------------------------------------------------------

_MCP_NAME_RE = re.compile(r"<!--\s*mcp-name:\s*(\S+)\s*-->")


def mcp_name_marker_violations(readme_text: str, expected_name: str) -> list[str]:
    """Every way the README can break MCP-registry ownership validation.

    Pure over its inputs so each failure mode can be mutation-tested on
    small fixtures: the marker missing entirely; the marker naming a
    different server than *expected_name*; the marker sitting after the
    first '## ' section heading (outside the header block, where a
    section-level restructure is most likely to orphan or delete it).
    """
    m = _MCP_NAME_RE.search(readme_text)
    if m is None:
        return ["<!-- mcp-name: ... --> marker not found"]
    violations: list[str] = []
    if m.group(1) != expected_name:
        violations.append(
            f"mcp-name marker says {m.group(1)!r} but server.json name is "
            f"{expected_name!r}"
        )
    first_section = readme_text.find("\n## ")
    if first_section != -1 and m.start() > first_section:
        violations.append(
            "the mcp-name marker must appear before the first '## ' section " "heading"
        )
    return violations


class TestMcpNameMarker:
    """README must carry the registry ownership marker, equal to server.json.

    The MCP registry validates PyPI-package ownership by finding
    ``<!-- mcp-name: ... -->`` in the package long description (the README).
    Losing or mistyping the marker — e.g. in a README restructure — fails
    ``publish-mcp.yml`` only AFTER the immutable PyPI upload, forcing a
    patch release. This turns the constraint into an executable check.
    """

    def test_readme_marker_is_valid(self):
        expected = json.loads(_read("server.json"))["name"]
        violations = mcp_name_marker_violations(_read("README.md"), expected)
        assert violations == [], "README.md: " + "; ".join(violations)

    def test_readme_still_has_a_section_heading(self):
        """Anti-rot for the header-block rule: it compares the marker's
        position against the first '## ' heading, so a README with no such
        heading would make that check vacuously green."""
        assert "\n## " in _read("README.md"), (
            "README.md has no '## ' section headings — the mcp-name "
            "header-block check has nothing to anchor to"
        )


#: Synthetic README fixture for the marker mutations — deliberately NOT the
#: real README, so these tests exercise the checker, not the repo state.
_MARKER_FIXTURE = (
    "# Predictive Maintenance MCP\n"
    "\n"
    "<!-- mcp-name: io.github.acme/pmm -->\n"
    "\n"
    "Intro paragraph.\n"
    "\n"
    "## Install\n"
    "\n"
    "pip install pmm\n"
)


class TestMcpNameMarkerMutations:
    """Executable proof the marker check catches each failure mode (and
    stays green on clean input)."""

    def test_clean_fixture_is_green(self):
        assert mcp_name_marker_violations(_MARKER_FIXTURE, "io.github.acme/pmm") == []

    def test_missing_marker_goes_red(self):
        mutated = _MARKER_FIXTURE.replace("<!-- mcp-name: io.github.acme/pmm -->\n", "")
        violations = mcp_name_marker_violations(mutated, "io.github.acme/pmm")
        assert any("not found" in v for v in violations), violations

    def test_name_mismatch_goes_red(self):
        violations = mcp_name_marker_violations(_MARKER_FIXTURE, "io.github.acme/other")
        assert any(
            "io.github.acme/pmm" in v and "io.github.acme/other" in v
            for v in violations
        ), violations

    def test_marker_after_first_heading_goes_red(self):
        marker = "<!-- mcp-name: io.github.acme/pmm -->\n"
        mutated = _MARKER_FIXTURE.replace(marker + "\n", "") + "\n" + marker
        violations = mcp_name_marker_violations(mutated, "io.github.acme/pmm")
        assert any("before the first" in v for v in violations), violations


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


#: Numeric claims recognized on the prose surfaces, per counted kind.
#: "coverage" is special: its expected value is the CI-enforced minimum
#: (``fail_under``), not a surface count.
CLAIM_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "tools": [re.compile(r"\b(\d+)\s+(?:specialized\s+)?(?:MCP\s+)?tools\b", re.I)],
    "endpoints": [
        re.compile(r"\b(\d+)\s+(?:specialized\s+)?(?:MCP\s+)?endpoints\b", re.I)
    ],
    "prompts": [re.compile(r"\b(\d+)\s+(?:MCP\s+|guided\s+)?prompts\b", re.I)],
    "resources": [re.compile(r"\b(\d+)\s+(?:MCP\s+)?resources?\b", re.I)],
    "skills": [
        re.compile(r"\b(\d+)\s+(?:diagnostic\s+|domain\s+|Claude\s+)?skills\b", re.I),
        re.compile(r"\bSkills\s*\((\d+)\)", re.I),
    ],
    "agents": [
        re.compile(r"\b(\d+)\s+(?:autonomous\s+)?agents\b", re.I),
        re.compile(r"\bAgents\s*\((\d+)\)", re.I),
    ],
    "commands": [
        re.compile(r"\b(\d+)\s+(?:slash\s+)?commands\b", re.I),
        re.compile(r"\bCommands\s*\((\d+)\)", re.I),
    ],
    "coverage": [
        # "86% test coverage" / "85%+ test coverage"
        re.compile(r"\b(\d+)\s*%\+?\s+test\s+coverage\b", re.I),
        # "test coverage: 85%" / "test coverage of 85%"
        re.compile(r"\btest\s+coverage\b\D{0,15}?(\d+)\s*%", re.I),
    ],
}

README_FILES = ["README.md", "plugin/README.md"]


def _pyproject_description() -> str:
    return tomllib.loads(_read("pyproject.toml"))["project"]["description"]


#: Every public prose surface the count-claim guard scans, mapping a display
#: name (used in test ids and failure messages) to a loader for its text.
#: docs/index.html is NOT here: its numbers hide in data-target attributes
#: and JSON-LD, so it gets the dedicated scanner below.
COUNTED_PROSE_SURFACES: dict[str, Callable[[], str]] = {
    "README.md": lambda: _read("README.md"),
    "plugin/README.md": lambda: _read("plugin/README.md"),
    "pyproject.toml (project.description)": _pyproject_description,
    "docs/_config.yml": lambda: _read("docs/_config.yml"),
    "docs/TOOL_CATALOG.md": lambda: _read("docs/TOOL_CATALOG.md"),
}


@pytest.fixture(scope="module")
def coverage_minimum() -> int:
    """The CI-enforced coverage floor — the only coverage number a public
    surface may state (a measured snapshot always goes stale; 86% shipped
    while CI enforced 85%)."""
    pyproject = tomllib.loads(_read("pyproject.toml"))
    return int(pyproject["tool"]["coverage"]["report"]["fail_under"])


def _claims(text: str, kind: str) -> list[tuple[int, str]]:
    """All numeric claims of *kind* in *text* as (value, matched snippet)."""
    found = []
    for pattern in CLAIM_PATTERNS[kind]:
        for m in pattern.finditer(text):
            found.append((int(m.group(1)), m.group(0)))
    return found


class TestEndpointCountClaims:
    @pytest.mark.parametrize("surface", COUNTED_PROSE_SURFACES)
    @pytest.mark.parametrize("kind", ["tools", "endpoints", "prompts", "resources"])
    def test_surface_claims_match_introspection(self, surface, kind, surface_counts):
        text = COUNTED_PROSE_SURFACES[surface]()
        expected = surface_counts[kind]
        wrong = [
            f"{surface}: claims '{snippet}' but the registered surface has "
            f"{expected} {kind}"
            for value, snippet in _claims(text, kind)
            if value != expected
        ]
        assert wrong == [], "\n".join(wrong)

    @pytest.mark.parametrize("surface", COUNTED_PROSE_SURFACES)
    @pytest.mark.parametrize("kind", ["skills", "agents", "commands"])
    def test_component_claims_match_disk(self, surface, kind, component_counts):
        text = COUNTED_PROSE_SURFACES[surface]()
        expected = component_counts[kind]
        wrong = [
            f"{surface}: claims '{snippet}' but plugin/ contains " f"{expected} {kind}"
            for value, snippet in _claims(text, kind)
            if value != expected
        ]
        assert wrong == [], "\n".join(wrong)

    @pytest.mark.parametrize("surface", COUNTED_PROSE_SURFACES)
    def test_coverage_claims_state_the_ci_minimum(self, surface, coverage_minimum):
        text = COUNTED_PROSE_SURFACES[surface]()
        wrong = [
            f"{surface}: claims '{snippet}' but the CI-enforced coverage "
            f"minimum (fail_under) is {coverage_minimum}"
            for value, snippet in _claims(text, "coverage")
            if value != coverage_minimum
        ]
        assert wrong == [], "\n".join(wrong)

    def test_readme_coverage_claim_recognizable(self):
        """Anti-rot: README states a coverage number on purpose (bound to
        fail_under). If the wording drifts out of CLAIM_PATTERNS, fail here
        instead of the guard silently checking nothing."""
        assert _claims(_read("README.md"), "coverage"), (
            "README.md: no recognizable coverage claim found — either the "
            "number was removed or the wording escaped CLAIM_PATTERNS"
        )

    def test_config_and_pyproject_state_the_endpoint_count(self):
        """Anti-rot for the two newly guarded surfaces: each must keep at
        least one endpoints claim the guard can recognize."""
        for surface in ("pyproject.toml (project.description)", "docs/_config.yml"):
            assert _claims(COUNTED_PROSE_SURFACES[surface](), "endpoints"), (
                f"{surface}: no recognizable 'N endpoints' claim found — "
                f"either the count was removed or the wording escaped "
                f"CLAIM_PATTERNS"
            )

    def test_tool_catalog_states_its_totals(self):
        """Anti-rot for the migrated catalog page (U2): it must keep a
        recognizable total for endpoints, tools AND prompts — the whole
        point of the page is the complete guarded inventory."""
        text = COUNTED_PROSE_SURFACES["docs/TOOL_CATALOG.md"]()
        for kind in ("endpoints", "tools", "prompts"):
            assert _claims(text, kind), (
                f"docs/TOOL_CATALOG.md: no recognizable '{kind}' count claim "
                f"found — either the total was removed or the wording escaped "
                f"CLAIM_PATTERNS"
            )

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
            assert _claims(
                text, kind
            ), f"plugin/README.md: no recognizable '{kind}' count claim"

    def test_expected_final_surface(self, surface_counts):
        """The current surface, restated here so a surface change makes
        BOTH the inventory test and the doc guards go red together.

        v0.10 added generate_diagnostic_report (33 -> 34 tools). The
        additions register lives in tests/test_surface_parity.py."""
        assert surface_counts == {
            "tools": 34,
            "resources": 0,
            "prompts": 3,
            "endpoints": 37,
        }


# ---------------------------------------------------------------------------
# docs/index.html: meta descriptions, JSON-LD, animated counters, BibTeX
# ---------------------------------------------------------------------------

_DATA_TARGET_RE = re.compile(r'data-target="(\d+)"')
#: An animated counter paired with its sibling label — the number lives in a
#: ``data-target`` attribute, the noun in the adjacent ``num-label`` element.
_COUNTER_WITH_LABEL_RE = re.compile(
    r'data-target="(\d+)"[^>]*>[^<]*</div>\s*<div class="num-label">([^<]+)</div>'
)

#: Counter-label substring (lowercased) -> expected-value key. First hit wins.
_COUNTER_LABEL_KINDS: list[tuple[str, str]] = [
    ("endpoint", "endpoints"),
    ("tool", "tools"),
    ("skill", "skills"),
    ("prompt", "prompts"),
    ("agent", "agents"),
    ("command", "commands"),
    ("coverage", "coverage"),
]


def scan_landing_page_claims(text: str, expected: dict[str, int | str]) -> list[str]:
    """Every wrong numeric or version claim in landing-page *text*.

    Occurrence-agnostic by design: claims are DERIVED by scanning the text —
    prose/meta/JSON-LD strings via CLAIM_PATTERNS, animated counters via
    their ``data-target`` attribute plus adjacent label, versions via the
    JSON-LD ``softwareVersion`` and BibTeX markers — never from a hardcoded
    list of N known copies, so adding or removing a copy keeps the guard
    valid. Pure function over text so it can be mutation-tested on small
    fixtures (same shape as the CWRU drift guard).

    *expected* carries the introspected surface counts, plugin component
    counts, the CI coverage minimum under ``"coverage"``, and the package
    version under ``"version"``.
    """
    findings: list[str] = []

    # 1. Text claims where the noun sits next to the number (prose, meta/OG/
    #    Twitter descriptions, nested JSON-LD strings — all just text here).
    for kind in CLAIM_PATTERNS:
        for value, snippet in _claims(text, kind):
            if value != expected[kind]:
                findings.append(
                    f"claims '{snippet}' but the real value is "
                    f"{expected[kind]} ({kind})"
                )

    # 2. Animated counters. Every data-target must be classifiable via its
    #    label — an unlabeled or unrecognized counter is itself a finding,
    #    because it would be a number outside any guard.
    labelled = _COUNTER_WITH_LABEL_RE.findall(text)
    n_targets = len(_DATA_TARGET_RE.findall(text))
    if len(labelled) != n_targets:
        findings.append(
            f"{n_targets} data-target counters but only {len(labelled)} have "
            f"an adjacent num-label — every counter needs a label the guard "
            f"can classify"
        )
    for raw_value, label in labelled:
        kind = next(
            (k for sub, k in _COUNTER_LABEL_KINDS if sub in label.lower()), None
        )
        if kind is None:
            findings.append(
                f'counter data-target="{raw_value}" labelled {label!r} is not '
                f"bound to any introspected value — extend _COUNTER_LABEL_KINDS"
            )
        elif int(raw_value) != expected[kind]:
            findings.append(
                f'counter data-target="{raw_value}" labelled {label!r} should '
                f"be {expected[kind]} ({kind})"
            )

    # 3. Version markers: JSON-LD softwareVersion and the BibTeX block.
    for version in _SOFTWARE_VERSION_RE.findall(text):
        if version != expected["version"]:
            findings.append(
                f'JSON-LD softwareVersion "{version}" != package version '
                f'{expected["version"]}'
            )
    for version in _BIBTEX_VERSION_RE.findall(text):
        if version.strip() != expected["version"]:
            findings.append(
                f"BibTeX version {{{version.strip()}}} != package version "
                f'{expected["version"]}'
            )
    return findings


@pytest.fixture(scope="module")
def landing_expected(surface_counts, component_counts) -> dict[str, int | str]:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    return {
        **surface_counts,
        **component_counts,
        "coverage": int(pyproject["tool"]["coverage"]["report"]["fail_under"]),
        "version": pyproject["project"]["version"],
    }


class TestLandingPageClaims:
    def test_index_html_carries_no_stale_claims(self, landing_expected):
        findings = scan_landing_page_claims(_read("docs/index.html"), landing_expected)
        assert findings == [], "docs/index.html:\n  " + "\n  ".join(findings)

    def test_index_html_remains_scannable(self):
        """Anti-rot: the scanner must keep finding something to check on the
        live page — a restructure that removed all recognizable claims would
        otherwise leave the guard green while checking nothing."""
        text = _read("docs/index.html")
        assert _DATA_TARGET_RE.search(text), "no data-target counters found"
        assert _SOFTWARE_VERSION_RE.search(text), "no JSON-LD softwareVersion"
        assert _claims(text, "endpoints"), "no recognizable endpoints claim"
        assert _claims(text, "skills"), "no recognizable skills claim"


#: Synthetic fixture values — deliberately NOT read from the repo, so these
#: mutation tests exercise the scanner itself, not the repo state.
_FIXTURE_EXPECTED: dict[str, int | str] = {
    "tools": 34,
    "endpoints": 37,
    "prompts": 3,
    "resources": 0,
    "skills": 8,
    "agents": 2,
    "commands": 3,
    "coverage": 85,
    "version": "0.12.0",
}

#: Minimal page exercising every claim carrier the scanner knows: meta
#: description, nested JSON-LD strings + softwareVersion, two labelled
#: counters, plain prose, and a BibTeX block.
_CLEAN_PAGE = """\
<meta name="description" content="Diagnostics via natural language. \
37 MCP endpoints, runs locally.">
<script type="application/ld+json">
{"softwareVersion": "0.12.0",
 "featureList": ["Claude Code plugin with 8 skills"],
 "text": "It provides 37 MCP endpoints for vibration analysis."}
</script>
<div class="num-val" data-target="37">0</div>
<div class="num-label">MCP Endpoints</div>
<div class="num-val" data-target="85">0</div>
<div class="num-label">%+ Test Coverage (CI Minimum)</div>
<p>8 skills, 2 autonomous agents, 3 slash commands</p>
<pre>version = {0.12.0},</pre>
"""


class TestLandingPageScannerMutations:
    """Executable proof the scanner catches each drift class (and stays
    green on clean input) — small-fixture style, like the CWRU drift guard."""

    def test_clean_fixture_is_green(self):
        assert scan_landing_page_claims(_CLEAN_PAGE, _FIXTURE_EXPECTED) == []

    def test_one_stale_occurrence_among_correct_ones_goes_red(self):
        """The edge case that matters: ONE stale copy among several correct
        ones must be flagged, and only that one."""
        mutated = _CLEAN_PAGE.replace("37 MCP endpoints", "52 MCP endpoints", 1)
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert len(findings) == 1, findings
        assert "52 MCP endpoints" in findings[0]

    def test_stale_counter_data_target_goes_red(self):
        mutated = _CLEAN_PAGE.replace('data-target="37"', 'data-target="52"')
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any(
            'data-target="52"' in f and "MCP Endpoints" in f for f in findings
        ), findings

    def test_stale_coverage_counter_goes_red(self):
        mutated = _CLEAN_PAGE.replace('data-target="85"', 'data-target="86"')
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any('data-target="86"' in f for f in findings), findings

    def test_stale_software_version_goes_red(self):
        mutated = _CLEAN_PAGE.replace(
            '"softwareVersion": "0.12.0"', '"softwareVersion": "0.8.0"'
        )
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any("softwareVersion" in f and "0.8.0" in f for f in findings), findings

    def test_stale_bibtex_version_goes_red(self):
        mutated = _CLEAN_PAGE.replace("version = {0.12.0}", "version = {0.8.0}")
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any("BibTeX" in f and "0.8.0" in f for f in findings), findings

    def test_version_bump_without_page_update_goes_red(self):
        """Simulated release bump: expected version moves, page untouched —
        both version carriers (JSON-LD + BibTeX) must be flagged."""
        bumped = {**_FIXTURE_EXPECTED, "version": "0.13.0"}
        findings = scan_landing_page_claims(_CLEAN_PAGE, bumped)
        assert len(findings) == 2, findings

    def test_unrecognized_counter_label_goes_red(self):
        mutated = _CLEAN_PAGE.replace("MCP Endpoints</div>", "GitHub Stars</div>")
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any("GitHub Stars" in f for f in findings), findings

    def test_counter_without_label_goes_red(self):
        mutated = _CLEAN_PAGE.replace('<div class="num-label">MCP Endpoints</div>', "")
        findings = scan_landing_page_claims(mutated, _FIXTURE_EXPECTED)
        assert any("num-label" in f for f in findings), findings

    def test_claim_pattern_wordings(self):
        """The qualifier spellings the public surfaces actually use must
        stay recognizable to CLAIM_PATTERNS."""
        assert _claims("8 Claude skills", "skills") == [(8, "8 Claude skills")]
        assert _claims("2 autonomous agents", "agents") == [(2, "2 autonomous agents")]
        assert _claims("3 slash commands", "commands") == [(3, "3 slash commands")]
        assert _claims("86% test coverage", "coverage") == [(86, "86% test coverage")]
        assert _claims("85%+ test coverage", "coverage")[0][0] == 85
        assert _claims("test coverage: 85%", "coverage")[0][0] == 85


# ---------------------------------------------------------------------------
# .zenodo.json: Zenodo reads this with a strict JSON parser
# ---------------------------------------------------------------------------


class TestZenodoMetadata:
    def test_zenodo_json_parses_as_strict_json(self):
        """A trailing comma once shipped in the keywords array; json.loads
        (strict, like Zenodo's parser) rejects it, so a deposit would fall
        back to default metadata. Strict parse is the guard."""
        try:
            data = json.loads(_read(".zenodo.json"))
        except json.JSONDecodeError as exc:
            pytest.fail(f".zenodo.json is not strict JSON: {exc}")
        keywords = data.get("keywords", [])
        assert keywords and all(
            isinstance(k, str) and k.strip() for k in keywords
        ), f".zenodo.json: empty or malformed keywords entries: {keywords!r}"


# ---------------------------------------------------------------------------
# Lockfile vs manifest: the drift that shipped the mcp 2.x migration
# ---------------------------------------------------------------------------


class TestLockfileAgreesWithManifest:
    """uv.lock must not contradict pyproject.toml's declared requirements.

    The mcp 1.x -> 2.x migration raised the floor to >=2.0.0 and left the
    lockfile resolving 1.16.0. CI and Docker install with pip and never read
    the lock, so nothing went red; the failure was reserved for whoever ran
    the documented `uv sync`. This repo already turns each such incident into
    a permanent guard (see the endpoint-count and package-version checks
    above) — this is that guard for the lockfile.
    """

    def test_lock_records_the_declared_dependency_specifiers(self):
        lock_path = REPO_ROOT / "uv.lock"
        if not lock_path.exists():
            pytest.skip("no uv.lock in this checkout")

        pyproject = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = {}
        for entry in pyproject["project"]["dependencies"]:
            name = re.split(r"[\[><=!~;]", entry, maxsplit=1)[0].strip()
            declared[name.lower()] = entry.strip()

        lock_text = lock_path.read_text(encoding="utf-8")
        mismatches = []
        for name, spec in declared.items():
            # The lock echoes each requirement verbatim in [package.metadata]
            # requires-dist as `specifier = "<constraint>"`.
            constraint = spec[len(spec.split(name)[0]) + len(name) :].strip()
            constraint = constraint.lstrip("]").strip()
            if constraint.startswith("["):
                constraint = constraint.split("]", 1)[-1].strip()
            if not constraint:
                continue
            if f'specifier = "{constraint}"' not in lock_text:
                mismatches.append(f"{name}: pyproject wants {constraint!r}")

        assert mismatches == [], (
            "uv.lock does not record these declared specifiers — run "
            "`uv lock` and commit the result:\n  " + "\n  ".join(mismatches)
        )
