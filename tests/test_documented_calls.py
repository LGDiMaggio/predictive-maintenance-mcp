"""U10 CI guard: every documented tool call is executable against the inventory.

Doc/code drift is this repo's #1 recurring bug class (audit 3.1/3.2: the
plugin shipped calls against parameters that never existed; prompts cited a
resource as a tool). This guard makes silent drift impossible:

- It introspects the REAL registered surface (``MCPServer`` + ``register_all``):
  tool names + input-schema parameter names, prompt names + argument names.
- It extracts every call-shaped snippet ``some_tool(kw=..., ...)`` from
  ``plugin/**/*.md`` AND from the RENDERED MCP prompt templates
  (``src/mcp_tools/prompts.py``), then validates each one:
    1. the called name exists on the registered surface (tool or prompt);
    2. every keyword argument is a real parameter of that endpoint;
    3. arguments are keyword-only — positional call examples caused real
       bugs (audit 3.2: ``generate_iso_report`` called positionally put
       ``machine_group`` into ``sampling_rate``). ``...`` placeholders are
       allowed.
- Retired v0.8.x endpoint names must not appear ANYWHERE in the golden-path
  docs (plugin, README.md, rendered prompts) — not even as prose.

If this test fails after an intentional signature change, fix the docs, not
the guard: the docs are a public API surface.
"""

import re
from pathlib import Path

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import prompts as prompt_templates
from predictive_maintenance_mcp.mcp_tools import register_all

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugin"
README = REPO_ROOT / "README.md"

# A documented call: snake_case identifier (>= 1 underscore, all lowercase —
# every registered endpoint matches this) IMMEDIATELY followed by "(" so that
# prose like "frequency (f_rot)" never matches.
CALL_RE = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\(")

KWARG_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)")

#: Argument segments that are documentation placeholders, not arguments.
PLACEHOLDER_ARGS = {"...", "…"}

#: snake_case calls in docs that are NOT MCP endpoints but are allowed
#: (currently none — add here ONLY with a comment justifying the exception).
ALLOWED_NON_ENDPOINT_CALLS: set[str] = set()

#: v0.8.x endpoint names removed by the U9 consolidation (see the OLD_TO_NEW
#: map in tests/test_surface_parity.py) plus names the old plugin invented.
#: None of these may appear in golden-path docs, even as prose.
RETIRED_NAMES = {
    # merged / renamed in U9
    "list_stored_signals",
    "clear_signal",
    "clear_all_signals",
    "compute_envelope_spectrum_tool",
    "evaluate_iso_20816",
    "assess_vibration_severity",
    "check_vibration_alert",
    "check_custom_vibration_alert",
    "check_bearing_fault_peak_tool",
    "check_bearing_faults_direct",
    "lookup_bearing_and_compute_tool",
    "diagnose_vibration_tool",
    "plot_spectrum",
    "plot_envelope",
    "plot_iso_20816_chart",
    "get_report_info",
    "detect_signal_degradation_onset",
    "generate_iso_diagnostic_report",
    # cited by the old plugin but never registered on the modular server
    "list_available_signals",
    "load_and_validate_metadata",
}


# ---------------------------------------------------------------------------
# Inventory introspection + call extraction / validation
# ---------------------------------------------------------------------------


def build_endpoints() -> dict[str, set[str]]:
    """Registered surface as {endpoint name: set of parameter names}."""
    mcp = MCPServer("documented-calls-guard")
    register_all(mcp)
    endpoints = {
        t.name: set(t.parameters.get("properties", {}))
        for t in mcp._tool_manager._tools.values()
    }
    for p in mcp._prompt_manager._prompts.values():
        endpoints[p.name] = {a.name for a in (p.arguments or [])}
    return endpoints


def extract_calls(text: str) -> list[tuple[str, str, int]]:
    """All ``name(args)`` snippets in *text* as (name, args, line_number)."""
    calls = []
    for m in CALL_RE.finditer(text):
        start = m.end()  # index just past "("
        depth, i, quote = 1, start, None
        while i < len(text) and depth:
            ch = text[i]
            if quote:
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            i += 1
        if depth:  # unbalanced to end of text: prose, not a call
            continue
        line = text.count("\n", 0, m.start()) + 1
        calls.append((m.group(1), text[start : i - 1], line))
    return calls


def split_top_level_args(args: str) -> list[str]:
    """Split an argument string on top-level commas (bracket/quote aware)."""
    parts: list[str] = []
    cur: list[str] = []
    depth, quote = 0, None
    for ch in args:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def validate_documented_calls(
    text: str, endpoints: dict[str, set[str]], source: str
) -> list[str]:
    """Return one violation string per non-executable documented call."""
    violations = []
    for name, args, line in extract_calls(text):
        if name in ALLOWED_NON_ENDPOINT_CALLS:
            continue
        if name not in endpoints:
            violations.append(
                f"{source}:{line}: call to unknown endpoint '{name}(...)' — "
                f"not a registered tool or prompt"
            )
            continue
        params = endpoints[name]
        for seg in split_top_level_args(args):
            if seg in PLACEHOLDER_ARGS:
                continue
            m = KWARG_RE.match(seg)
            if not m:
                violations.append(
                    f"{source}:{line}: {name}(...): positional/malformed "
                    f"argument '{seg}' — documented calls must be keyword-only"
                )
                continue
            kwarg = m.group(1)
            if kwarg not in params:
                violations.append(
                    f"{source}:{line}: {name}({kwarg}=...): '{kwarg}' is not "
                    f"a parameter of {name} (valid: {sorted(params)})"
                )
    return violations


def rendered_prompt_texts() -> dict[str, str]:
    """The 3 MCP prompt templates rendered on both argument branches."""
    return {
        "prompts.diagnose_bearing[minimal]": prompt_templates.diagnose_bearing(
            signal_id="sig_demo"
        ),
        "prompts.diagnose_bearing[full]": prompt_templates.diagnose_bearing(
            signal_id="sig_demo",
            sampling_rate=97656.0,
            machine_group=2,
            support_type="rigid",
            rpm=1797.0,
            bpfo=107.4,
            bpfi=162.2,
            bsf=70.6,
            ftf=14.3,
        ),
        "prompts.diagnose_gear[minimal]": prompt_templates.diagnose_gear(
            signal_id="sig_demo"
        ),
        "prompts.diagnose_gear[full]": prompt_templates.diagnose_gear(
            signal_id="sig_demo",
            sampling_rate=10000.0,
            num_teeth=32,
            rpm=1500.0,
        ),
        "prompts.quick_diagnostic_report": prompt_templates.quick_diagnostic_report(
            signal_id="sig_demo"
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def endpoints() -> dict[str, set[str]]:
    return build_endpoints()


def _plugin_markdown_files() -> list[Path]:
    files = sorted(PLUGIN_DIR.rglob("*.md"))
    assert len(files) >= 14, (
        f"Expected the full plugin surface (8 skills + 2 agents + 3 commands "
        f"+ README), found only {len(files)} markdown files in {PLUGIN_DIR}"
    )
    return files


PLUGIN_FILES = _plugin_markdown_files()
PLUGIN_IDS = [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in PLUGIN_FILES]


# ---------------------------------------------------------------------------
# Meta: the validator itself works (a broken validator = a useless guard)
# ---------------------------------------------------------------------------


class TestValidatorMeta:
    def test_flags_nonexistent_parameter(self, endpoints):
        """Fixture skill calling analyze_fft(signal_path=...) is flagged."""
        fixture_skill = (
            "### Step 1\n"
            'Call `analyze_fft(signal_path="data/foo.csv", fs=10000)` to '
            "inspect the spectrum.\n"
        )
        violations = validate_documented_calls(fixture_skill, endpoints, "fixture")
        assert any("signal_path" in v for v in violations), violations
        assert any("fs" in v for v in violations), violations

    def test_flags_unknown_tool(self, endpoints):
        violations = validate_documented_calls(
            "Call `list_available_signals()` first.", endpoints, "fixture"
        )
        assert any("list_available_signals" in v for v in violations), violations

    def test_flags_positional_arguments(self, endpoints):
        """The audit 3.2 bug shape: positional args land in the wrong param."""
        violations = validate_documented_calls(
            'Call `generate_iso_report("my_signal", 2)`.', endpoints, "fixture"
        )
        assert any("keyword-only" in v for v in violations), violations

    def test_accepts_valid_call(self, endpoints):
        violations = validate_documented_calls(
            'Call `analyze_fft(signal_id="sig", max_frequency=2000)`.',
            endpoints,
            "fixture",
        )
        assert violations == []

    def test_accepts_placeholder_and_nested_structures(self, endpoints):
        text = (
            "Call `load_signal(...)` then\n"
            '`check_bearing_faults(signal_id="s", rpm=1480, '
            'frequencies={"GMF": 350.0, "BPFO": 107.4})` and\n'
            '`estimate_rul(signal_ids=["a", "b", "c"], '
            "timestamps=[0, 24, 48], failure_threshold=4.5)`.\n"
        )
        assert validate_documented_calls(text, endpoints, "fixture") == []

    def test_prose_parentheses_do_not_match(self, endpoints):
        """Identifiers followed by a space + paren are prose, not calls."""
        text = "sidebands spaced by the shaft frequency f_rot (n=1..3)"
        assert extract_calls(text) == []


# ---------------------------------------------------------------------------
# Plugin markdown: every documented call is executable
# ---------------------------------------------------------------------------


class TestPluginDocumentedCalls:
    @pytest.mark.parametrize("md_file", PLUGIN_FILES, ids=PLUGIN_IDS)
    def test_all_calls_executable(self, md_file, endpoints):
        text = md_file.read_text(encoding="utf-8")
        source = str(md_file.relative_to(REPO_ROOT)).replace("\\", "/")
        violations = validate_documented_calls(text, endpoints, source)
        assert violations == [], "\n".join(violations)

    def test_parser_actually_bites(self, endpoints):
        """Sanity: the plugin docs DO contain call snippets to validate
        (protects against a regression that silently matches nothing)."""
        total = sum(
            len(extract_calls(p.read_text(encoding="utf-8"))) for p in PLUGIN_FILES
        )
        assert total >= 25, f"only {total} documented calls found across plugin/"


# ---------------------------------------------------------------------------
# MCP prompt templates: every rendered call template is executable
# ---------------------------------------------------------------------------


class TestPromptTemplateCalls:
    @pytest.mark.parametrize("label", sorted(rendered_prompt_texts()))
    def test_all_calls_executable(self, label, endpoints):
        text = rendered_prompt_texts()[label]
        violations = validate_documented_calls(text, endpoints, label)
        assert violations == [], "\n".join(violations)

    @pytest.mark.parametrize("label", sorted(rendered_prompt_texts()))
    def test_prompts_contain_calls(self, label):
        """Each rendered template must actually guide with tool calls."""
        assert len(extract_calls(rendered_prompt_texts()[label])) >= 3

    def test_diagnose_bearing_cites_no_resources_as_tools(self, endpoints):
        """The diagnose_bearing prompt contains only executable calls: no
        resource URIs, no retired names (audit 3.2 regression guard)."""
        for label, text in rendered_prompt_texts().items():
            if not label.startswith("prompts.diagnose_bearing"):
                continue
            assert "signal://" not in text, label
            assert "manual://" not in text, label
            assert validate_documented_calls(text, endpoints, label) == []


# ---------------------------------------------------------------------------
# Retired names: gone from every golden-path doc surface
# ---------------------------------------------------------------------------


def _golden_path_texts() -> dict[str, str]:
    texts = {
        str(p.relative_to(REPO_ROOT)).replace("\\", "/"): p.read_text(encoding="utf-8")
        for p in PLUGIN_FILES
    }
    texts["README.md"] = README.read_text(encoding="utf-8")
    texts.update(rendered_prompt_texts())
    return texts


class TestRetiredNamesAbsent:
    @pytest.mark.parametrize("source", sorted(_golden_path_texts()))
    def test_no_retired_endpoint_names(self, source):
        text = _golden_path_texts()[source]
        found = sorted(
            name for name in RETIRED_NAMES if re.search(rf"\b{re.escape(name)}\b", text)
        )
        assert found == [], f"{source} still references retired names: {found}"

    def test_retired_names_really_are_retired(self, endpoints):
        """If a name in RETIRED_NAMES gets re-registered, the sweep above
        would wrongly ban documenting it — keep the list honest."""
        resurrected = sorted(RETIRED_NAMES & set(endpoints))
        assert resurrected == [], resurrected
