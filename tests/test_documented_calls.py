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
  docs (plugin, README.md, docs/TOOL_CATALOG.md, rendered prompts) — not
  even as prose.
- ``docs/TOOL_CATALOG.md`` (the catalog migrated out of the README in U2)
  gets the same call-shape sweep PLUS a name-parity check: the set of
  backticked names in its table rows must equal the registered surface
  exactly, so the catalog can neither list a phantom endpoint nor silently
  omit a real one.
- ``docs/ADAPTER_GUIDE.md`` (U3) gets the same call-shape and retired-names
  sweeps PLUS a declaration-parity check: the parameter names in its marked
  declaration table must equal the raw-declaration surface the code exports
  (``RAW_PARAM_DEFAULTS`` plus the required ``sample_format``/
  ``sampling_rate`` and the unit declaration), including allowed-value
  vocabularies (set-equality, per table cell) and documented defaults (in
  the 'Default' cell specifically) — both directions, so the guide can
  neither document a parameter the code dropped nor omit one it gained.

If this test fails after an intentional signature change, fix the docs, not
the guard: the docs are a public API surface.
"""

import re
from pathlib import Path
from typing import Optional

import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.mcp_tools import prompts as prompt_templates
from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.signal_acquisition.loaders import (
    RAW_PARAM_DEFAULTS,
    VALID_BYTE_ORDERS,
    VALID_SAMPLE_FORMATS,
)
from predictive_maintenance_mcp.signal_acquisition.repository import VALID_SIGNAL_UNITS

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugin"
README = REPO_ROOT / "README.md"
TOOL_CATALOG = REPO_ROOT / "docs" / "TOOL_CATALOG.md"
ADAPTER_GUIDE = REPO_ROOT / "docs" / "ADAPTER_GUIDE.md"

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
# docs/TOOL_CATALOG.md: the migrated catalog stays executable and complete
# ---------------------------------------------------------------------------

#: Any backticked-name markdown table row: a backticked snake_case
#: identifier as the FIRST cell of the row. Shared by the tool-catalog
#: parity check and the adapter-guide declaration-table parser.
TABLE_ROW_NAME_RE = re.compile(r"^\|\s*`([a-z][a-z0-9_]*)`\s*\|", re.M)


def catalog_parity_violations(text: str, endpoints: dict[str, set[str]]) -> list[str]:
    """Name parity between catalog *text* and *endpoints*, both directions.

    Pure over its inputs so it can be mutation-tested on small fixtures.
    One violation string each for: duplicate table rows, table-row names
    that are not registered endpoints, and registered endpoints without a
    table row (the catalog is the COMPLETE inventory it claims to be).
    """
    listed = TABLE_ROW_NAME_RE.findall(text)
    violations: list[str] = []
    duplicated = sorted({name for name in listed if listed.count(name) > 1})
    if duplicated:
        violations.append(f"duplicate catalog rows: {duplicated}")
    unknown = sorted(set(listed) - set(endpoints))
    if unknown:
        violations.append(f"lists names that are not registered endpoints: {unknown}")
    missing = sorted(set(endpoints) - set(listed))
    if missing:
        violations.append(f"omits registered endpoints: {missing}")
    return violations


class TestToolCatalogDocumentedCalls:
    def test_all_calls_executable(self, endpoints):
        text = TOOL_CATALOG.read_text(encoding="utf-8")
        violations = validate_documented_calls(text, endpoints, "docs/TOOL_CATALOG.md")
        assert violations == [], "\n".join(violations)

    def test_catalog_rows_are_exactly_the_registered_surface(self, endpoints):
        """Name parity, both directions: every table-row name is a real
        registered endpoint (no phantom or retired names) and every
        registered endpoint has a row (the catalog is the COMPLETE
        inventory it claims to be)."""
        violations = catalog_parity_violations(
            TOOL_CATALOG.read_text(encoding="utf-8"), endpoints
        )
        assert violations == [], "docs/TOOL_CATALOG.md: " + "; ".join(violations)

    def test_parity_check_actually_parses_rows(self):
        """Anti-rot: a table-format change that stopped the row regex from
        matching would make the parity test pass vacuously."""
        listed = TABLE_ROW_NAME_RE.findall(TOOL_CATALOG.read_text(encoding="utf-8"))
        assert len(listed) >= 30, (
            f"only {len(listed)} catalog rows parsed — the table format "
            f"escaped TABLE_ROW_NAME_RE"
        )


class TestCatalogParityMutations:
    """Executable proof :func:`catalog_parity_violations` catches each
    drift class (and stays green on clean input) — small-fixture style,
    like the landing-page scanner mutations."""

    _ENDPOINTS: dict[str, set[str]] = {"analyze_fft": set(), "load_signal": set()}
    _CLEAN = (
        "| Tool | Purpose |\n"
        "|------|---------|\n"
        "| `analyze_fft` | Spectrum analysis |\n"
        "| `load_signal` | Load a signal |\n"
    )

    def test_clean_fixture_is_green(self):
        assert catalog_parity_violations(self._CLEAN, self._ENDPOINTS) == []

    def test_phantom_name_goes_red(self):
        mutated = self._CLEAN + "| `plot_spectrum` | Retired name |\n"
        violations = catalog_parity_violations(mutated, self._ENDPOINTS)
        assert any("plot_spectrum" in v for v in violations), violations

    def test_omitted_endpoint_goes_red(self):
        mutated = self._CLEAN.replace("| `load_signal` | Load a signal |\n", "")
        violations = catalog_parity_violations(mutated, self._ENDPOINTS)
        assert any("load_signal" in v and "omits" in v for v in violations), violations

    def test_duplicate_row_goes_red(self):
        mutated = self._CLEAN + "| `analyze_fft` | Duplicate row |\n"
        violations = catalog_parity_violations(mutated, self._ENDPOINTS)
        assert any(
            "duplicate" in v.lower() and "analyze_fft" in v for v in violations
        ), violations


# ---------------------------------------------------------------------------
# docs/ADAPTER_GUIDE.md: the documented declaration surface cannot drift
# ---------------------------------------------------------------------------

#: Markers delimiting the guide's declaration-parameter table. The table is
#: parsed ONLY between these, so other tables in the guide (formats, merge
#: precedence) can never pollute the parity check.
ADAPTER_DECL_START = "<!-- adapter-declaration:start -->"
ADAPTER_DECL_END = "<!-- adapter-declaration:end -->"


def expected_adapter_declaration_params() -> set[str]:
    """The declaration surface an adapter targets, derived from code — never
    a hand-maintained copy: the two REQUIRED raw declarations (absent from
    RAW_PARAM_DEFAULTS by design — they have no default), the unit
    declaration ISO severity verdicts require, and every optional raw
    parameter with a documented default."""
    return {"sample_format", "sampling_rate", "signal_unit", *RAW_PARAM_DEFAULTS}


#: A backticked value inside a table cell.
_BACKTICKED_RE = re.compile(r"`([^`]+)`")


def _row_cells(row: str) -> list[str]:
    """A markdown table row's cells, in order, outer pipes stripped."""
    return [c.strip() for c in row.strip().strip("|").split("|")]


def adapter_declaration_rows(text: str) -> Optional[dict[str, list[str]]]:
    """Parse ``{parameter name: [full table row, ...]}`` from the marked table.

    EVERY row for a name is kept (not last-wins) so the caller can flag
    duplicates instead of letting a stale duplicate silently shadow — or
    be shadowed by — the correct row.

    Returns None when the markers are missing or misordered — the caller
    treats that as a violation, so deleting a marker cannot silently
    disable the guard.
    """
    start = text.find(ADAPTER_DECL_START)
    end = text.find(ADAPTER_DECL_END)
    if start == -1 or end == -1 or end < start:
        return None
    rows: dict[str, list[str]] = {}
    for line in text[start:end].splitlines():
        m = TABLE_ROW_NAME_RE.match(line)
        if m:
            rows.setdefault(m.group(1), []).append(line)
    return rows


def _adapter_table_header(text: str) -> Optional[list[str]]:
    """Lowercased header cells of the marked declaration table (the first
    ``|``-row between the markers), or None when it cannot be found."""
    start = text.find(ADAPTER_DECL_START)
    end = text.find(ADAPTER_DECL_END)
    if start == -1 or end == -1 or end < start:
        return None
    for line in text[start:end].splitlines():
        if line.lstrip().startswith("|"):
            return [c.lower() for c in _row_cells(line)]
    return None


def validate_adapter_declaration_table(text: str) -> list[str]:
    """One violation string per way the guide's declaration table can drift.

    Checks, all derived from the code's own exports and CELL-anchored
    (each row is split on ``|`` and every check reads the column the
    header names, so a value in the wrong column cannot satisfy it):
    - name parity BOTH ways against :func:`expected_adapter_declaration_params`,
      with duplicate rows reported by name;
    - closed vocabularies (VALID_SAMPLE_FORMATS / VALID_BYTE_ORDERS /
      VALID_SIGNAL_UNITS) via SET EQUALITY on the backticked values in the
      'Allowed values' cell — a missing value and a phantom value both go
      red;
    - each RAW_PARAM_DEFAULTS default shown backticked in the 'Default'
      cell specifically (None defaults must say 'none' there);
    - the two required parameters marked required in the 'Scope' cell,
      negation-aware: 'not required' or 'optional' is a violation.
    """
    rows = adapter_declaration_rows(text)
    if rows is None:
        return [
            "docs/ADAPTER_GUIDE.md: adapter-declaration markers missing or "
            f"misordered — the declaration table must sit between "
            f"'{ADAPTER_DECL_START}' and '{ADAPTER_DECL_END}'"
        ]
    violations = []
    duplicated = sorted(name for name, lines in rows.items() if len(lines) > 1)
    if duplicated:
        violations.append(
            f"docs/ADAPTER_GUIDE.md: duplicate declaration row(s) for "
            f"{duplicated} — each parameter must have exactly one row"
        )
    expected = expected_adapter_declaration_params()
    phantom = sorted(set(rows) - expected)
    missing = sorted(expected - set(rows))
    if phantom:
        violations.append(
            f"docs/ADAPTER_GUIDE.md documents parameter(s) that are not part "
            f"of the code's raw-declaration surface: {phantom}"
        )
    if missing:
        violations.append(
            f"docs/ADAPTER_GUIDE.md omits declaration parameter(s) the code "
            f"exports: {missing}"
        )

    header = _adapter_table_header(text)
    columns: dict[str, int] = {}
    if header is None:
        violations.append(
            "docs/ADAPTER_GUIDE.md: no header row found in the "
            "adapter-declaration table"
        )
    else:
        for label in ("scope", "allowed values", "default"):
            if label in header:
                columns[label] = header.index(label)
            else:
                violations.append(
                    f"docs/ADAPTER_GUIDE.md: the declaration table header "
                    f"lacks a '{label}' column — the cell-anchored checks "
                    f"need it"
                )

    def cell(name: str, label: str) -> Optional[str]:
        """The named column's cell of *name*'s (last) row, or None when the
        row or column is unavailable (each already reported above)."""
        if label not in columns or name not in rows:
            return None
        cells = _row_cells(rows[name][-1])
        return cells[columns[label]] if columns[label] < len(cells) else ""

    vocabularies = {
        "sample_format": VALID_SAMPLE_FORMATS,
        "byte_order": VALID_BYTE_ORDERS,
        "signal_unit": VALID_SIGNAL_UNITS,
    }
    for name, vocabulary in vocabularies.items():
        allowed_cell = cell(name, "allowed values")
        if allowed_cell is None:
            continue
        documented = set(_BACKTICKED_RE.findall(allowed_cell))
        absent = sorted(set(vocabulary) - documented)
        extra = sorted(documented - set(vocabulary))
        if absent:
            violations.append(
                f"docs/ADAPTER_GUIDE.md: 'Allowed values' cell for '{name}' "
                f"is missing value(s) {absent} (each must appear backticked)"
            )
        if extra:
            violations.append(
                f"docs/ADAPTER_GUIDE.md: 'Allowed values' cell for '{name}' "
                f"lists value(s) {extra} the code does not accept"
            )
    for name, default in RAW_PARAM_DEFAULTS.items():
        default_cell = cell(name, "default")
        if default_cell is None:
            continue
        if default is None:
            if "none" not in default_cell.lower():
                violations.append(
                    f"docs/ADAPTER_GUIDE.md: 'Default' cell for '{name}' "
                    f"must state its default is none"
                )
        elif f"`{default}`" not in default_cell:
            violations.append(
                f"docs/ADAPTER_GUIDE.md: 'Default' cell for '{name}' does "
                f"not show the documented default `{default}`"
            )
    for name in ("sample_format", "sampling_rate"):
        scope_cell = cell(name, "scope")
        if scope_cell is None:
            continue
        lowered = scope_cell.lower()
        if (
            "not required" in lowered
            or "optional" in lowered
            or "required" not in lowered
        ):
            violations.append(
                f"docs/ADAPTER_GUIDE.md: 'Scope' cell for '{name}' must "
                f"state — without negation — that it is required (it has no "
                f"default by design)"
            )
    return violations


class TestAdapterGuideDeclarationParams:
    def test_expected_set_is_on_the_load_signal_surface(self, endpoints):
        """The code-derived declaration set must itself exist on the
        registered load_signal tool — ties the loaders module's exports to
        the MCP surface, so a rename on either side goes red."""
        extra = sorted(expected_adapter_declaration_params() - endpoints["load_signal"])
        assert extra == [], (
            f"declaration parameter(s) {extra} are not parameters of the "
            f"registered load_signal tool"
        )

    def test_guide_table_matches_the_code(self):
        violations = validate_adapter_declaration_table(
            ADAPTER_GUIDE.read_text(encoding="utf-8")
        )
        assert violations == [], "\n".join(violations)

    def test_parity_check_actually_parses_rows(self):
        """Anti-rot: a table-format change that stopped the row regex from
        matching would surface as mass 'omits' violations, but assert the
        parse directly too so the failure names the real cause."""
        rows = adapter_declaration_rows(ADAPTER_GUIDE.read_text(encoding="utf-8"))
        assert rows is not None, "adapter-declaration markers not found"
        assert len(rows) >= 8, (
            f"only {len(rows)} declaration rows parsed — the table format "
            f"escaped TABLE_ROW_NAME_RE"
        )

    # --- mutation tests: the checker really goes red on drifted text ---

    def test_mutation_renamed_param_goes_red(self):
        """Renaming a documented parameter (code rename not mirrored, or
        guide typo) is flagged in BOTH directions."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            "`header_offset`", "`hdr_offset`"
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("hdr_offset" in v for v in violations), violations
        assert any("header_offset" in v for v in violations), violations

    def test_mutation_phantom_param_goes_red(self):
        """A documented parameter the code does not export is flagged."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            ADAPTER_DECL_END,
            "| `endianness` | Raw files | `little`, `big` | `little` |\n\n"
            + ADAPTER_DECL_END,
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("endianness" in v for v in violations), violations

    def test_mutation_dropped_vocabulary_value_goes_red(self):
        """A sample format missing from the guide's allowed values is
        flagged (the vocabularies are checked, not just the names)."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            "`int32`", "`int64`"
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("int32" in v for v in violations), violations

    def test_mutation_stale_default_goes_red(self):
        """A default that no longer matches RAW_PARAM_DEFAULTS is flagged."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            "`little`", "`middle`"
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("byte_order" in v for v in violations), violations

    def test_mutation_duplicate_row_goes_red(self):
        """A stale duplicate row placed BEFORE the correct one is flagged —
        last-wins parsing used to let the correct row silently mask it."""
        text = ADAPTER_GUIDE.read_text(encoding="utf-8")
        row = next(
            line for line in text.splitlines() if line.startswith("| `byte_order`")
        )
        stale = "| `byte_order` | Raw files | `little`, `big` | `big` |"
        mutated = text.replace(row, stale + "\n" + row)
        violations = validate_adapter_declaration_table(mutated)
        assert any(
            "duplicate" in v.lower() and "byte_order" in v for v in violations
        ), violations

    def test_mutation_phantom_vocabulary_value_goes_red(self):
        """A value the code does NOT accept, added to an allowed-values
        cell, is flagged — the cell must EQUAL the code vocabulary, not
        merely contain it."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            "`int32`", "`int32`, `int64`"
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("int64" in v for v in violations), violations

    def test_mutation_default_outside_default_cell_goes_red(self):
        """Cell anchoring: the documented default moved OUT of the 'Default'
        cell — but still present backticked elsewhere in the same row — is
        flagged (a whole-row substring check would wrongly pass it)."""
        text = ADAPTER_GUIDE.read_text(encoding="utf-8")
        row = next(
            line for line in text.splitlines() if line.startswith("| `byte_order`")
        )
        mutated_row = (
            "| `byte_order` | Raw files (default `little`) | `little`, `big` | — |"
        )
        mutated = text.replace(row, mutated_row)
        violations = validate_adapter_declaration_table(mutated)
        assert any(
            "byte_order" in v and "`little`" in v for v in violations
        ), violations

    def test_mutation_not_required_wording_goes_red(self):
        """Negation awareness: 'not required' contains 'required', so a
        plain substring check would wrongly pass it."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            "**required**", "**not required**"
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any(
            "sample_format" in v and "required" in v for v in violations
        ), violations
        assert any(
            "sampling_rate" in v and "required" in v for v in violations
        ), violations

    def test_mutation_removed_marker_goes_red(self):
        """Deleting a marker cannot silently disable the guard."""
        mutated = ADAPTER_GUIDE.read_text(encoding="utf-8").replace(
            ADAPTER_DECL_START, ""
        )
        violations = validate_adapter_declaration_table(mutated)
        assert any("marker" in v for v in violations), violations


class TestAdapterGuideDocumentedCalls:
    def test_all_calls_executable(self, endpoints):
        text = ADAPTER_GUIDE.read_text(encoding="utf-8")
        violations = validate_documented_calls(text, endpoints, "docs/ADAPTER_GUIDE.md")
        assert violations == [], "\n".join(violations)

    def test_guide_contains_calls(self):
        """Anti-rot: the guide's worked example DOES document real calls
        (protects against a rewrite that silently drops them)."""
        calls = extract_calls(ADAPTER_GUIDE.read_text(encoding="utf-8"))
        assert len(calls) >= 3, f"only {len(calls)} documented calls found"


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
    texts["docs/TOOL_CATALOG.md"] = TOOL_CATALOG.read_text(encoding="utf-8")
    texts["docs/ADAPTER_GUIDE.md"] = ADAPTER_GUIDE.read_text(encoding="utf-8")
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
