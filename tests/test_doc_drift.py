"""Public documentation must describe the surface that ships, not an older one.

The v0.9 consolidation retired 20-odd endpoint names, and the Python API has
moved since the notebooks and examples were written. ``test_documented_calls``
already keeps the golden-path docs (plugin, README, catalog, adapter guide,
rendered prompts) executable. Everything else a reader can find in this
repository was unguarded: EXAMPLES.md, the architecture page, the notebooks,
the example scripts, the benchmark entry point, the issue templates, and the
server's own ``instructions`` text, which every MCP client hands to its model
verbatim. A stale name there does not fail anything; it teaches a reader, or a
model, to call something that is not there.

Three checks, each a pure function over text so it can be driven by small
synthetic fixtures:

* :func:`retired_name_violations`: the retired v0.8 endpoint names (every key
  of ``OLD_TO_NEW`` in ``tests/test_surface_parity.py`` whose destination is
  not the same name, drops included) do not appear as whole identifiers in
  public content or in the instructions. The table is loaded from that test
  module, never copied, so the two cannot disagree.
* :func:`import_violations`: ``from predictive_maintenance_mcp... import ...``
  and ``import predictive_maintenance_mcp...`` statements in markdown code
  blocks, notebook code cells, example scripts and benchmark scripts resolve:
  the module imports, and each name is an attribute or submodule of it. The
  package is really imported; nothing is inferred from file names.
* :func:`instruction_violations`: every call-shaped or backticked snake_case
  identifier in the server ``instructions`` is a registered tool or prompt.

The scan reads public, tracked content only, by explicit include globs
(:data:`INCLUDE_GLOBS`) minus the internal paths the ``.gitignore`` keeps
local (:func:`is_public`), which are never opened.

Today's violations are listed in :data:`ALLOWLIST` with a reason each, and the
comparison runs both ways, as in ``tests/test_ci_gates.py``: new drift fails,
and an entry whose drift was fixed fails until it is removed. A guard that
stopped finding anything would look like a clean tree, so
:func:`test_the_scan_actually_read_public_content` establishes that real
files, imports and notebook cells were parsed before "found nothing" is
believed.
"""

import asyncio
import fnmatch
import functools
import importlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Callable, Collection, Iterable, Mapping, NamedTuple, Optional

import pytest

from test_install_docs import fenced_blocks

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "predictive_maintenance_mcp"

#: What the scan reads. Explicit, so new public content is opted in by its
#: location rather than discovered by walking the whole tree.
INCLUDE_GLOBS = (
    "*.md",
    "docs/**/*.md",
    "plugin/**/*.md",
    "examples/**/*.md",
    "examples/**/*.py",
    "notebooks/*.ipynb",
    "benchmarks/*.py",
    ".github/ISSUE_TEMPLATE/*",
)

#: Internal, gitignored locations. Matched before any file is opened.
INTERNAL_PREFIXES = (
    "docs/plans/",
    "docs/brainstorms/",
    "docs/solutions/",
    "docs/residual-review-findings/",
    "docs/issue-drafts/",
    "docs/distribution-kit/",
    ".claude/",
    "htmlcov/",
    ".venv/",
)
INTERNAL_FILES = ("CLAUDE.md", "PRD_INDUSTRIAL_TRANSFORMATION.md")
INTERNAL_PATTERNS = ("docs/AUDIT-*.md",)

#: Files exempt from the retired-name rule, with the reason. The import rule
#: still applies to them.
RETIRED_NAME_EXEMPT = {
    "CHANGELOG.md": (
        "A dated record: each entry names the surface as it was at that "
        "release, and the v0.9 migration table maps the retired names to their "
        "successors on purpose."
    ),
}

#: Where the instructions text is reported from.
INSTRUCTIONS = "src/server.py:instructions"

#: snake_case identifiers in the instructions that are not tools or prompts
#: but legitimately appear call-shaped or backticked. Currently none: add one
#: only with a comment saying what it is.
ALLOWED_INSTRUCTION_IDENTIFIERS: frozenset[str] = frozenset()

#: Known drift at the time this guard landed: (file, rule, token) -> reason.
#: Recorded instead of fixed, so new drift stays blocked while old drift
#: stays visible; each entry is an issue candidate. Remove an entry when the
#: content is corrected: the stale-entry test insists on it.
_SEVERITY_IMPORT = f"{PACKAGE}.diagnostics:assess_vibration_severity"
_ISO_SEVERITY_IMPORT = f"{PACKAGE}.diagnostics.iso20816:assess_vibration_severity"

ALLOWLIST: dict[tuple[str, str, str], str] = {
    (
        ".github/ISSUE_TEMPLATE/domain_validation.md",
        "retired-name",
        "evaluate_iso_20816",
    ): (
        "The example tool list names the pre-v0.9 severity tool, which "
        "assess_severity replaced."
    ),
    ("EXAMPLES.md", "retired-name", "manual://read/{filename}"): (
        "The manual walkthrough shows the manual resource dropped in v0.9, as "
        "a response field and as the full-text access path; "
        "read_manual_excerpt replaced it."
    ),
    ("docs/architecture.md", "retired-name", "assess_vibration_severity"): (
        "The Block 4 data-flow diagram and the code example name a severity "
        "function that no longer exists."
    ),
    ("docs/architecture.md", "import", _ISO_SEVERITY_IMPORT): (
        "The code example imports a function the iso20816 module no longer " "defines."
    ),
    (
        "notebooks/01_getting_started.ipynb",
        "retired-name",
        "assess_vibration_severity",
    ): ("The severity cell imports and calls a function that no longer exists."),
    ("notebooks/01_getting_started.ipynb", "import", _SEVERITY_IMPORT): (
        "The severity cell fails at import: the diagnostics package no longer "
        "exports this function."
    ),
    (
        "notebooks/03_condition_monitoring.ipynb",
        "retired-name",
        "assess_vibration_severity",
    ): ("The severity cell imports and calls a function that no longer exists."),
    ("notebooks/03_condition_monitoring.ipynb", "import", _SEVERITY_IMPORT): (
        "The severity cell fails at import: the diagnostics package no longer "
        "exports this function."
    ),
    ("benchmarks/bench_phase1.py", "retired-name", "assess_vibration_severity"): (
        "The benchmark imports and times a function that no longer exists."
    ),
    ("benchmarks/bench_phase1.py", "import", _ISO_SEVERITY_IMPORT): (
        "The script fails at import, so the benchmark cannot run."
    ),
}


class Violation(NamedTuple):
    """One finding. ``(file, rule, token)`` is its identity in the allowlist;
    line and detail locate it for the reader of a failure."""

    file: str
    rule: str
    token: str
    line: int
    detail: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.file, self.rule, self.token)


class ImportStatement(NamedTuple):
    module: str
    names: tuple[str, ...]
    line: int


class Snippet(NamedTuple):
    """Code inside a document: absolute line = offset + line within text."""

    offset: int
    text: str
    where: str


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def is_public(relative: str) -> bool:
    """Whether a repo-relative POSIX path may be read by this scan."""
    if relative.startswith(INTERNAL_PREFIXES) or relative in INTERNAL_FILES:
        return False
    if any(fnmatch.fnmatch(relative, pattern) for pattern in INTERNAL_PATTERNS):
        return False
    return "__pycache__" not in relative.split("/")


def public_files() -> list[str]:
    found = set()
    for pattern in INCLUDE_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if path.is_file() and is_public(relative):
                found.add(relative)
    return sorted(found)


# ---------------------------------------------------------------------------
# Rule 1: retired endpoint names
# ---------------------------------------------------------------------------


def load_old_to_new() -> dict:
    """``OLD_TO_NEW`` from the surface-parity test, imported, not copied."""
    path = REPO_ROOT / "tests" / "test_surface_parity.py"
    spec = importlib.util.spec_from_file_location("_doc_drift_surface_parity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OLD_TO_NEW


def retired_names(old_to_new: Mapping[str, tuple]) -> frozenset[str]:
    """Old names whose destination is another name, or a drop (None)."""
    return frozenset(old for old, (new, _) in old_to_new.items() if new != old)


def _retired_pattern(name: str) -> str:
    """A whole-identifier pattern for *name*.

    A resource URI template matches its literal part with or without a
    concrete argument (``manual://read/test.pdf``), never a longer URI.
    """
    if "://" in name:
        literal = name.split("{", 1)[0].rstrip("/")
        return rf"(?<![\w/]){re.escape(literal)}(?![\w-])"
    return rf"(?<!\w){re.escape(name)}(?!\w)"


@functools.lru_cache(maxsize=None)
def _retired_regex(retired: frozenset[str]) -> re.Pattern:
    ordered = sorted(retired, key=lambda n: (-len(n), n))
    groups = "|".join(f"(?P<n{i}>{_retired_pattern(n)})" for i, n in enumerate(ordered))
    return re.compile(groups or r"(?!x)x")


def retired_name_violations(
    text: str,
    source: str,
    retired: frozenset[str],
    offset: int = 0,
    where: str = "",
) -> list[Violation]:
    """One violation per occurrence of a retired name, in text order."""
    ordered = sorted(retired, key=lambda n: (-len(n), n))
    violations = []
    for match in _retired_regex(frozenset(retired)).finditer(text):
        name = ordered[int(match.lastgroup[1:])]
        line = offset + text.count("\n", 0, match.start()) + 1
        violations.append(
            Violation(
                source,
                "retired-name",
                name,
                line,
                f"{where + ': ' if where else ''}names the retired endpoint "
                f"'{name}'",
            )
        )
    return violations


# ---------------------------------------------------------------------------
# Rule 2: project imports resolve
# ---------------------------------------------------------------------------

_PROMPT = r"(?:>>>[ \t]*|\.\.\.[ \t]*)?"
_FROM_RE = re.compile(
    rf"^[ \t]*{_PROMPT}from[ \t]+(?P<module>{PACKAGE}(?:\.\w+)*)[ \t]+import[ \t]+"
    r"(?P<names>\([^)]*\)|[^\n]*)",
    re.M,
)
_IMPORT_RE = re.compile(rf"^[ \t]*{_PROMPT}import[ \t]+(?P<modules>[^\n]*)", re.M)


def _clean_names(raw: str) -> tuple[str, ...]:
    """Imported names from an import list: aliases, comments and wrapping
    parentheses or continuations removed."""
    if not raw.lstrip().startswith("("):
        raw = raw.split(";", 1)[0]
    lines = [line.split("#", 1)[0] for line in raw.splitlines()]
    flat = " ".join(lines).replace("(", " ").replace(")", " ").replace("\\", " ")
    names = []
    for part in flat.split(","):
        words = part.split()
        if words:
            names.append(words[0])
    return tuple(names)


def project_imports(code: str) -> list[ImportStatement]:
    """Absolute imports of the project package in *code*, in text order."""
    found = []
    for match in _FROM_RE.finditer(code):
        line = code.count("\n", 0, match.start()) + 1
        found.append(
            (
                match.start(),
                ImportStatement(
                    match.group("module"), _clean_names(match.group("names")), line
                ),
            )
        )
    for match in _IMPORT_RE.finditer(code):
        line = code.count("\n", 0, match.start()) + 1
        for module in _clean_names(match.group("modules")):
            if module == PACKAGE or module.startswith(PACKAGE + "."):
                found.append((match.start(), ImportStatement(module, (), line)))
    return [statement for _, statement in sorted(found, key=lambda item: item[0])]


@functools.lru_cache(maxsize=None)
def resolve_project_import(module: str, name: Optional[str]) -> Optional[str]:
    """None when ``module`` (and ``name`` in it) resolve, else the reason."""
    try:
        imported = importlib.import_module(module)
    except ImportError as error:
        return f"{module} does not import ({type(error).__name__}: {error})"
    if name is None or name == "*" or hasattr(imported, name):
        return None
    try:
        importlib.import_module(f"{module}.{name}")
    except ImportError:
        return f"{module} has no attribute or submodule '{name}'"
    return None


_DEFINITION_RE = re.compile(
    r"^[ \t]*(?:async[ \t]+def|def|class)[ \t]+([A-Za-z_]\w*)", re.M
)


def defined_names(snippets: Iterable["Snippet"]) -> set[str]:
    """Functions and classes a document's own code defines.

    A tutorial writes a function in one block and imports it from the
    project in the next; that import resolves once the reader has followed
    the tutorial, so it is not drift. The cost: a document that defines a
    name could also hide a stale import of the same name.
    """
    return {name for s in snippets for name in _DEFINITION_RE.findall(s.text)}


def import_violations(
    statements: Iterable[ImportStatement],
    source: str,
    resolve: Callable[[str, Optional[str]], Optional[str]],
    where: str = "",
    defined: Collection[str] = (),
) -> list[Violation]:
    """One violation per unresolved module, else per unresolved name. Names
    in *defined* (see :func:`defined_names`) are the document's own."""
    prefix = f"{where}: " if where else ""
    violations = []
    for statement in statements:
        problem = resolve(statement.module, None)
        if problem:
            violations.append(
                Violation(
                    source, "import", statement.module, statement.line, prefix + problem
                )
            )
            continue
        for name in statement.names:
            if name in defined:
                continue
            problem = resolve(statement.module, name)
            if problem:
                violations.append(
                    Violation(
                        source,
                        "import",
                        f"{statement.module}:{name}",
                        statement.line,
                        prefix + problem,
                    )
                )
    return violations


def notebook_cells(text: str) -> list[tuple[int, str, str]]:
    """``(1-based cell number, cell type, source)`` of a Jupyter notebook.
    nbformat stores a source as a list of lines or as one string."""
    cells = []
    for number, cell in enumerate(json.loads(text).get("cells", []), start=1):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append((number, cell.get("cell_type", ""), source))
    return cells


def _markdown_snippets(text: str, where: str = "") -> list[Snippet]:
    return [
        Snippet(first - 1, "\n".join(lines) + "\n", where)
        for _, first, lines in fenced_blocks(text)
    ]


def code_snippets(relative: str, text: str) -> list[Snippet]:
    """The code a reader would run: fenced blocks of markdown (in any
    language, a fence is what makes it code), notebook code cells and the
    fences of markdown cells, and Python files whole."""
    if relative.endswith(".md"):
        return _markdown_snippets(text)
    if relative.endswith(".ipynb"):
        snippets = []
        for number, kind, source in notebook_cells(text):
            if kind == "code":
                snippets.append(Snippet(0, source, f"cell {number}"))
            elif kind == "markdown":
                snippets += _markdown_snippets(source, f"cell {number}")
        return snippets
    if relative.endswith(".py"):
        return [Snippet(0, text, "")]
    return []


# ---------------------------------------------------------------------------
# Rule 3: tool references in the server instructions are registered
# ---------------------------------------------------------------------------

#: A snake_case identifier with at least one underscore: every registered
#: endpoint has one, and prose words mostly do not.
_IDENTIFIER = r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+"
#: Immediately followed by "(" (so "frequency (f_rot)" is prose) ...
_CALL_RE = re.compile(rf"\b({_IDENTIFIER})\(")
#: ... or alone inside backticks, optionally with an argument list.
_BACKTICK_RE = re.compile(rf"`({_IDENTIFIER})(?:\([^`]*\))?`")


def instruction_references(text: str) -> list[tuple[str, int]]:
    """``(name, line)`` of every call-shaped or backticked identifier."""
    found: dict[int, str] = {}
    for pattern in (_CALL_RE, _BACKTICK_RE):
        for match in pattern.finditer(text):
            found[match.start(1)] = match.group(1)
    return [
        (name, text.count("\n", 0, position) + 1)
        for position, name in sorted(found.items())
    ]


def instruction_violations(
    text: str, registered: Collection[str], allowed: Collection[str]
) -> list[Violation]:
    return [
        Violation(
            INSTRUCTIONS,
            "instructions-name",
            name,
            line,
            f"'{name}' is referenced as a tool but is not a registered tool or prompt",
        )
        for name, line in instruction_references(text)
        if name not in registered and name not in allowed
    ]


# ---------------------------------------------------------------------------
# The real repository
# ---------------------------------------------------------------------------


class Scan(NamedTuple):
    violations: list[Violation]
    files: list[str]
    import_statements: int
    notebook_cells: int
    instruction_references: int


@pytest.fixture(scope="module")
def server():
    from predictive_maintenance_mcp import server

    return server


@pytest.fixture(scope="module")
def registered(server) -> frozenset[str]:
    """Tool and prompt names, through the server's public listing API."""
    tools = asyncio.run(server.mcp.list_tools())
    prompts = asyncio.run(server.mcp.list_prompts())
    return frozenset(t.name for t in tools) | frozenset(p.name for p in prompts)


@pytest.fixture(scope="module")
def retired() -> frozenset[str]:
    return retired_names(load_old_to_new())


@pytest.fixture(scope="module")
def scan(server, registered, retired) -> Scan:
    violations: list[Violation] = []
    files = public_files()
    statements = cells = 0
    for relative in files:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if relative.endswith(".ipynb"):
            parsed = notebook_cells(text)
            cells += len(parsed)
            if relative not in RETIRED_NAME_EXEMPT:
                for number, _, source in parsed:
                    violations += retired_name_violations(
                        source, relative, retired, where=f"cell {number}"
                    )
        elif relative not in RETIRED_NAME_EXEMPT:
            violations += retired_name_violations(text, relative, retired)
        snippets = code_snippets(relative, text)
        defined = defined_names(snippets)
        for snippet in snippets:
            found = [
                s._replace(line=s.line + snippet.offset)
                for s in project_imports(snippet.text)
            ]
            statements += len(found)
            violations += import_violations(
                found, relative, resolve_project_import, snippet.where, defined
            )
    instructions = server.mcp.instructions or ""
    violations += retired_name_violations(instructions, INSTRUCTIONS, retired)
    violations += instruction_violations(
        instructions, registered, ALLOWED_INSTRUCTION_IDENTIFIERS
    )
    return Scan(
        violations,
        files,
        statements,
        cells,
        len(instruction_references(instructions)),
    )


def test_the_scan_actually_read_public_content(scan):
    """Anchor for the rest: "found nothing" has to mean something."""
    expected = {
        "README.md",
        "EXAMPLES.md",
        "docs/architecture.md",
        "docs/TOOL_CATALOG.md",
        "plugin/README.md",
        "examples/adapters/stwinbox/README.md",
        "examples/adapters/stwinbox/stwinbox_to_measurement.py",
        "notebooks/01_getting_started.ipynb",
        "benchmarks/bench_phase1.py",
        ".github/ISSUE_TEMPLATE/domain_validation.md",
    }
    missing = expected - set(scan.files)
    assert not missing, f"the include globs no longer reach {sorted(missing)}"
    assert scan.import_statements >= 1, "no project import statement was parsed"
    assert scan.notebook_cells >= 1, "no notebook cell was parsed"
    assert scan.instruction_references >= 5, (
        f"only {scan.instruction_references} tool references parsed from the "
        f"server instructions"
    )


def test_no_internal_content_is_scanned(scan):
    leaked = [relative for relative in scan.files if not is_public(relative)]
    assert not leaked, leaked


def test_the_retired_names_really_are_retired(retired, registered):
    """A re-registered name must not stay banned from the docs."""
    assert len(retired) >= 15, sorted(retired)
    assert not retired & registered, sorted(retired & registered)


def test_no_violation_outside_the_allowlist(scan):
    """The guard itself: new drift in public content fails here."""
    unexpected = [v for v in scan.violations if v.key not in ALLOWLIST]
    assert not unexpected, "\n".join(
        f"{v.file}:{v.line}: [{v.rule}] {v.detail}" for v in unexpected
    ) + (
        "\nFix the content. Only if the drift must stay for now, add "
        "(file, rule, token) to ALLOWLIST with a reason."
    )


def test_the_allowlist_has_no_stale_entries(scan):
    """An excuse for drift that is gone excuses nothing."""
    gone = sorted(set(ALLOWLIST) - {v.key for v in scan.violations})
    assert not gone, f"ALLOWLIST outlived what it excused: {gone}"


def test_every_allowlist_entry_has_a_reason():
    assert all(reason.strip() for reason in ALLOWLIST.values())
    assert all(reason.strip() for reason in RETIRED_NAME_EXEMPT.values())


# ---------------------------------------------------------------------------
# Synthetic fixtures: each check's failing paths, none of it on the real repo
# ---------------------------------------------------------------------------

RETIRED = frozenset(
    {
        "evaluate_iso_20816",
        "compute_envelope_spectrum_tool",
        "clear_signal",
        "manual://read/{filename}",
        "signal://list",
    }
)


def keys(violations):
    return [(v.rule, v.token) for v in violations]


class TestRetiredNameTable:
    def test_only_names_whose_destination_moved_are_retired(self):
        table = {
            "analyze_fft": ("analyze_fft", "kept"),
            "plot_spectrum": ("generate_fft_report", "merged"),
            "generate_iso_diagnostic_report": (None, "dropped"),
        }
        assert retired_names(table) == {
            "plot_spectrum",
            "generate_iso_diagnostic_report",
        }


class TestRetiredNames:
    def check(self, text):
        return retired_name_violations(text, "fixture.md", RETIRED)

    def test_a_retired_name_in_an_issue_template_is_reported(self):
        text = "- **Tool**: (e.g., `analyze_envelope`, `evaluate_iso_20816`)\n"
        assert keys(self.check(text)) == [("retired-name", "evaluate_iso_20816")]

    def test_the_successor_name_passes(self):
        assert self.check("- **Tool**: (e.g., `assess_severity`)\n") == []

    def test_whole_identifiers_only(self):
        """A live name that merely contains, or is contained in, a retired one."""
        text = (
            "my_evaluate_iso_20816_v2 and compute_envelope_spectrum(x)\n"
            "clear_signals(signal_id=None)\n"
        )
        assert self.check(text) == []

    def test_two_occurrences_report_twice(self):
        text = "evaluate_iso_20816 here\nand evaluate_iso_20816 again\n"
        assert [v.line for v in self.check(text)] == [1, 2]

    def test_a_retired_resource_uri_is_reported_with_or_without_a_filename(self):
        text = (
            '"uri": "manual://read/test_pump_manual.pdf"\n'
            "via `manual://read/{filename}`\n"
            "see signal://list\n"
        )
        assert keys(self.check(text)) == [
            ("retired-name", "manual://read/{filename}"),
            ("retired-name", "manual://read/{filename}"),
            ("retired-name", "signal://list"),
        ]

    def test_a_longer_uri_is_not_the_retired_one(self):
        assert self.check("manual://reader and signal://listing\n") == []

    def test_empty_text_has_nothing(self):
        assert self.check("") == []


SYNTHETIC_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "from predictive_maintenance_mcp.diagnostics import "
                    "assess_vibration_severity\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": "from predictive_maintenance_mcp.diagnostics import classify_zone",
            },
        ],
    }
)


class TestProjectImportExtraction:
    def test_from_imports_with_aliases_and_parentheses(self):
        code = (
            "from predictive_maintenance_mcp.diagnostics import classify_zone as cz, x\n"
            "from predictive_maintenance_mcp.signal_processing import (\n"
            "    compute_psd,  # comment\n"
            "    compute_stft_spectrogram,\n"
            ")\n"
            "    from predictive_maintenance_mcp import server\n"
            ">>> from predictive_maintenance_mcp.models import y\n"
        )
        assert project_imports(code) == [
            ImportStatement(
                "predictive_maintenance_mcp.diagnostics", ("classify_zone", "x"), 1
            ),
            ImportStatement(
                "predictive_maintenance_mcp.signal_processing",
                ("compute_psd", "compute_stft_spectrogram"),
                2,
            ),
            ImportStatement("predictive_maintenance_mcp", ("server",), 6),
            ImportStatement("predictive_maintenance_mcp.models", ("y",), 7),
        ]

    def test_plain_imports(self):
        code = "import numpy as np, predictive_maintenance_mcp.server as s\n"
        assert project_imports(code) == [
            ImportStatement("predictive_maintenance_mcp.server", (), 1)
        ]

    def test_other_packages_and_relative_imports_are_not_ours(self):
        code = "from ._utils import refusal_boundary\nfrom mcp import Client\n"
        assert project_imports(code) == []

    def test_empty_code_has_no_imports(self):
        assert project_imports("") == []


class TestProjectImportResolution:
    def check(self, code):
        return import_violations(
            project_imports(code), "fixture", resolve_project_import
        )

    def test_a_removed_attribute_is_reported(self):
        code = "from predictive_maintenance_mcp.diagnostics import assess_vibration_severity\n"
        assert keys(self.check(code)) == [
            (
                "import",
                "predictive_maintenance_mcp.diagnostics:assess_vibration_severity",
            )
        ]

    def test_a_real_attribute_and_a_submodule_pass(self):
        code = (
            "from predictive_maintenance_mcp.diagnostics import classify_zone\n"
            "from predictive_maintenance_mcp import server\n"
            "import predictive_maintenance_mcp.mcp_tools.prompts\n"
        )
        assert self.check(code) == []

    def test_one_bad_name_among_good_ones_is_reported(self):
        """Add: a phantom name next to real ones."""
        code = (
            "from predictive_maintenance_mcp.signal_processing import (\n"
            "    compute_psd,\n"
            "    compute_nothing,\n"
            ")\n"
        )
        assert keys(self.check(code)) == [
            ("import", "predictive_maintenance_mcp.signal_processing:compute_nothing")
        ]

    def test_a_missing_module_is_reported(self):
        code = (
            "import predictive_maintenance_mcp.nope\n"
            "from predictive_maintenance_mcp.gone import x\n"
        )
        assert keys(self.check(code)) == [
            ("import", "predictive_maintenance_mcp.nope"),
            ("import", "predictive_maintenance_mcp.gone"),
        ]


class TestSnippets:
    def test_markdown_imports_are_read_from_fenced_blocks_only(self):
        """Remove-the-marker: the same line in prose is not code."""
        fenced = "Text\n\n```python\nfrom predictive_maintenance_mcp import x\n```\n"
        prose = "Text: from predictive_maintenance_mcp import x\n"
        assert [s.offset for s in code_snippets("a.md", fenced)] == [3]
        assert code_snippets("a.md", prose) == []

    def test_notebook_code_cells_are_snippets(self):
        snippets = code_snippets("n.ipynb", SYNTHETIC_NOTEBOOK)
        assert [s.where for s in snippets] == ["cell 2", "cell 3"]
        assert "assess_vibration_severity" in snippets[0].text

    def test_notebook_cells_parse_both_source_shapes(self):
        cells = notebook_cells(SYNTHETIC_NOTEBOOK)
        assert [kind for _, kind, _ in cells] == ["markdown", "code", "code"]
        assert cells[2][2].startswith("from predictive_maintenance_mcp")

    def test_a_synthetic_notebook_reports_the_removed_import(self):
        statements = [
            s
            for snippet in code_snippets("n.ipynb", SYNTHETIC_NOTEBOOK)
            for s in project_imports(snippet.text)
        ]
        violations = import_violations(statements, "n.ipynb", resolve_project_import)
        assert keys(violations) == [
            (
                "import",
                "predictive_maintenance_mcp.diagnostics:assess_vibration_severity",
            )
        ]

    def test_python_files_are_one_snippet(self):
        assert code_snippets("b.py", "import os\n") == [Snippet(0, "import os\n", "")]


class TestTutorialDefinedNames:
    """A tutorial writes a function in one block and imports it in the next:
    that import resolves once the reader follows the tutorial."""

    TUTORIAL = (
        "```python\n"
        "def analyze_temperature_trend(temperatures):\n"
        "    return {}\n"
        "class ThermalReport:\n"
        "    pass\n"
        "```\n"
        "```python\n"
        "from predictive_maintenance_mcp.mcp_tools.analysis_tools import (\n"
        "    analyze_temperature_trend,\n"
        "    analyze_humidity_trend,\n"
        ")\n"
        "from predictive_maintenance_mcp.models import ThermalReport\n"
        "```\n"
    )

    def violations(self, text):
        snippets = code_snippets("t.md", text)
        statements = [s for snippet in snippets for s in project_imports(snippet.text)]
        return import_violations(
            statements, "t.md", resolve_project_import, defined=defined_names(snippets)
        )

    def test_names_the_document_defines_are_found(self):
        snippets = code_snippets("t.md", self.TUTORIAL)
        assert defined_names(snippets) == {"analyze_temperature_trend", "ThermalReport"}

    def test_an_import_of_a_name_the_document_defines_passes(self):
        """Only the name the tutorial never wrote is reported."""
        assert keys(self.violations(self.TUTORIAL)) == [
            (
                "import",
                "predictive_maintenance_mcp.mcp_tools.analysis_tools:"
                "analyze_humidity_trend",
            )
        ]

    def test_without_the_definition_the_import_is_reported(self):
        """Remove-the-marker: drop the def and the import is drift again."""
        text = self.TUTORIAL.replace("def analyze_temperature_trend", "def other")
        assert (
            "import",
            "predictive_maintenance_mcp.mcp_tools.analysis_tools:"
            "analyze_temperature_trend",
        ) in keys(self.violations(text))

    def test_a_definition_in_prose_does_not_count(self):
        text = "Write def analyze_humidity_trend yourself.\n" + self.TUTORIAL
        assert defined_names(code_snippets("t.md", text)) == {
            "analyze_temperature_trend",
            "ThermalReport",
        }


class TestInstructionReferences:
    REGISTERED = frozenset({"load_signal", "list_signals", "diagnose_bearing"})

    def check(self, text, allowed=frozenset()):
        return instruction_violations(text, self.REGISTERED, allowed)

    def test_registered_names_pass(self):
        text = (
            "- load_signal(signal_unit='g') then `list_signals`\n"
            "- diagnose_bearing() - workflow\n"
        )
        assert self.check(text) == []

    def test_an_unregistered_call_is_reported(self):
        assert keys(self.check("Use foo_tool(x=1) first.\n")) == [
            ("instructions-name", "foo_tool")
        ]

    def test_an_unregistered_backticked_name_is_reported(self):
        assert keys(self.check("Prefer `foo_tool` here.\n")) == [
            ("instructions-name", "foo_tool")
        ]

    def test_prose_identifiers_and_spaced_parentheses_are_not_references(self):
        text = (
            "the rms_velocity_mm_s route; status processing_not_homogeneous\n"
            "shaft frequency f_rot (n=1..3); use generate_*_report() or report()\n"
        )
        assert self.check(text) == []

    def test_two_mentions_report_twice(self):
        assert [v.line for v in self.check("foo_tool()\nfoo_tool()\n")] == [1, 2]

    def test_an_allowed_identifier_passes(self):
        assert self.check("foo_tool()\n", allowed=frozenset({"foo_tool"})) == []


class TestPublicScope:
    @pytest.mark.parametrize(
        "relative",
        [
            "docs/plans/2026-x-plan.md",
            "docs/brainstorms/x.md",
            "docs/solutions/conventions/x.md",
            "docs/residual-review-findings/x.md",
            "docs/issue-drafts/x.md",
            "docs/distribution-kit/x.md",
            "docs/AUDIT-2026-07-10.md",
            "CLAUDE.md",
            "PRD_INDUSTRIAL_TRANSFORMATION.md",
            ".claude/worktrees/a/README.md",
            "htmlcov/index.md",
            ".venv/lib/x.md",
        ],
    )
    def test_internal_content_is_never_public(self, relative):
        assert not is_public(relative)

    @pytest.mark.parametrize(
        "relative",
        ["README.md", "docs/architecture.md", "plugin/README.md", "EXAMPLES.md"],
    )
    def test_public_content_is_public(self, relative):
        assert is_public(relative)
