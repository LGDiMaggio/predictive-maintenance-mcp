"""Audit report builder: results in, one report and issue candidates out.

``python -m tools.audit report --results DIR --out DIR`` is the last step of
the weekly audit. It reads what the check jobs uploaded and writes three
files into ``--out``:

``audit-report.json``
    ``pmm-audit-report/1``: run metadata, a per-check status table, every
    finding merged by fingerprint with its de-duplication decision, the
    candidate list and the verdict. This is the interface for later analysis
    (by a person or an AI assistant run locally); it holds no waveforms.
``audit-report.md``
    The same, for reading: summary line, per-check table, findings by
    category, already tracked, suppressed, recorded only, environment.
``issue-candidates.md``
    Ready-to-paste blocks for a human to file with the "Audit finding" issue
    form. Nothing is filed automatically.

Inputs
------
``--results`` is searched recursively (one directory per downloaded
artifact) and files are recognized by shape:

* ``*.json`` whose ``schema`` is ``pmm-audit-result/1``: a check result
  (``smoke``, ``package-smoke``, ``upstream``, ``security``). A
  ``package-smoke`` result is also a leg for the upstream join, with the
  versions it embeds under ``facts.installed_versions``.
* ``junit*.xml``: one suite leg, named after the directory holding it (the
  artifact name). Failures and errors become findings by test module;
  strict xfails whose reason starts with ``audit-known-defect:`` become
  diagnostic-correctness findings; other xfails are ignored.
* ``pip-list*.json`` (``pip list --format json``) next to a JUnit file: the
  versions that leg installed. A pip list with no JUnit beside it is a suite
  leg that did not finish: it becomes an error leg and a problem (exit 2).
* ``pytest-exit-code*.txt`` next to a JUnit file: pytest's exit status. A
  non-zero status with no failed test (the coverage gate, a session error)
  fails the leg with a ``pytest-exit`` finding.
* Any other JSON is listed as ignored (raw pip-audit output, for example).

Upstream join
-------------
The upstream checker emits ``test-gated`` findings unresolved. Each is
resolved from the legs that installed one of its ``evidence.candidates``:
``harmless`` when all of them passed, ``breaking-candidate`` when any did
not (the subject then gains the upstream ``major.minor``, so each upstream
line is reviewed once), and ``not-checked`` when none installed a candidate.
mcp ``review`` findings get the same leg results attached as evidence.

De-duplication
--------------
Findings merge by fingerprint across results and variants. Against existing
issues (``--known-issues``, ``gh issue list --json
number,title,state,stateReason,body,labels``), an issue matches when its
title or body holds ``pmm-audit:<fingerprint>`` and a maintainer has
accepted it by adding the :data:`ACCEPTED_LABEL` label. Anyone can open an
issue and quote a marker, and an author can close their own issue as not
planned, so an issue without that label never tracks or suppresses a
finding: the finding stays a candidate, with a note naming the issue. Only
users with triage access can add a label, and no issue form applies this one.
Issues closed as duplicates are skipped (the issue they duplicate carries the
marker if it tracks the finding). Among accepted matches:

* an open match: ``duplicate-open #N``, not a candidate;
* severity info or low with classification harmless: recorded only;
* the newest closed match closed as not planned: ``suppressed #N``, except
  security findings, which stay candidates annotated with ``#N``;
* any other closed match: ``regression-of #N``, a candidate;
* otherwise ``new``, a candidate.

Issue titles and bodies are written by anyone with a GitHub account, so they
are used for marker matching only and never rendered: the outputs show an
issue's number, state and the matched fingerprint.

Vulnerabilities and tracking
----------------------------
Tracking decides whether a finding is proposed again as an issue candidate;
it never decides whether a vulnerability is resolved. Every security
advisory in the results is listed under ``vulnerabilities`` with its
tracking issue, if any. An advisory with a fix installable within the
declared range keeps the verdict red while it is present, tracked or not.

Verdict and exit codes
----------------------
2 after writing whatever it can, when evidence is missing: the results
directory is missing or empty, an expected check or matrix leg has no result
(``--expect``: check ids or ``check:variant`` entries, default
:data:`results.EXPECTED_WEEKLY_CHECKS`), a check result has status
``error``, a file is unreadable or malformed, a JUnit file has no test cases,
or the known-issues list reaches :data:`results.KNOWN_ISSUES_LIMIT` (it may be
truncated). 1 after writing all three outputs, when a check result has
status ``fail`` (a failed suite leg or smoke run) or a security advisory with
a fix within the declared range is present. 0 otherwise. The verdict repeats
what the check jobs already show, so the report alone says whether the week
passed.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from tools.audit import results
from tools.audit.results import Finding

__all__ = [
    "ACCEPTED_LABEL",
    "CANDIDATE_FIELDS",
    "KNOWN_DEFECT_PREFIX",
    "MODULE_CATEGORIES",
    "CheckRun",
    "Inputs",
    "KnownIssue",
    "Leg",
    "add_arguments",
    "build_report",
    "collect_inputs",
    "decide",
    "parse_junit",
    "parse_known_issues",
    "parse_pip_list",
    "render_candidates",
    "render_report",
    "render_summary",
    "resolve_upstream",
    "run",
    "template_parity",
]

REPORT_JSON = "audit-report.json"
REPORT_MD = "audit-report.md"
CANDIDATES_MD = "issue-candidates.md"

KNOWN_DEFECT_PREFIX = "audit-known-defect:"

#: pytest's exit status, written by the suite job next to its JUnit file.
PYTEST_EXIT_PATTERN = "pytest-exit-code*.txt"
TEST_FAILED_RULE = "test-failed"
KNOWN_DEFECT_RULE = "known-defect"

#: De-duplication decision kinds (see the module docstring).
DUPLICATE_OPEN_KIND = "duplicate-open"
RECORDED_KIND = "recorded"
SUPPRESSED_KIND = "suppressed"
REGRESSION_OF_KIND = "regression-of"
NEW_KIND = "new"

#: Test module -> finding category for failures; anything else is regression.
MODULE_CATEGORIES = {
    "test_tool_contracts": "tool-contract",
    "test_doc_drift": "doc-drift",
    "test_install_docs": "doc-drift",
    "test_diagnostic_invariants": "diagnostic-correctness",
}

#: Packages whose ``review`` findings get the leg results attached (lock-step).
MCP_PACKAGES = ("mcp", "mcp-types")

MARKER_PREFIX = "pmm-audit:"
_MARKER_RE = re.compile(
    re.escape(MARKER_PREFIX)
    + r"([0-9a-f]{%d})(?![0-9a-f])" % results.FINGERPRINT_LENGTH
)

ISSUE_TITLE_PREFIX = "[Audit] "

#: The label a maintainer adds to an issue to accept it as the tracker of the
#: fingerprints it carries (see the module docstring). The weekly workflow
#: fetches only issues with this label, and tests/test_workflow_policy.py
#: keeps it out of every issue form.
ACCEPTED_LABEL = "audit-accepted"

#: (issue form field id, label) in the order a candidate block renders them.
#: The issue form ``.github/ISSUE_TEMPLATE/audit_finding.yml`` declares the
#: same fields; :func:`template_parity` keeps the two in step.
CANDIDATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("fingerprint", "Audit fingerprint"),
    ("category", "Category"),
    ("severity", "Severity"),
    ("check", "Check and variants"),
    ("evidence", "Evidence"),
    ("reproduction", "Reproduction"),
    ("run-link", "Audit run"),
    ("duplicate-search", "Duplicate search"),
)

#: Caps for Markdown output (the JSON keeps results.EXCERPT_LIMIT).
MD_EXCERPT_LIMIT = 2000
MD_VALUE_LIMIT = 500
TITLE_LIMIT = 200

_SEVERITY_RANK = {name: rank for rank, name in enumerate(results.SEVERITIES)}
_CATEGORY_RANK = {name: rank for rank, name in enumerate(results.CATEGORIES)}
_CHECK_RANK = {name: rank for rank, name in enumerate(results.CHECK_IDS)}
#: When occurrences of one fingerprint disagree, the most cautious wins
#: (results.CLASSIFICATIONS lists them most cautious first).
_CLASSIFICATION_RANK = {name: rank for rank, name in enumerate(results.CLASSIFICATIONS)}
_ENV_COLUMNS = ("mcp", "mcp-types", "numpy", "scipy", "pandas", "pydantic")
_GITHUB_ENV = (
    ("GITHUB_SERVER_URL", "server_url"),
    ("GITHUB_REPOSITORY", "repository"),
    ("GITHUB_RUN_ID", "run_id"),
    ("GITHUB_SHA", "sha"),
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    """A suite or package-smoke run, with the versions it installed."""

    check: str
    variant: str
    status: str
    installed: Mapping[str, str]
    source: str = ""


@dataclass
class CheckRun:
    """One entry of the per-check table: a result file or a suite leg."""

    check: str
    variant: str
    status: str
    source: str = ""
    findings: list[Finding] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    tests: Optional[dict[str, int]] = None


@dataclass
class Inputs:
    """Everything read from ``--results``."""

    runs: list[CheckRun] = field(default_factory=list)
    legs: list[Leg] = field(default_factory=list)
    problems: list[dict[str, str]] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnownIssue:
    """An existing issue reduced to what de-duplication may use."""

    number: int
    state: str
    state_reason: Optional[str]
    fingerprints: frozenset[str]
    accepted: bool = False


def _one_line(text: str, limit: int = TITLE_LIMIT) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _problem(source: str, problem: str) -> dict[str, str]:
    return {"source": source, "problem": problem}


# ---------------------------------------------------------------------------
# JUnit
# ---------------------------------------------------------------------------

_PARAM_RE = re.compile(r"^(?P<base>[^\[]*)\[(?P<param>.*)\]$", re.S)


def _node(classname: str, name: str) -> tuple[str, Optional[str], str]:
    """``(node id without parametrization, parameter id, test module)``.

    pytest writes ``classname="tests.test_x.TestY" name="test_z[p]"``; the
    node id ``tests/test_x.py::TestY::test_z`` is stable across runs and
    platforms, and the parameter ids are kept as evidence instead.
    """
    match = _PARAM_RE.match(name)
    base, param = (match["base"], match["param"]) if match else (name, None)
    dotted = [part for part in classname.split(".") if part] if classname else []
    if not dotted:
        # A collection error: pytest names the module in ``name``.
        parts = [part for part in base.split(".") if part] or [base]
        return "/".join(parts) + ".py", param, parts[-1]
    index = next(
        (
            i
            for i in range(len(dotted) - 1, -1, -1)
            if dotted[i].startswith("test_") or dotted[i].endswith("_test")
        ),
        None,
    )
    if index is None:
        return f"{classname}::{base}", param, dotted[-1]
    path = "/".join(dotted[: index + 1]) + ".py"
    return "::".join([path, *dotted[index + 1 :], base]), param, dotted[index]


def parse_junit(text: str, variant: str, source: str = "junit.xml") -> CheckRun:
    """Turn pytest ``--junitxml`` output into one ``suite`` entry.

    Args:
        text: The JUnit XML.
        variant: The leg's name (the artifact directory).
        source: Path shown in the report.

    Returns:
        A ``suite`` :class:`CheckRun`: status ``fail`` when any test failed or
        errored, findings for those and for ``audit-known-defect:`` xfails.

    Raises:
        ValueError: When the text is not JUnit XML or holds no test cases (a
            suite that ran nothing must never read as a pass).
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"not JUnit XML: {exc}") from None
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError(f"the root element is <{root.tag}>, not <testsuites>")
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError(
            "the JUnit file has no test cases: the suite collected or ran nothing"
        )
    tests = dict.fromkeys(
        ("total", "passed", "failures", "errors", "skipped", "xfailed"), 0
    )
    tests["total"] = len(cases)
    failed: dict[str, dict[str, Any]] = {}
    defects: dict[str, dict[str, Any]] = {}
    for case in cases:
        node, param, module = _node(case.get("classname", ""), case.get("name", ""))
        failure, error = case.find("failure"), case.find("error")
        problem = failure if failure is not None else error
        skipped = case.find("skipped")
        if problem is not None:
            outcome = "failure" if failure is not None else "error"
            tests["failures" if outcome == "failure" else "errors"] += 1
            entry = failed.setdefault(
                node,
                {
                    "module": module,
                    "outcome": outcome,
                    "message": problem.get("message") or "",
                    "excerpt": problem.text or "",
                    "cases": [],
                },
            )
            if param is not None:
                entry["cases"].append(param)
        elif skipped is not None and skipped.get("type") == "pytest.xfail":
            tests["xfailed"] += 1
            reason = (skipped.get("message") or "").strip()
            if reason.startswith(KNOWN_DEFECT_PREFIX):
                rest = reason[len(KNOWN_DEFECT_PREFIX) :].strip()
                defect_id, _, description = rest.partition(":")
                defect_id = defect_id.strip() or node
                entry = defects.setdefault(
                    defect_id,
                    {"description": description.strip(), "reason": reason, "tests": []},
                )
                entry["tests"].append(node)
        elif skipped is not None:
            tests["skipped"] += 1
        else:
            tests["passed"] += 1

    findings = []
    for node, entry in sorted(failed.items()):
        category = MODULE_CATEGORIES.get(entry["module"], "regression")
        message = _one_line(
            entry["message"].splitlines()[0] if entry["message"] else ""
        )
        evidence: dict[str, Any] = {
            "test": node,
            "outcome": entry["outcome"],
            "junit": source,
        }
        if entry["cases"]:
            evidence["cases"] = sorted(set(entry["cases"]))
        if message:
            evidence["message"] = message
        if entry["excerpt"].strip():
            evidence["excerpt"] = results.truncate_evidence(
                entry["excerpt"].strip(), keep="tail"
            )
        findings.append(
            Finding(
                rule=TEST_FAILED_RULE,
                category=category,
                severity="medium" if category == "doc-drift" else "high",
                subject=node,
                title=_one_line(f"{node} fails" + (f": {message}" if message else "")),
                evidence=evidence,
            )
        )
    for defect_id, entry in sorted(defects.items()):
        tests_seen = sorted(set(entry["tests"]))
        evidence = {
            "test": entry["tests"][0],
            "reason": results.truncate_evidence(entry["reason"], MD_VALUE_LIMIT),
            "junit": source,
        }
        if len(tests_seen) > 1:
            evidence["tests"] = tests_seen
        findings.append(
            Finding(
                rule=KNOWN_DEFECT_RULE,
                category="diagnostic-correctness",
                severity="medium",
                subject=defect_id,
                title=_one_line(entry["description"] or f"known defect {defect_id}"),
                evidence=evidence,
            )
        )
    status = "fail" if tests["failures"] or tests["errors"] else "pass"
    return CheckRun(
        check="suite",
        variant=variant,
        status=status,
        source=source,
        findings=findings,
        tests=tests,
    )


def parse_pip_list(text: str) -> dict[str, str]:
    """``pip list --format json`` -> ``{normalized name: version}``.

    Raises:
        ValueError: On anything but a non-empty list of name/version entries.
    """
    return {
        results.canonical_name(name): version.strip()
        for name, version in results.parse_pip_list_json(text).items()
    }


# ---------------------------------------------------------------------------
# Reading the results directory
# ---------------------------------------------------------------------------


def _environment(doc_environment: Any) -> dict[str, Any]:
    env = doc_environment if isinstance(doc_environment, Mapping) else {}
    packages = env.get("packages") if isinstance(env.get("packages"), Mapping) else {}
    return {
        "python": env.get("python"),
        "platform": env.get("platform"),
        "packages": {
            name: packages.get(name) for name in results.DEFAULT_ENVIRONMENT_PACKAGES
        },
    }


def _read_result(path: Path, source: str, inputs: Inputs) -> None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        inputs.problems.append(_problem(source, f"unreadable JSON: {exc}"))
        return
    if not isinstance(doc, Mapping) or doc.get("schema") != results.RESULT_SCHEMA:
        inputs.ignored.append(source)
        return
    try:
        # The local-processing guard and the excerpt cap apply on the way in.
        doc = results.canonicalize(doc)
        findings = results.validate_result(doc)
        for key in ("facts", "environment"):
            if not isinstance(doc[key], Mapping):
                raise ValueError(f"{key} must be an object")
    except ValueError as exc:
        inputs.problems.append(_problem(source, f"invalid result: {exc}"))
        return
    inputs.runs.append(
        CheckRun(
            check=doc["check"],
            variant=str(doc["variant"]),
            status=doc["status"],
            source=source,
            findings=findings,
            environment=_environment(doc["environment"]),
        )
    )
    installed = doc["facts"].get("installed_versions")
    if doc["check"] == "package-smoke" and isinstance(installed, Mapping):
        versions = {
            results.canonical_name(name): version.strip()
            for name, version in installed.items()
            if isinstance(version, str)
        }
        if versions:
            inputs.legs.append(
                Leg(
                    "package-smoke",
                    str(doc["variant"]),
                    doc["status"],
                    versions,
                    source,
                )
            )


def _leg_exit_code(paths: Sequence[Path], rel: Any, inputs: Inputs) -> Optional[int]:
    """pytest's exit status recorded in a leg directory, or None.

    None also when the leg recorded none (a local run); an unreadable or
    duplicated record is a problem.
    """
    if not paths:
        return None
    if len(paths) > 1:
        inputs.problems.append(
            _problem(rel(paths[0].parent), "several pytest exit-code files in one leg")
        )
        return None
    try:
        text = paths[0].read_text(encoding="utf-8").strip()
        return int(text)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        inputs.problems.append(
            _problem(rel(paths[0]), f"unreadable pytest exit code: {exc}")
        )
        return None


def _pytest_exit_finding(code: int, source: str) -> Finding:
    return Finding(
        rule="pytest-exit",
        category="regression",
        severity="high",
        subject="pytest exit status",
        title=(
            f"pytest exited with status {code} although no test failed: the "
            f"coverage gate or a session-level error failed the run"
        ),
        evidence={"exit_code": code, "junit": source},
    )


def collect_inputs(root: Path) -> Inputs:
    """Read every recognized file under ``root`` (see the module docstring).

    Unreadable or malformed files become problems naming the file; reading
    continues so the report shows everything that was readable.
    """
    inputs = Inputs()
    if not root.exists():
        inputs.problems.append(
            _problem(str(root), "the results directory does not exist")
        )
        return inputs
    if not root.is_dir():
        inputs.problems.append(
            _problem(str(root), "the results path is not a directory")
        )
        return inputs
    files = sorted(path for path in root.rglob("*") if path.is_file())

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix()

    junit = [p for p in files if fnmatch.fnmatch(p.name.lower(), "junit*.xml")]
    pip_lists = [p for p in files if fnmatch.fnmatch(p.name.lower(), "pip-list*.json")]
    exit_files = [
        p for p in files if fnmatch.fnmatch(p.name.lower(), PYTEST_EXIT_PATTERN)
    ]
    other_json = [
        p for p in files if p.suffix.lower() == ".json" and p not in pip_lists
    ]
    if not (junit or pip_lists or other_json):
        inputs.problems.append(
            _problem(
                str(root),
                "the results directory holds no result JSON, JUnit XML or pip "
                "list files: no check delivered evidence",
            )
        )
        return inputs

    for path in other_json:
        _read_result(path, rel(path), inputs)

    junit_dirs = sorted({path.parent for path in junit})
    for directory in junit_dirs:
        files_here = [p for p in junit if p.parent == directory]
        lists_here = [p for p in pip_lists if p.parent == directory]
        installed: dict[str, str] = {}
        if len(lists_here) > 1:
            inputs.problems.append(
                _problem(
                    rel(directory) if directory != root else ".",
                    "several pip-list files in one leg directory; expected one",
                )
            )
        elif lists_here:
            try:
                installed = parse_pip_list(lists_here[0].read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                inputs.problems.append(_problem(rel(lists_here[0]), str(exc)))
        exit_code = _leg_exit_code(
            [p for p in exit_files if p.parent == directory], rel, inputs
        )
        for path in files_here:
            variant = (
                directory.name
                if len(files_here) == 1
                else f"{directory.name}/{path.stem}"
            )
            try:
                run = parse_junit(path.read_text(encoding="utf-8"), variant, rel(path))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                inputs.problems.append(_problem(rel(path), str(exc)))
                continue
            if exit_code and run.status == "pass":
                # Every test passed, yet pytest failed the run: the coverage
                # gate or a session-level error, which JUnit does not record.
                run.status = "fail"
                run.findings.append(_pytest_exit_finding(exit_code, rel(path)))
            run.environment = {
                "python": None,
                "platform": None,
                "packages": {
                    name: installed.get(name)
                    for name in results.DEFAULT_ENVIRONMENT_PACKAGES
                },
            }
            inputs.runs.append(run)
            inputs.legs.append(Leg("suite", variant, run.status, installed, rel(path)))

    # A suite job records its installed versions even after a failed install
    # or a crashed test run, so a pip list with no JUnit beside it is a leg
    # that did not finish. It stays in the report as an error leg (its
    # versions still count against upstream releases it installed) and makes
    # the report fail closed, instead of dropping out of the verdict.
    for path in pip_lists:
        if path.parent in junit_dirs:
            continue
        directory = path.parent
        variant = directory.name if directory != root else rel(path)
        try:
            installed = parse_pip_list(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            inputs.problems.append(_problem(rel(path), str(exc)))
            installed = {}
        inputs.problems.append(
            _problem(
                rel(path),
                "suite leg produced no JUnit results: its install or test step "
                "did not run to completion",
            )
        )
        inputs.runs.append(CheckRun("suite", variant, "error", rel(path)))
        inputs.legs.append(Leg("suite", variant, "error", installed, rel(path)))
    return inputs


# ---------------------------------------------------------------------------
# Upstream join
# ---------------------------------------------------------------------------


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def _major_minor(version: str) -> str:
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    return f"{match[1]}.{match[2]}" if match else version


def _rebuild(finding: Finding, **changes: Any) -> Finding:
    """A copy with ``changes``; the fingerprint follows the new fields."""
    return replace(finding, fingerprint="", **changes)


def _most_severe(severities: Iterable[str]) -> str:
    """The most severe of ``severities`` (the first, on a tie)."""
    return min(severities, key=lambda name: _SEVERITY_RANK[name])


def resolve_upstream(finding: Finding, legs: Sequence[Leg]) -> Finding:
    """Resolve a ``test-gated`` finding, or annotate an mcp ``review`` one.

    Args:
        finding: An ``upstream`` finding.
        legs: Suite and package-smoke legs with their installed versions.

    Returns:
        ``finding`` itself when neither case applies; otherwise a new finding
        whose evidence lists, under ``legs``, every leg that installed one of
        ``evidence.candidates`` of ``evidence.package``.
    """
    evidence = finding.evidence
    package = evidence.get("package")
    candidates = evidence.get("candidates")
    mcp_review = (
        finding.classification == "review"
        and isinstance(package, str)
        and results.canonical_name(package) in MCP_PACKAGES
    )
    if finding.classification != "test-gated" and not mcp_review:
        return finding
    new_evidence = dict(evidence)
    if (
        not isinstance(package, str)
        or not isinstance(candidates, list)
        or not candidates
    ):
        if mcp_review:
            return finding
        new_evidence["resolution"] = "the finding names no package and candidates"
        return _rebuild(
            finding,
            classification="not-checked",
            severity=_most_severe((finding.severity, "medium")),
            evidence=new_evidence,
        )
    wanted = {str(candidate).strip() for candidate in candidates}
    name = results.canonical_name(package)
    rows = sorted(
        (
            {
                "check": leg.check,
                "variant": leg.variant,
                "status": leg.status,
                "version": leg.installed[name],
            }
            for leg in legs
            if leg.installed.get(name) in wanted
        ),
        key=lambda row: (row["variant"], row["check"]),
    )
    new_evidence["legs"] = rows
    if mcp_review:
        return _rebuild(finding, evidence=new_evidence)
    listed = ", ".join(str(candidate) for candidate in candidates)
    if not rows:
        new_evidence["resolution"] = (
            "no suite or package-smoke leg installed a candidate version"
        )
        return _rebuild(
            finding,
            classification="not-checked",
            severity=_most_severe((finding.severity, "medium")),
            title=f"{package} {listed}: no suite or package-smoke leg installed it",
            evidence=new_evidence,
        )
    failing = [row for row in rows if row["status"] != "pass"]
    if not failing:
        new_evidence["resolution"] = "every leg that installed a candidate passed"
        return _rebuild(
            finding,
            classification="harmless",
            title=f"{package} {listed}: every leg that installed it passed",
            evidence=new_evidence,
        )
    failing_names = [f"{row['check']}/{row['variant']}" for row in failing]
    new_evidence["failing_legs"] = failing_names
    new_evidence["resolution"] = "a leg that installed a candidate did not pass"
    versions = sorted({row["version"] for row in failing}, key=_version_key)
    line = _major_minor(versions[-1])
    return _rebuild(
        finding,
        classification="breaking-candidate",
        severity=_most_severe((finding.severity, "medium")),
        subject=f"{package} {line}",
        title=_one_line(
            f"{package} {', '.join(versions)} is installed by legs that did not "
            f"pass: {', '.join(failing_names)}"
        ),
        evidence=new_evidence,
    )


# ---------------------------------------------------------------------------
# Known issues and decisions
# ---------------------------------------------------------------------------


def _label_names(value: Any, where: str) -> set[str]:
    """Casefolded names from gh's ``labels`` (objects with ``name``, or strings).

    A missing ``labels`` key reads as no labels, so an issue fetched without
    them is never accepted.
    """
    if value is None:
        return set()
    if not isinstance(value, list):
        raise ValueError(f"{where} labels is not a list")
    names = set()
    for label in value:
        name = label.get("name") if isinstance(label, Mapping) else label
        if not isinstance(name, str):
            raise ValueError(f"{where} has a label without a string name")
        names.add(name.casefold())
    return names


def parse_known_issues(
    text: str, limit: int = results.KNOWN_ISSUES_LIMIT
) -> list[KnownIssue]:
    """Read ``gh issue list --json number,title,state,stateReason,body,labels``.

    Titles and bodies are scanned for ``pmm-audit:<fingerprint>`` markers and
    then dropped: nothing else of them is kept. An issue is accepted when its
    labels include :data:`ACCEPTED_LABEL` (GitHub label names are
    case-insensitive).

    Raises:
        ValueError: On a malformed list, or one whose length reaches
            ``limit`` (the ``--limit`` it was fetched with, so it may be
            truncated and de-duplication could not be trusted).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"known issues are not JSON: {exc}") from None
    if not isinstance(data, list):
        raise ValueError("known issues must be a JSON list (gh issue list --json)")
    if len(data) >= limit:
        raise ValueError(
            f"the known-issues list has {len(data)} entries, which reaches the "
            f"fetch limit {limit}: it may be truncated"
        )
    issues = []
    for index, item in enumerate(data):
        where = f"known issue [{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{where} is not an object")
        number, state = item.get("number"), item.get("state")
        if not isinstance(number, int) or isinstance(number, bool):
            raise ValueError(f"{where} has no integer number")
        if state not in ("OPEN", "CLOSED"):
            raise ValueError(f"{where} state {state!r} is not OPEN or CLOSED")
        reason = item.get("stateReason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"{where} stateReason is not a string")
        texts = []
        for key in ("title", "body"):
            value = item.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{where} {key} is not a string")
            texts.append(value or "")
        labels = _label_names(item.get("labels"), where)
        issues.append(
            KnownIssue(
                number=number,
                state=state,
                state_reason=reason.upper() if reason else None,
                fingerprints=frozenset(_MARKER_RE.findall("\n".join(texts))),
                accepted=ACCEPTED_LABEL.casefold() in labels,
            )
        )
    return issues


def _decision(
    kind: str,
    issue: Optional[int],
    candidate: bool,
    note: Optional[str] = None,
    label: Optional[str] = None,
) -> dict[str, Any]:
    if label is None:
        label = kind if issue is None else f"{kind} #{issue}"
    return {
        "kind": kind,
        "issue": issue,
        "label": label,
        "note": note,
        "candidate": candidate,
    }


def decide(finding: Mapping[str, Any], issues: Sequence[KnownIssue]) -> dict[str, Any]:
    """Apply the de-duplication table to one merged finding.

    Args:
        finding: Needs ``fingerprint``, ``category``, ``severity`` and
            ``classification``.
        issues: Existing issues. Only accepted ones decide; issues closed as
            duplicates are skipped. With several accepted matches, an open one
            wins, then the highest-numbered (newest) closed one.

    Returns:
        ``{"kind", "issue", "label", "note", "candidate"}``.
    """
    fingerprint = finding["fingerprint"]
    mentions = [
        issue
        for issue in issues
        if fingerprint in issue.fingerprints and issue.state_reason != "DUPLICATE"
    ]
    matches = sorted(
        (issue for issue in mentions if issue.accepted),
        key=lambda issue: issue.number,
        reverse=True,
    )
    unaccepted = sorted(issue.number for issue in mentions if not issue.accepted)
    if not matches and unaccepted:
        numbers = ", ".join(f"#{number}" for number in unaccepted[:5])
        if len(unaccepted) > 5:
            numbers += f" and {len(unaccepted) - 5} more"
        # Recorded-only findings never become candidates, so they need no note.
        if not (
            finding["severity"] in ("info", "low")
            and finding.get("classification") == "harmless"
        ):
            return _decision(
                NEW_KIND,
                None,
                True,
                f"mentioned by {numbers}, which no maintainer has accepted "
                f"(label {ACCEPTED_LABEL}); it does not track this finding",
            )
    open_matches = [issue for issue in matches if issue.state == "OPEN"]
    if open_matches:
        number = open_matches[0].number
        return _decision(
            DUPLICATE_OPEN_KIND,
            number,
            False,
            f"tracked by open issue #{number}; add new evidence there by hand",
        )
    if (
        finding["severity"] in ("info", "low")
        and finding.get("classification") == "harmless"
    ):
        return _decision(RECORDED_KIND, None, False)
    if not matches:
        return _decision(NEW_KIND, None, True)
    # Every match is closed here; the newest one decides.
    latest = matches[0]
    number = latest.number
    if latest.state_reason != "NOT_PLANNED":
        return _decision(
            REGRESSION_OF_KIND,
            number,
            True,
            f"matches #{number}, closed as completed: the problem is back",
        )
    if finding["category"] == "security":
        return _decision(
            NEW_KIND,
            number,
            True,
            f"matches #{number}, closed as not planned; security findings are "
            f"never suppressed",
            label=f"{NEW_KIND} (see #{number}, closed as not planned)",
        )
    return _decision(
        SUPPRESSED_KIND, number, False, f"closed as not planned in #{number}"
    )


# ---------------------------------------------------------------------------
# Building the report
# ---------------------------------------------------------------------------


def _check_order(check: str) -> int:
    """Sort key: checks in ``results.CHECK_IDS`` order, unknown ones last."""
    return _CHECK_RANK.get(check, 99)


def _run_order(run_: CheckRun) -> tuple[int, str, str]:
    """Sort key for check runs: by check, then variant, then source."""
    return (_check_order(run_.check), run_.variant, run_.source)


def _run_label(run_: CheckRun) -> str:
    """``check (variant)`` with the variant on one bounded line."""
    return f"{run_.check} ({_one_line(run_.variant, 120)})"


def _merge(runs: Sequence[CheckRun], legs: Sequence[Leg]) -> list[dict[str, Any]]:
    """One entry per fingerprint, listing every check and variant behind it."""
    occurrences: dict[str, list[tuple[Finding, CheckRun]]] = {}
    for run_ in sorted(runs, key=_run_order):
        for finding in run_.findings:
            if run_.check == "upstream":
                finding = resolve_upstream(finding, legs)
            occurrences.setdefault(finding.fingerprint, []).append((finding, run_))
    merged = []
    for fingerprint, items in occurrences.items():
        first = items[0][0]
        severity = _most_severe(finding.severity for finding, _ in items)
        classes = [f.classification for f, _ in items if f.classification is not None]
        classification = (
            min(classes, key=lambda name: _CLASSIFICATION_RANK.get(name, 99))
            if classes
            else None
        )
        reported_by: list[dict[str, str]] = []
        for _, run_ in items:
            row = {"check": run_.check, "variant": run_.variant, "source": run_.source}
            if row not in reported_by:
                reported_by.append(row)
        merged.append(
            {
                "fingerprint": fingerprint,
                "marker": f"{MARKER_PREFIX}{fingerprint}",
                "rule": first.rule,
                "category": first.category,
                "severity": severity,
                "classification": classification,
                "subject": first.subject,
                "title": first.title,
                "evidence": dict(first.evidence),
                "checks": sorted(
                    {row["check"] for row in reported_by}, key=_check_order
                ),
                "variants": sorted({row["variant"] for row in reported_by}),
                "reported_by": reported_by,
            }
        )
    merged.sort(
        key=lambda item: (
            _CATEGORY_RANK[item["category"]],
            _SEVERITY_RANK[item["severity"]],
            item["subject"],
            item["fingerprint"],
        )
    )
    return merged


def _is_advisory(item: Mapping[str, Any]) -> bool:
    return (
        item["category"] == "security"
        and item["rule"] == "advisory"
        and isinstance(item["evidence"].get("package"), str)
    )


def _vulnerabilities(findings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every advisory present, with its fix status and its tracking issue.

    An advisory in the results is unresolved by definition; ``tracked_by``
    names the accepted open issue that tracks it and changes nothing else.
    """

    def strings(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(part) for part in value]

    rows = []
    for item in findings:
        if not _is_advisory(item):
            continue
        evidence, decision = item["evidence"], item["decision"]
        versions = evidence.get("versions")
        advisory = evidence.get("advisory")
        rows.append(
            {
                "fingerprint": item["fingerprint"],
                "marker": item["marker"],
                "package": evidence["package"],
                "advisory": advisory if isinstance(advisory, str) else None,
                "versions": (
                    {str(k): str(v) for k, v in versions.items()}
                    if isinstance(versions, Mapping)
                    else {}
                ),
                "fix_versions": strings(evidence.get("fix_versions")),
                "seen_in": strings(evidence.get("seen_in")),
                "fix_in_declared_range": evidence.get("fix_in_declared_range"),
                "fix_reachable": evidence.get("fix_reachable"),
                "fixable": evidence.get("fixable_in_range") is True,
                "tracked_by": (
                    decision["issue"]
                    if decision["kind"] == DUPLICATE_OPEN_KIND
                    else None
                ),
            }
        )
    return rows


def candidate_groups(
    findings: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Candidates in paste order; advisories of one package share a block."""
    candidates = sorted(
        (item for item in findings if item["candidate"]),
        key=lambda item: (
            _SEVERITY_RANK[item["severity"]],
            _CATEGORY_RANK[item["category"]],
            item["subject"],
            item["fingerprint"],
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    by_package: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        if _is_advisory(item):
            package = item["evidence"]["package"]
            if package not in by_package:
                by_package[package] = []
                groups.append(by_package[package])
            by_package[package].append(dict(item))
        else:
            groups.append([dict(item)])
    for group in groups:
        group.sort(key=lambda item: (item["subject"], item["fingerprint"]))
    return groups


def run_metadata(environ: Mapping[str, str]) -> dict[str, Any]:
    """Timestamp plus the GitHub run coordinates present in ``environ``."""
    meta: dict[str, Any] = {"generated_at": results.utc_now()}
    for variable, key in _GITHUB_ENV:
        if environ.get(variable):
            meta[key] = environ[variable]
    if all(key in meta for key in ("server_url", "repository", "run_id")):
        meta["run_url"] = (
            f"{meta['server_url']}/{meta['repository']}/actions/runs/{meta['run_id']}"
        )
    return meta


def build_report(
    inputs: Inputs,
    expect: Sequence[str],
    issues: Optional[Sequence[KnownIssue]] = None,
    problems: Sequence[Mapping[str, str]] = (),
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Assemble the ``pmm-audit-report/1`` document (see module docstring).

    Args:
        inputs: What :func:`collect_inputs` read.
        expect: Check ids that must have at least one result, or
            ``check:variant`` entries for one matrix leg each.
        issues: Existing issues, or None when none were provided.
        problems: Problems found outside the results directory.
        environ: Environment for the run metadata (``GITHUB_*``).
    """
    all_problems = [dict(p) for p in (*inputs.problems, *problems)]
    present = {run_.check for run_ in inputs.runs}
    present_legs = {(run_.check, run_.variant) for run_ in inputs.runs}
    missing = [
        entry
        for entry in expect
        if (
            tuple(entry.split(":", 1)) not in present_legs
            if ":" in entry
            else entry not in present
        )
    ]
    ordered = sorted(inputs.runs, key=_run_order)
    # Variants come from result files and artifact names: one line, bounded,
    # before they reach the verdict text, the Markdown and stderr.
    errored = [_run_label(r) for r in ordered if r.status == "error"]
    failed = [_run_label(r) for r in ordered if r.status == "fail"]
    findings = _merge(inputs.runs, inputs.legs)
    matched: list[dict[str, Any]] = []
    known = issues or ()
    states = {issue.number: issue for issue in known}
    for item in findings:
        decision = decide(item, known)
        item["candidate"] = decision.pop("candidate")
        item["decision"] = decision
        if decision["issue"] is not None:
            issue = states[decision["issue"]]
            matched.append(
                {
                    "number": issue.number,
                    "state": issue.state,
                    "state_reason": issue.state_reason,
                    "fingerprint": item["fingerprint"],
                }
            )
    groups = candidate_groups(findings)
    candidates = [item["fingerprint"] for group in groups for item in group]
    kinds = [item["decision"]["kind"] for item in findings]
    vulnerabilities = _vulnerabilities(findings)
    # Tracking prevents a duplicate issue; it does not resolve anything.
    unresolved = [row["fingerprint"] for row in vulnerabilities if row["fixable"]]
    tracked = [
        row["fingerprint"]
        for row in vulnerabilities
        if row["fixable"] and row["tracked_by"] is not None
    ]
    reasons = []
    if missing:
        reasons.append(f"expected results missing: {', '.join(missing)}")
    if errored:
        reasons.append(f"checks that did not complete: {', '.join(errored)}")
    if all_problems:
        reasons.append(f"input problems: {len(all_problems)}")
    if failed:
        reasons.append(f"failed checks: {', '.join(failed)}")
    if unresolved:
        reasons.append(
            f"security advisories with a fix within the declared range are "
            f"still present: {len(unresolved)} ({len(tracked)} tracked by an "
            f"accepted open issue)"
        )
    exit_code = (
        2
        if (missing or errored or all_problems)
        else 1 if (failed or unresolved) else 0
    )
    checks = [
        {
            "check": run_.check,
            "variant": run_.variant,
            "status": run_.status,
            "source": run_.source,
            "findings": len(run_.findings),
            **({"tests": run_.tests} if run_.tests is not None else {}),
        }
        for run_ in ordered
    ]
    checks += [
        {
            "check": entry.split(":", 1)[0],
            "variant": entry.split(":", 1)[1] if ":" in entry else None,
            "status": "missing",
            "source": None,
            "findings": 0,
        }
        for entry in missing
    ]
    return {
        "schema": results.REPORT_SCHEMA,
        "run": run_metadata(environ or {}),
        "verdict": {
            "status": {0: "green", 1: "red", 2: "incomplete"}[exit_code],
            "exit_code": exit_code,
            "reasons": reasons,
            "failed_checks": failed,
            "errored_checks": errored,
            "unresolved_fixable_advisories": unresolved,
            "tracked_unresolved_advisories": tracked,
        },
        "counts": {
            "check_results": len(inputs.runs),
            "missing_checks": len(missing),
            "failed_checks": len(failed),
            "errored_checks": len(errored),
            "problems": len(all_problems),
            "findings": len(findings),
            "advisories": len(vulnerabilities),
            "unresolved_fixable_advisories": len(unresolved),
            "candidates": len(candidates),
            "candidate_blocks": len(groups),
            "already_tracked": kinds.count(DUPLICATE_OPEN_KIND),
            "suppressed": kinds.count(SUPPRESSED_KIND),
            "recorded_only": kinds.count(RECORDED_KIND),
        },
        "expected_checks": list(expect),
        "missing_checks": missing,
        "problems": all_problems,
        "checks": checks,
        "vulnerabilities": vulnerabilities,
        "findings": findings,
        "candidates": candidates,
        "known_issues": {
            "provided": issues is not None,
            "count": len(known),
            "matched": matched,
        },
        "environment": [
            {"check": run_.check, "variant": run_.variant, **run_.environment}
            for run_ in ordered
            if run_.environment
        ],
        "ignored_inputs": sorted(inputs.ignored),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_ESCAPE_RE = re.compile(r"([\\`*_{}\[\]<>()#+!|~])")


def _escape(text: Any) -> str:
    """Plain inline text with Markdown and HTML punctuation escaped."""
    return _ESCAPE_RE.sub(r"\\\1", _one_line(str(text)))


def _code(text: Any, limit: int = TITLE_LIMIT) -> str:
    """An inline code span holding any text literally (titles, subjects)."""
    text = _one_line(str(text), limit)
    if not text:
        return "(empty)"
    longest = max((len(run_) for run_ in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if longest else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _cell(text: Any, code: bool = False) -> str:
    """A table cell; ``|`` is escaped even inside code spans (GFM tables)."""
    if code:
        return _code(text).replace("|", "\\|")
    return _escape(text)


def _fence(content: str, info: str = "text", char: str = "`") -> str:
    """A fenced block no line of ``content`` can close early."""
    longest = max(
        (len(run_) for run_ in re.findall(re.escape(char) + "+", content)), default=0
    )
    fence = char * max(3, longest + 1)
    return f"{fence}{info}\n{content}\n{fence}"


def _plain_value(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif value == [] or value == {}:
        text = "(none)"
    elif isinstance(value, list) and all(
        value_ is None or isinstance(value_, (str, int, float, bool))
        for value_ in value
    ):
        text = ", ".join(str(value_) for value_ in value)
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return results.truncate_evidence(text, MD_VALUE_LIMIT)


def _evidence_text(evidence: Mapping[str, Any], skip: Sequence[str] = ()) -> str:
    lines = [
        f"{key}: {_plain_value(evidence[key])}"
        for key in sorted(evidence)
        if key not in skip and not (key == "excerpt" and isinstance(evidence[key], str))
    ]
    excerpt = evidence.get("excerpt")
    if isinstance(excerpt, str) and excerpt.strip():
        lines.append("excerpt:")
        lines.append(results.truncate_evidence(excerpt, MD_EXCERPT_LIMIT))
    return "\n".join(lines) if lines else "(no evidence recorded)"


def _summary_line(report: Mapping[str, Any]) -> str:
    counts, verdict = report["counts"], report["verdict"]
    return (
        f"Audit {verdict['status']} (exit {verdict['exit_code']}). "
        f"Check results: {counts['check_results']}. "
        f"Missing results: {counts['missing_checks']}. "
        f"Failed checks: {counts['failed_checks']}. "
        f"Errored checks: {counts['errored_checks']}. "
        f"Unresolved advisories with a fix in range: "
        f"{counts['unresolved_fixable_advisories']} "
        f"(tracked: {len(verdict['tracked_unresolved_advisories'])}). "
        f"Findings: {counts['findings']}. "
        f"Issue candidates: {counts['candidates']} "
        f"(paste blocks: {counts['candidate_blocks']}). "
        f"Already tracked: {counts['already_tracked']}. "
        f"Suppressed: {counts['suppressed']}. "
        f"Recorded only: {counts['recorded_only']}."
    )


def _run_line(report: Mapping[str, Any]) -> str:
    run_ = report["run"]
    parts = [f"Run: {run_['run_url']}" if "run_url" in run_ else "Run: local run"]
    if run_.get("sha"):
        parts.append(f"commit {_code(run_['sha'])}")
    parts.append(f"generated {run_['generated_at']}")
    known = report["known_issues"]
    parts.append(
        f"known issues read: {known['count']}"
        if known["provided"]
        else "known issues: not provided, so no de-duplication against issues"
    )
    if report["ignored_inputs"]:
        parts.append(f"ignored files: {len(report['ignored_inputs'])}")
    return "; ".join(parts) + "."


def _tests_cell(tests: Optional[Mapping[str, int]]) -> str:
    if not tests:
        return ""
    return (
        f"{tests['total']} run, {tests['failures']} failed, {tests['errors']} "
        f"errors, {tests['xfailed']} xfailed, {tests['skipped']} skipped"
    )


def _vulnerability_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """The vulnerabilities table: fix status and tracking side by side."""
    lines = ["", "## Vulnerabilities", ""]
    if not rows:
        return lines + ["No security advisory in this run."]
    lines += [
        "Every advisory listed is present in an audited environment. Tracking "
        "only stops a duplicate issue; a tracked advisory is still unresolved.",
        "",
        "| Package | Advisory | Installed | Fix versions | Fix within declared "
        "range | Tracking |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        installed = ", ".join(
            f"{version} ({label})" for label, version in sorted(row["versions"].items())
        )
        if row["fixable"]:
            fixable = "yes"
        elif row["fix_in_declared_range"] and row["fix_reachable"] is False:
            fixable = "no: the newest allowed resolution is still affected"
        else:
            fixable = "no"
        tracking = (
            f"#{row['tracked_by']} (open, accepted); still unresolved"
            if row["tracked_by"] is not None
            else "untracked"
        )
        lines.append(
            f"| {_cell(row['package'])} | {_cell(row['advisory'] or '?')} | "
            f"{_cell(installed or '?')} | "
            f"{_cell(', '.join(row['fix_versions']) or 'none')} | {fixable} | "
            f"{tracking} |"
        )
    return lines


def render_summary(report: Mapping[str, Any]) -> str:
    """Summary line, verdict, per-check table, problems and candidate index."""
    verdict = report["verdict"]
    lines = [
        "# Quality audit report",
        "",
        _summary_line(report),
        "",
        f"Verdict: **{verdict['status']}** (exit {verdict['exit_code']})"
        + (
            ": " + "; ".join(_escape(reason) for reason in verdict["reasons"])
            if verdict["reasons"]
            else ""
        )
        + ".",
        "",
        _run_line(report),
        "",
        "## Checks",
        "",
        "| Check | Variant | Status | Findings | Tests | Source |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["checks"]:
        lines.append(
            "| {check} | {variant} | {status} | {findings} | {tests} | {source} |".format(
                check=row["check"],
                variant=_cell(row["variant"] or "none"),
                status=row["status"],
                findings=row["findings"],
                tests=_tests_cell(row.get("tests")),
                source=_cell(row["source"] or "none"),
            )
        )
    if report["problems"]:
        lines += ["", "## Input problems", ""]
        lines += [
            f"- {_code(p['source'])}: {_code(p['problem'], 600)}"
            for p in report["problems"]
        ]
    lines += _vulnerability_lines(report["vulnerabilities"])
    lines += ["", "## Issue candidates", ""]
    by_fp = {item["fingerprint"]: item for item in report["findings"]}
    if report["candidates"]:
        lines += [
            "Each candidate is a paste block in issue-candidates.md. Nothing is "
            "filed automatically.",
            "",
            "| # | Fingerprint | Category | Severity | Decision | Title |",
            "|---|---|---|---|---|---|",
        ]
        for number, fingerprint in enumerate(report["candidates"], 1):
            item = by_fp[fingerprint]
            lines.append(
                f"| {number} | {_cell(item['marker'], code=True)} | "
                f"{item['category']} | {item['severity']} | "
                f"{item['decision']['label']} | "
                f"{_cell(item['title'], code=True)} |"
            )
    else:
        lines.append("No issue candidates in this run.")
    return "\n".join(lines) + "\n"


def _finding_section(item: Mapping[str, Any]) -> list[str]:
    reported = "; ".join(
        f"{row['check']} ({_escape(row['variant'])})" for row in item["reported_by"]
    )
    return [
        f"#### {_code(item['title'])}",
        "",
        f"- Fingerprint: {_code(item['marker'])}; rule {_code(item['rule'])}; "
        f"severity {item['severity']}; classification "
        f"{item['classification'] or 'none'}",
        f"- Subject: {_code(item['subject'])}",
        f"- Reported by: {reported}",
        # Decision labels and notes are composed here from a fixed vocabulary
        # and issue numbers, never from issue text.
        f"- Decision: {item['decision']['label']}"
        + (f" ({item['decision']['note']})" if item["decision"]["note"] else ""),
        "",
        f"Evidence from {item['reported_by'][0]['check']} "
        f"({_escape(item['reported_by'][0]['variant'])}):",
        "",
        _fence(_evidence_text(item["evidence"])),
        "",
    ]


def _index_table(
    items: Sequence[Mapping[str, Any]], report: Mapping[str, Any]
) -> list[str]:
    states = {
        (row["number"], row["fingerprint"]): row
        for row in report["known_issues"]["matched"]
    }
    lines = ["| Issue | State | Fingerprint | Finding |", "|---|---|---|---|"]
    for item in items:
        number = item["decision"]["issue"]
        row = states.get((number, item["fingerprint"]), {})
        state = row.get("state", "")
        if row.get("state_reason"):
            state += f" ({row['state_reason']})"
        lines.append(
            f"| #{number} | {_cell(state)} | {_cell(item['marker'], code=True)} | "
            f"{_cell(item['title'], code=True)} |"
        )
    return lines


def render_report(report: Mapping[str, Any]) -> str:
    """The full Markdown report (the summary plus the detail sections)."""
    lines = [render_summary(report).rstrip("\n"), "", "## Findings by category", ""]
    findings = report["findings"]
    if not findings:
        lines += ["No findings.", ""]
    for category in results.CATEGORIES:
        items = [item for item in findings if item["category"] == category]
        if not items:
            continue
        lines += [f"### {category} ({len(items)})", ""]
        for item in items:
            lines += _finding_section(item)
    lines += ["## Already tracked (open issues a maintainer accepted)", ""]
    tracked = [i for i in findings if i["decision"]["kind"] == DUPLICATE_OPEN_KIND]
    lines += (_index_table(tracked, report) if tracked else ["None."]) + [""]
    lines += ["## Suppressed (closed as not planned)", ""]
    suppressed = [i for i in findings if i["decision"]["kind"] == SUPPRESSED_KIND]
    lines += (_index_table(suppressed, report) if suppressed else ["None."]) + [""]
    lines += ["## Recorded only (info or low, harmless)", ""]
    recorded = [i for i in findings if i["decision"]["kind"] == RECORDED_KIND]
    lines += [f"- {_code(i['marker'])} {_code(i['title'])}" for i in recorded] or [
        "None."
    ]
    lines += ["", "## Environment", ""]
    if report["environment"]:
        lines += [
            "| Check | Variant | Python | Platform | "
            + " | ".join(_ENV_COLUMNS)
            + " |",
            "|---|---|---|---|" + "---|" * len(_ENV_COLUMNS),
        ]
        for row in report["environment"]:
            packages = row.get("packages") or {}
            cells = [
                row["check"],
                _cell(row["variant"]),
                _cell(row.get("python") or "?"),
                _cell(row.get("platform") or "?"),
                *(_cell(packages.get(name) or "?") for name in _ENV_COLUMNS),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("No environment recorded.")
    return "\n".join(lines) + "\n"


def _reproduction(group: Sequence[Mapping[str, Any]]) -> str:
    lead = group[0]
    evidence = lead["evidence"]
    check = lead["checks"][0]
    commands: list[str] = []
    if lead["rule"] in (TEST_FAILED_RULE, KNOWN_DEFECT_RULE) and isinstance(
        evidence.get("test"), str
    ):
        flags = " -rxX" if lead["rule"] == KNOWN_DEFECT_RULE else ""
        commands.append(f'python -m pytest "{evidence["test"]}"{flags}')
    elif check == "smoke":
        commands.append("python -m tools.audit smoke")
    elif check == "package-smoke":
        commands += [
            "python -m build",
            "python -m tools.audit package-smoke --dist dist --artifact wheel",
        ]
    elif check == "artifact-licensing":
        commands += [
            "python -m build",
            "python -m tools.audit artifact-licensing --dist dist",
        ]
    elif check == "upstream":
        commands.append("python -m tools.audit upstream")
    elif check == "security":
        commands += [
            "python -m tools.audit lock-reqs --output lock-reqs.txt",
            "pip-audit -r lock-reqs.txt --no-deps --disable-pip --format json "
            "--output lock.json",
            "# in a fresh environment with the project installed:",
            "pip-audit --format json --output fresh.json",
            "python -m tools.audit security --lock lock.json --fresh fresh.json",
        ]
    else:
        commands.append("python -m pytest")
    recorded = evidence.get("command")
    if isinstance(recorded, list) and all(isinstance(part, str) for part in recorded):
        recorded = " ".join(recorded)
    if isinstance(recorded, str) and recorded.strip():
        commands += ["# command recorded by the check:", _one_line(recorded, 1000)]
    return _fence("\n".join(commands), "shell")


def _block(
    index: int,
    total: int,
    group: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> str:
    lead = group[0]
    advisory = _is_advisory(lead)
    if advisory:
        package = lead["evidence"]["package"]
        versions: set[str] = set()
        for item in group:
            recorded = item["evidence"].get("versions")
            if isinstance(recorded, Mapping):
                versions.update(str(v) for v in recorded.values() if v)
        count = len(group)
        noun = "advisory" if count == 1 else "advisories"
        version_text = ", ".join(sorted(versions, key=_version_key)) or "?"
        title = f"{package} {version_text}: {count} {noun}"
    else:
        title = lead["title"]
    issue_title = f"{ISSUE_TITLE_PREFIX}{_one_line(title)} ({lead['marker']})"

    by_check: dict[str, list[str]] = {}
    for item in group:
        for row in item["reported_by"]:
            variants = by_check.setdefault(row["check"], [])
            if row["variant"] not in variants:
                variants.append(row["variant"])
    check_text = "; ".join(
        f"{check}: {', '.join(sorted(variants))}"
        for check, variants in by_check.items()
    )
    severity = _most_severe(item["severity"] for item in group)

    evidence_parts = []
    for item in group:
        part = []
        if advisory:
            part.append(f"Audit fingerprint: {item['marker']}")
        if item["decision"]["note"]:
            part.append(f"triage note: {item['decision']['note']}")
        if advisory or item["classification"]:
            part.append(f"classification: {item['classification'] or 'none'}")
        part.append(
            _evidence_text(item["evidence"], skip=("package",) if advisory else ())
        )
        evidence_parts.append("\n".join(part))
    evidence_text = "\n\n".join(evidence_parts)

    run_ = report["run"]
    sections = {
        "fingerprint": " ".join(item["marker"] for item in group),
        "category": lead["category"],
        "severity": severity,
        "check": _one_line(check_text, 1000),
        "evidence": _fence(evidence_text, "text"),
        "reproduction": _reproduction(group),
        "run-link": run_.get("run_url", "local run"),
        "duplicate-search": "- [ ] I searched existing issues, open and closed, "
        "for every pmm-audit: marker above.",
    }
    body = [f"Audit fingerprint: {item['marker']}" for item in group]
    for field_id, label in CANDIDATE_FIELDS:
        body += ["", f"### {label}", "", sections[field_id]]
    body_text = "\n".join(body)

    labels = sorted({item["decision"]["label"] for item in group})
    return "\n".join(
        [
            f"## Candidate {index} of {total}",
            "",
            f"Decision: {'; '.join(labels)}.",
            "",
            "Title:",
            "",
            _fence(issue_title, "text"),
            "",
            "Body:",
            "",
            _fence(body_text, "markdown", char="~"),
            "",
        ]
    )


def render_candidates(report: Mapping[str, Any]) -> str:
    """``issue-candidates.md``: one ready-to-paste block per candidate."""
    groups = candidate_groups(report["findings"])
    lines = [
        "# Issue candidates",
        "",
        _summary_line(report),
        "",
        "File each block by hand with the Audit finding issue form, after "
        "searching existing issues (open and closed) for its pmm-audit: "
        "markers. Nothing here has been filed. Advisories for one package "
        "share a block; keep every marker so later runs recognize each one.",
        "",
    ]
    if not groups:
        lines.append("No issue candidates in this run.")
        return "\n".join(lines) + "\n"
    blocks = [
        _block(index, len(groups), group, report)
        for index, group in enumerate(groups, 1)
    ]
    return "\n".join(lines) + "\n" + "\n".join(blocks)


# ---------------------------------------------------------------------------
# Issue form parity
# ---------------------------------------------------------------------------


def template_parity(template: Any, candidate_labels: Sequence[str]) -> list[str]:
    """Problems between the issue form and the candidate block fields.

    Args:
        template: The parsed ``audit_finding.yml``.
        candidate_labels: The ``### <label>`` sections a candidate block has.

    Returns:
        One message per problem; empty when the two agree. Checked: the
        title prefix and label, a required ``fingerprint`` input naming the
        ``pmm-audit:`` prefix, category and severity dropdowns equal to the
        ``results.py`` vocabularies both ways, and the field labels equal to
        the candidate sections both ways.
    """
    if not isinstance(template, Mapping):
        return ["the issue template is not a mapping"]
    problems = []
    title = template.get("title")
    if not (isinstance(title, str) and title.startswith(ISSUE_TITLE_PREFIX.strip())):
        problems.append(f"title {title!r} does not start with {ISSUE_TITLE_PREFIX!r}")
    labels = template.get("labels")
    if not (isinstance(labels, list) and "audit" in labels):
        problems.append("labels do not include 'audit'")
    body = template.get("body")
    if not isinstance(body, list):
        return problems + ["the template body is not a list"]
    fields: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(body):
        if not isinstance(item, Mapping):
            problems.append(f"body[{index}] is not a mapping")
            continue
        if item.get("type") == "markdown":
            continue
        field_id = item.get("id")
        if not isinstance(field_id, str):
            problems.append(f"body[{index}] ({item.get('type')}) has no id")
            continue
        fields[field_id] = item

    def attributes(item: Mapping[str, Any]) -> Mapping[str, Any]:
        value = item.get("attributes")
        return value if isinstance(value, Mapping) else {}

    fingerprint = fields.get("fingerprint")
    if fingerprint is None:
        problems.append("the template has no 'fingerprint' field")
    else:
        if fingerprint.get("type") != "input":
            problems.append("the fingerprint field is not an input")
        validations = fingerprint.get("validations")
        if not (
            isinstance(validations, Mapping) and validations.get("required") is True
        ):
            problems.append("the fingerprint field is not required")
        if MARKER_PREFIX not in str(attributes(fingerprint).get("description", "")):
            problems.append(
                f"the fingerprint description does not name {MARKER_PREFIX}"
            )
    for field_id, vocabulary in (
        ("category", results.CATEGORIES),
        ("severity", results.SEVERITIES),
    ):
        item = fields.get(field_id)
        if item is None or item.get("type") != "dropdown":
            problems.append(f"the template has no '{field_id}' dropdown")
            continue
        options = attributes(item).get("options")
        options = list(options) if isinstance(options, list) else []
        lacking = [value for value in vocabulary if value not in options]
        extra = [value for value in options if value not in vocabulary]
        if lacking:
            problems.append(f"{field_id} options lack {lacking}")
        if extra:
            problems.append(f"{field_id} options {extra} are not in the vocabulary")
    template_labels = [str(attributes(item).get("label")) for item in fields.values()]
    for label in template_labels:
        if label not in candidate_labels:
            problems.append(f"template field {label!r} has no candidate block section")
    for label in candidate_labels:
        if label not in template_labels:
            problems.append(f"candidate block section {label!r} has no template field")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_expect(text: Optional[str]) -> list[str]:
    if text is None:
        return list(results.EXPECTED_WEEKLY_CHECKS)
    expect: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if part and part not in expect:
            expect.append(part)
    unknown = [
        entry
        for entry in expect
        if entry.split(":", 1)[0] not in results.CHECK_IDS
        or (":" in entry and not entry.split(":", 1)[1])
    ]
    if unknown:
        raise ValueError(
            f"--expect names unknown checks or empty variants {unknown}; known "
            f"checks: {list(results.CHECK_IDS)}"
        )
    if not expect:
        raise ValueError("--expect names no check")
    return expect


def write_outputs(report: Mapping[str, Any], out: Path) -> list[str]:
    """Write the three outputs; return one message per output that failed."""
    errors = []
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [f"cannot create {out}: {exc}"]
    writers = (
        (REPORT_JSON, lambda path: results.write_json(report, path)),
        (
            REPORT_MD,
            lambda path: results.write_text_atomic(render_report(report), path),
        ),
        (
            CANDIDATES_MD,
            lambda path: results.write_text_atomic(render_candidates(report), path),
        ),
    )
    for name, write in writers:
        try:
            write(out / name)
        except (OSError, ValueError) as exc:
            errors.append(f"could not write {out / name}: {exc}")
    return errors


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the ``report`` subcommand's flags."""
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="directory of downloaded result artifacts, searched recursively",
    )
    parser.add_argument(
        "--expect",
        default=None,
        help="comma-separated check ids, or check:variant for one matrix leg, "
        "that must have a result (default: "
        f"{','.join(results.EXPECTED_WEEKLY_CHECKS)})",
    )
    parser.add_argument(
        "--known-issues",
        type=Path,
        default=None,
        help="output of gh issue list --state all --json "
        f"number,title,state,stateReason,body,labels; only issues labelled "
        f"{ACCEPTED_LABEL} track findings",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=f"directory for {REPORT_JSON}, {REPORT_MD} and {CANDIDATES_MD}",
    )
    parser.add_argument(
        "--github-summary",
        type=Path,
        default=None,
        help="append the Markdown summary to this file (GITHUB_STEP_SUMMARY)",
    )


def run(args: argparse.Namespace) -> int:
    """``report`` subcommand: read, decide, write, and return the verdict."""
    expect = _parse_expect(args.expect)
    inputs = collect_inputs(Path(args.results))
    problems: list[dict[str, str]] = []
    issues: Optional[list[KnownIssue]] = None
    if args.known_issues is not None:
        path = Path(args.known_issues)
        try:
            issues = parse_known_issues(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(_problem(str(path), "the known-issues file does not exist"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            problems.append(_problem(str(path), str(exc)))
    report = build_report(inputs, expect, issues, problems, os.environ)
    errors = write_outputs(report, Path(args.out))
    if args.github_summary is not None:
        try:
            with open(args.github_summary, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(render_summary(report))
        except OSError as exc:
            errors.append(f"could not append to {args.github_summary}: {exc}")
    for problem in report["problems"]:
        print(f"error: {problem['source']}: {problem['problem']}", file=sys.stderr)
    for entry in report["missing_checks"]:
        print(f"error: expected result {entry!r} is missing", file=sys.stderr)
    for check in report["verdict"]["errored_checks"]:
        print(f"error: check {check} did not complete", file=sys.stderr)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    print(_summary_line(report))
    print(f"outputs: {Path(args.out)}")
    code: int = report["verdict"]["exit_code"]
    return 2 if errors else code
