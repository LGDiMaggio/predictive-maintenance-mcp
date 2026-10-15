"""Guards for the audit report builder (``python -m tools.audit report``).

The report job is the last step of the weekly audit. It reads every check's
``pmm-audit-result/1`` file, the suite legs' JUnit XML and ``pip list`` JSON,
and the existing audit issues, then writes ``audit-report.json``,
``audit-report.md`` and ``issue-candidates.md``.

What these tests pin, in the order a reviewer would ask about it:

* de-duplication follows the table in the report module's docstring: one
  finding per fingerprint across variants, ``duplicate-open`` /
  ``regression-of`` / ``suppressed`` against existing issues, security
  findings never suppressed, info or low harmless findings recorded only;
* the builder fails closed (exit 2) on every empty or unreadable input: no
  results, a missing expected check, malformed JSON, a JUnit file with no
  test cases, a known-issues list long enough to be truncated;
* the weekly verdict (exit 1) is a fixable advisory that no open issue
  tracks;
* test-gated upstream findings are resolved from the legs that installed
  the candidate version;
* text from issues and test output cannot shape the rendered Markdown, and
  no waveform-sized numeric array reaches the report;
* the issue form and the candidate blocks declare the same fields.

``tests/fixtures/audit/report`` holds a trimmed real upstream result, a JUnit
file shaped like pytest's own output (``test_real_pytest_junit_*`` checks the
parser against a fresh pytest run), ``pip list`` JSON per leg, and a
known-issues list in the ``gh issue list --json`` shape.
"""

import json
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from tools.audit import __main__ as audit_main
from tools.audit import deps, report, results, security
from tools.audit.results import Finding, Result

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audit"
REPORT_FIXTURES = FIXTURES / "report"
TEMPLATE = REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "audit_finding.yml"

#: Fingerprints of the findings in ``junit_sample.xml`` (see known_issues.json).
FP_NAN = results.fingerprint(
    "diagnostic-correctness", "known-defect", "nan-sample-accepted"
)
FP_INF = results.fingerprint(
    "diagnostic-correctness", "known-defect", "sampling-rate-inf-accepted"
)
FP_CONTRACT = results.fingerprint(
    "tool-contract",
    "test-failed",
    "tests/test_tool_contracts.py::test_output_schema_frozen",
)
FP_LEDGER = results.fingerprint(
    "regression",
    "test-failed",
    "tests/test_ledger.py::TestAppend::test_append_is_atomic",
)
FP_DOCS = results.fingerprint(
    "doc-drift",
    "test-failed",
    "tests/test_install_docs.py::test_extras_named_in_docs_exist",
)

FAILURE = '<failure message="AssertionError: boom">trace</failure>'

#: The ``### <label>`` sections a candidate block renders, in order.
CANDIDATE_LABELS = [label for _, label in report.CANDIDATE_FIELDS]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _result(check, variant, findings=(), status="pass", facts=None, packages=None):
    return Result(
        check=check,
        variant=variant,
        status=status,
        started_at="2026-10-15T00:00:00Z",
        finished_at="2026-10-15T00:00:01Z",
        environment={
            "python": "3.12.11",
            "platform": "linux-x86_64",
            "packages": packages or {"mcp": "2.3.0", "numpy": "2.5.4"},
        },
        facts=facts or {},
        findings=list(findings),
    )


def _finding(
    rule="tool-missing",
    category="tool-contract",
    severity="high",
    subject="analyze_fft",
    title="analyze_fft is not listed by the server",
    classification=None,
    **evidence,
):
    return Finding(
        rule=rule,
        category=category,
        severity=severity,
        subject=subject,
        title=title,
        classification=classification,
        evidence=evidence,
    )


def _advisory(package, advisory, fixable=True, version="2.10.1"):
    return Finding(
        rule="advisory",
        category="security",
        severity="high" if fixable else "medium",
        subject=f"{package} {advisory}",
        title=f"{package} {version} is affected by {advisory}",
        classification="breaking-candidate" if fixable else "review",
        evidence={
            "package": package,
            "advisory": advisory,
            "versions": {"fresh": version, "lock": version},
            "fix_versions": ["99.0.0"],
            "aliases": [f"CVE-2026-{advisory[-4:]}"],
            "seen_in": ["fresh", "lock"],
            "declared": None,
            "fixable_in_range": fixable,
            "excerpt": f"Description of {advisory}.",
        },
    )


def _test_gated(package="numpy", candidates=("2.5.4",)):
    return Finding(
        rule="upstream-minor",
        category="dependency",
        severity="low",
        subject=package,
        title=f"{package} {', '.join(candidates)} is newer than reviewed",
        classification="test-gated",
        evidence={
            "package": package,
            "reviewed": "2.3.3",
            "declared": ">=2.3.3",
            "latest": {"3.12": candidates[-1]},
            "candidates": list(candidates),
            "interpreters": ["3.12"],
        },
    )


def _mcp_review():
    return Finding(
        rule="upstream-minor",
        category="mcp-compat",
        severity="medium",
        subject="mcp 2.4",
        title="mcp 2.4.0 is a new minor over reviewed 2.3.0: review its behavior",
        classification="review",
        evidence={
            "package": "mcp",
            "reviewed": "2.3.0",
            "declared": ">=2.0.0,<3.0.0",
            "candidates": ["2.4.0"],
            "interpreters": ["3.12"],
        },
    )


def _junit_sample():
    """``junit_sample.xml``: failures, an error and audit tripwires."""
    return (REPORT_FIXTURES / "junit_sample.xml").read_text(encoding="utf-8")


def _junit(*cases):
    """JUnit text in pytest's shape from (classname, name, inner XML)."""
    body = "".join(
        f'<testcase classname="{c}" name="{n}" time="0.001">{inner}</testcase>'
        for c, n, inner in cases
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(cases)}">{body}</testsuite>'
        "</testsuites>"
    )


PASSING_JUNIT = _junit(("tests.test_ledger", "test_ok", ""))


def _pip_list(**versions):
    return json.dumps(
        [{"name": name.replace("_", "-"), "version": v} for name, v in versions.items()]
    )


def _leg(root, variant, junit_text=PASSING_JUNIT, pip_list=None):
    leg = root / variant
    leg.mkdir(parents=True, exist_ok=True)
    (leg / "junit.xml").write_text(junit_text, encoding="utf-8")
    (leg / "pip-list.json").write_text(
        pip_list if pip_list is not None else _pip_list(numpy="2.5.4", mcp="2.3.0"),
        encoding="utf-8",
    )
    return leg


def _write(root, relative, result):
    return results.write_json(result, root / relative)


def _issue(number, fp, state="OPEN", reason=None):
    return {
        "number": number,
        "title": f"[Audit] something (pmm-audit:{fp})",
        "state": state,
        "stateReason": reason,
        "body": "Filed from the audit report.",
    }


def _run(tmp_path, results_dir, expect="suite", known=None, extra=()):
    out = tmp_path / "out"
    argv = [
        "report",
        "--results",
        str(results_dir),
        "--out",
        str(out),
        "--expect",
        expect,
    ]
    if known is not None:
        path = tmp_path / "known-issues.json"
        if isinstance(known, (str, Path)) and Path(known).is_file():
            shutil.copyfile(known, path)
        else:
            path.write_text(json.dumps(known), encoding="utf-8")
        argv += ["--known-issues", str(path)]
    argv += list(extra)
    code = audit_main.main(argv)
    return code, out


def _report_json(out):
    return json.loads((out / "audit-report.json").read_text(encoding="utf-8"))


def _md(out, name="audit-report.md"):
    return (out / name).read_text(encoding="utf-8")


def _by_fp(doc):
    return {item["fingerprint"]: item for item in doc["findings"]}


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _blocks(candidates_md):
    """The candidate blocks of issue-candidates.md, split on real headings.

    Fence-aware, like a Markdown renderer: a ``## Candidate`` line inside
    fenced evidence does not start a block.
    """
    blocks, fence = [], None
    for line in candidates_md.splitlines():
        match = _FENCE.match(line)
        if fence is None and match:
            if not (match.group(1)[0] == "`" and "`" in match.group(2)):
                fence = (match.group(1)[0], len(match.group(1)))
        elif (
            fence is not None
            and match
            and match.group(1)[0] == fence[0]
            and len(match.group(1)) >= fence[1]
            and not match.group(2).strip()
        ):
            fence = None
        elif fence is None and line.startswith("## Candidate "):
            blocks.append([])
        if blocks:
            blocks[-1].append(line)
    return ["\n".join(block) for block in blocks]


def _headings(block):
    return re.findall(r"(?m)^### (.+?)\s*$", block)


def _outside_fences(text):
    """Lines a CommonMark renderer would not treat as fenced code."""
    kept, fence = [], None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None:
            if match and not (match.group(1)[0] == "`" and "`" in match.group(2)):
                fence = (match.group(1)[0], len(match.group(1)))
                continue
            kept.append(line)
        elif (
            match
            and match.group(1)[0] == fence[0]
            and len(match.group(1)) >= fence[1]
            and not match.group(2).strip()
        ):
            fence = None
    return kept


def _fenced_bodies(text):
    """Contents of the top-level fenced blocks of ``text``, in order."""
    bodies, fence, current = [], None, []
    for line in text.splitlines():
        match = _FENCE.match(line)
        if fence is None:
            if match:
                fence, current = (match.group(1)[0], len(match.group(1))), []
        elif (
            match
            and match.group(1)[0] == fence[0]
            and len(match.group(1)) >= fence[1]
            and not match.group(2).strip()
        ):
            bodies.append("\n".join(current))
            fence = None
        else:
            current.append(line)
    return bodies


# ---------------------------------------------------------------------------
# JUnit parsing
# ---------------------------------------------------------------------------


class TestJUnit:
    def sample(self):
        return report.parse_junit(
            _junit_sample(), "suite-floors-py3.11", "floors/junit.xml"
        )

    def test_a_leg_with_failures_or_errors_fails_and_counts_its_cases(self):
        run = self.sample()
        assert (run.check, run.variant, run.status) == (
            "suite",
            "suite-floors-py3.11",
            "fail",
        )
        assert run.tests == {
            "total": 10,
            "passed": 2,
            "failures": 3,
            "errors": 1,
            "skipped": 1,
            "xfailed": 3,
        }

    def test_failures_become_findings_keyed_on_the_stable_node_id(self):
        findings = {f.fingerprint: f for f in self.sample().findings}
        assert set(findings) == {FP_CONTRACT, FP_DOCS, FP_LEDGER, FP_NAN, FP_INF}
        contract = findings[FP_CONTRACT]
        assert contract.category == "tool-contract"
        # Parametrized cases of one test are one finding listing the cases.
        assert contract.evidence["cases"] == ["analyze_envelope", "analyze_fft"]
        assert "output schema differs" in contract.title
        assert findings[FP_LEDGER].evidence["outcome"] == "error"
        assert findings[FP_DOCS].category == "doc-drift"

    def test_prefixed_xfail_is_a_finding_and_a_plain_xfail_is_ignored(self):
        findings = {f.fingerprint: f for f in self.sample().findings}
        nan = findings[FP_NAN]
        assert (nan.category, nan.rule, nan.subject) == (
            "diagnostic-correctness",
            "known-defect",
            "nan-sample-accepted",
        )
        assert nan.title.startswith("a CSV with one NaN sample loads")
        assert nan.evidence["test"] == (
            "tests/test_diagnostic_invariants.py::TestInvalidInputs"
            "::test_nan_sample_is_refused"
        )
        assert not any("rounding" in f.title for f in findings.values())

    @pytest.mark.parametrize(
        "module, category",
        [
            ("test_tool_contracts", "tool-contract"),
            ("test_doc_drift", "doc-drift"),
            ("test_install_docs", "doc-drift"),
            ("test_diagnostic_invariants", "diagnostic-correctness"),
            ("test_ledger", "regression"),
            ("test_tool_contracts_extra", "regression"),
        ],
    )
    def test_failure_category_follows_the_test_module(self, module, category):
        text = _junit((f"tests.{module}.TestX", "test_y[a-b]", FAILURE))
        (finding,) = report.parse_junit(text, "leg").findings
        assert finding.category == category
        assert finding.subject == f"tests/{module}.py::TestX::test_y"

    def test_a_collection_error_is_a_finding(self):
        text = _junit(
            ("", "tests.test_doc_drift", '<error message="collection failure"/>')
        )
        run = report.parse_junit(text, "leg")
        (finding,) = run.findings
        assert (finding.category, finding.subject) == (
            "doc-drift",
            "tests/test_doc_drift.py",
        )
        assert run.status == "fail"

    def test_a_passing_leg_has_no_findings(self):
        run = report.parse_junit(PASSING_JUNIT, "leg")
        assert run.status == "pass" and run.findings == []

    def test_zero_test_cases_are_refused(self):
        empty = (
            '<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="0">'
            "</testsuite></testsuites>"
        )
        with pytest.raises(ValueError, match="no test cases"):
            report.parse_junit(empty, "leg")

    def test_text_that_is_not_junit_is_refused(self):
        with pytest.raises(ValueError):
            report.parse_junit("not xml at all", "leg")
        with pytest.raises(ValueError, match="testsuite"):
            report.parse_junit("<html><body/></html>", "leg")

    def test_real_pytest_junit_lifts_only_prefixed_xfails(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        ini = tmp_path / "pytest.ini"
        ini.write_text("[pytest]\n", encoding="utf-8")
        (tests_dir / "test_diagnostic_invariants.py").write_text(
            textwrap.dedent("""
                import pytest

                def test_passes():
                    pass

                @pytest.mark.xfail(
                    strict=True,
                    raises=AssertionError,
                    reason="audit-known-defect: probe-defect: accepts what it "
                    "should refuse",
                )
                def test_known_defect():
                    assert False

                @pytest.mark.xfail(reason="not an audit tripwire")
                def test_plain():
                    assert False
                """),
            encoding="utf-8",
        )
        junit = tmp_path / "junit.xml"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-c",
                str(ini),
                "--rootdir",
                str(tmp_path),
                "-q",
                str(tests_dir),
                f"--junitxml={junit}",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        run = report.parse_junit(junit.read_text(encoding="utf-8"), "probe")
        assert [(f.category, f.rule, f.subject) for f in run.findings] == [
            ("diagnostic-correctness", "known-defect", "probe-defect")
        ]
        assert run.findings[0].title == "accepts what it should refuse"
        assert (run.status, run.tests["xfailed"], run.tests["passed"]) == (
            "pass",
            2,
            1,
        )


class TestPipList:
    def test_names_are_normalized(self):
        text = (REPORT_FIXTURES / "pip_list_newest.json").read_text(encoding="utf-8")
        versions = report.parse_pip_list(text)
        assert versions["scikit-learn"] == "1.9.1"
        assert versions["numpy"] == "2.5.4"

    @pytest.mark.parametrize(
        "text", ["", "{}", "[]", '[{"name": "numpy"}]', "not json"]
    )
    def test_malformed_or_empty_listings_are_refused(self, text):
        with pytest.raises(ValueError):
            report.parse_pip_list(text)


# ---------------------------------------------------------------------------
# Upstream join
# ---------------------------------------------------------------------------


class TestUpstreamJoin:
    def test_all_installing_legs_passed_resolves_harmless(self):
        legs = [
            report.Leg("suite", "suite-newest-py3.12", "pass", {"numpy": "2.5.4"}),
            report.Leg("package-smoke", "wheel-macos", "pass", {"numpy": "2.5.4"}),
            # Did not install the candidate: its failure says nothing about it.
            report.Leg("suite", "suite-floors-py3.11", "fail", {"numpy": "2.0.0"}),
        ]
        original = _test_gated()
        resolved = report.resolve_upstream(original, legs)
        assert resolved.classification == "harmless"
        assert resolved.fingerprint == original.fingerprint
        assert [leg["variant"] for leg in resolved.evidence["legs"]] == [
            "suite-newest-py3.12",
            "wheel-macos",
        ]

    def test_one_installing_leg_failed_resolves_breaking_candidate(self):
        legs = [
            report.Leg("suite", "suite-newest-py3.12", "pass", {"numpy": "2.5.4"}),
            report.Leg("package-smoke", "wheel-windows", "fail", {"numpy": "2.5.4"}),
        ]
        resolved = report.resolve_upstream(_test_gated(), legs)
        assert resolved.classification == "breaking-candidate"
        # A breaking-candidate carries the upstream major.minor.
        assert resolved.subject == "numpy 2.5"
        assert resolved.fingerprint == results.fingerprint(
            "dependency", "upstream-minor", "numpy 2.5"
        )
        assert resolved.evidence["failing_legs"] == ["package-smoke/wheel-windows"]

    def test_no_installing_leg_resolves_not_checked(self):
        legs = [report.Leg("suite", "suite-floors-py3.11", "pass", {"numpy": "2.0.0"})]
        resolved = report.resolve_upstream(_test_gated(), legs)
        assert resolved.classification == "not-checked"
        assert resolved.evidence["legs"] == []
        assert resolved.severity == "medium"

    def test_mcp_review_findings_get_the_leg_results_as_evidence(self):
        legs = [
            report.Leg("suite", "suite-newest-py3.12", "fail", {"mcp": "2.4.0"}),
            report.Leg("suite", "suite-floors-py3.11", "pass", {"mcp": "2.0.0"}),
        ]
        original = _mcp_review()
        resolved = report.resolve_upstream(original, legs)
        assert resolved.classification == "review"
        assert resolved.fingerprint == original.fingerprint
        assert resolved.evidence["legs"] == [
            {
                "check": "suite",
                "variant": "suite-newest-py3.12",
                "status": "fail",
                "version": "2.4.0",
            }
        ]

    def test_other_findings_pass_through_unchanged(self):
        finding = _finding()
        assert report.resolve_upstream(finding, []) is finding

    def test_end_to_end_with_suite_and_package_smoke_legs(self, tmp_path):
        res = tmp_path / "results"
        upstream_findings = [
            _test_gated("numpy", ("2.5.4",)),
            _test_gated("scipy", ("1.18.1",)),
            _test_gated("jsonschema", ("4.26.0",)),
            _test_gated("pydantic", ("2.99.0",)),
        ]
        _write(
            res,
            "upstream/upstream.json",
            _result("upstream", "upstream", upstream_findings),
        )
        _leg(
            res,
            "suite-newest-py3.12",
            PASSING_JUNIT,
            _pip_list(numpy="2.5.4", jsonschema="4.26.0"),
        )
        # The package-smoke leg's versions are embedded in its own result.
        _write(
            res,
            "package-smoke-ubuntu/package-smoke.json",
            _result(
                "package-smoke",
                "wheel-ubuntu-py3.12",
                status="fail",
                facts={"installed_versions": {"scipy": "1.18.1", "numpy": "2.5.4"}},
            ),
        )
        code, out = _run(tmp_path, res, expect="upstream,suite,package-smoke")
        by_subject = {f["subject"]: f for f in _report_json(out)["findings"]}
        assert by_subject["numpy 2.5"]["classification"] == "breaking-candidate"
        assert by_subject["scipy 1.18"]["classification"] == "breaking-candidate"
        assert by_subject["jsonschema"]["classification"] == "harmless"
        assert by_subject["pydantic"]["classification"] == "not-checked"
        assert [
            leg["variant"] for leg in by_subject["numpy 2.5"]["evidence"]["legs"]
        ] == [
            "suite-newest-py3.12",
            "wheel-ubuntu-py3.12",
        ]
        assert code == 0

    def test_harmless_resolution_is_recorded_only(self, tmp_path):
        res = tmp_path / "results"
        _write(
            res,
            "upstream/upstream.json",
            _result("upstream", "upstream", [_test_gated("numpy", ("2.5.4",))]),
        )
        _leg(res, "suite-newest-py3.12", PASSING_JUNIT, _pip_list(numpy="2.5.4"))
        code, out = _run(tmp_path, res, expect="upstream,suite")
        (numpy,) = _report_json(out)["findings"]
        assert numpy["classification"] == "harmless"
        assert numpy["decision"]["kind"] == "recorded"
        assert not numpy["candidate"]
        assert f"pmm-audit:{numpy['fingerprint']}" not in _md(
            out, "issue-candidates.md"
        )
        assert code == 0

    def test_recorded_upstream_fixture_joins_against_the_fixture_legs(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        shutil.copyfile(REPORT_FIXTURES / "result_upstream.json", res / "upstream.json")
        _leg(
            res,
            "suite-newest-py3.12",
            PASSING_JUNIT,
            (REPORT_FIXTURES / "pip_list_newest.json").read_text(encoding="utf-8"),
        )
        _leg(
            res,
            "suite-floors-py3.11",
            _junit_sample(),
            (REPORT_FIXTURES / "pip_list_floors.json").read_text(encoding="utf-8"),
        )
        code, out = _run(tmp_path, res, expect="upstream,suite")
        by_subject = {f["subject"]: f for f in _report_json(out)["findings"]}
        # numpy 2.5.4 was installed only by the passing newest leg.
        assert by_subject["numpy"]["classification"] == "harmless"
        assert by_subject["pandas 3.0"]["classification"] == "breaking-candidate"
        assert code == 0


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_two_variants_yield_one_candidate_listing_both(self, tmp_path):
        res = tmp_path / "results"
        missing = _finding()
        _write(
            res, "a/smoke.json", _result("smoke", "checkout-ubuntu-py3.11", [missing])
        )
        _write(
            res, "b/smoke.json", _result("smoke", "checkout-windows-py3.12", [missing])
        )
        code, out = _run(tmp_path, res, expect="smoke")
        (merged,) = _report_json(out)["findings"]
        assert merged["variants"] == [
            "checkout-ubuntu-py3.11",
            "checkout-windows-py3.12",
        ]
        candidates = _md(out, "issue-candidates.md")
        blocks = _blocks(candidates)
        assert len(blocks) == 1
        assert f"(pmm-audit:{missing.fingerprint})" in blocks[0]
        assert "checkout-ubuntu-py3.11" in blocks[0]
        assert "checkout-windows-py3.12" in blocks[0]
        assert code == 0

    def test_fixture_issues_decide_open_completed_and_not_planned(self, tmp_path):
        res = tmp_path / "results"
        sample = _junit_sample()
        _leg(res, "suite-floors-py3.11", sample)
        _leg(res, "suite-newest-py3.12", sample)
        code, out = _run(tmp_path, res, known=REPORT_FIXTURES / "known_issues.json")
        doc = _report_json(out)
        decisions = {fp: f["decision"] for fp, f in _by_fp(doc).items()}
        assert decisions[FP_NAN]["label"] == "duplicate-open #41"
        assert decisions[FP_CONTRACT]["label"] == "regression-of #38"
        assert decisions[FP_LEDGER]["label"] == "suppressed #35"
        assert decisions[FP_DOCS]["label"] == "new"
        assert decisions[FP_INF]["label"] == "new"
        assert set(doc["candidates"]) == {FP_CONTRACT, FP_DOCS, FP_INF}

        candidates = _md(out, "issue-candidates.md")
        assert f"pmm-audit:{FP_NAN}" not in candidates
        assert f"pmm-audit:{FP_LEDGER}" not in candidates
        (contract_block,) = [b for b in _blocks(candidates) if FP_CONTRACT in b]
        assert "regression-of #38" in contract_block

        md = _md(out)
        tracked = md.split("## Already tracked", 1)[1].split("\n## ", 1)[0]
        assert "#41" in tracked and f"pmm-audit:{FP_NAN}" in tracked
        suppressed = md.split("## Suppressed", 1)[1].split("\n## ", 1)[0]
        assert "#35" in suppressed and f"pmm-audit:{FP_LEDGER}" in suppressed
        # Merged across both legs.
        assert _by_fp(doc)[FP_NAN]["variants"] == [
            "suite-floors-py3.11",
            "suite-newest-py3.12",
        ]
        assert code == 0

    def test_issue_text_never_reaches_the_outputs(self, tmp_path):
        res = tmp_path / "results"
        _leg(res, "suite-floors-py3.11", _junit_sample())
        summary = tmp_path / "summary.md"
        _run(
            tmp_path,
            res,
            known=REPORT_FIXTURES / "known_issues.json",
            extra=["--github-summary", str(summary)],
        )
        out = tmp_path / "out"
        for text in (
            _md(out),
            _md(out, "issue-candidates.md"),
            _md(out, "audit-report.json"),
            summary.read_text(encoding="utf-8"),
        ):
            assert "UNTRUSTED-BODY-TEXT" not in text
            assert "IGNORE PREVIOUS" not in text
            assert "onerror" not in text

    def test_info_and_harmless_findings_are_reported_but_not_candidates(self, tmp_path):
        res = tmp_path / "results"
        info = _finding(
            rule="import-location",
            category="install",
            severity="info",
            subject="predictive_maintenance_mcp import location",
            title="predictive_maintenance_mcp imports from the recorded location",
            classification="harmless",
        )
        _write(res, "smoke/smoke.json", _result("smoke", "checkout", [info]))
        code, out = _run(tmp_path, res, expect="smoke")
        doc = _report_json(out)
        (item,) = doc["findings"]
        assert item["decision"]["kind"] == "recorded"
        assert doc["candidates"] == []
        assert info.fingerprint in _md(out)
        assert info.fingerprint not in _md(out, "issue-candidates.md")
        assert code == 0

    def test_decide_prefers_an_open_issue_over_a_closed_one(self):
        fp = _finding().fingerprint
        issues = report.parse_known_issues(
            json.dumps([_issue(5, fp, "CLOSED", "NOT_PLANNED"), _issue(9, fp, "OPEN")])
        )
        decision = report.decide(
            {
                "fingerprint": fp,
                "category": "tool-contract",
                "severity": "high",
                "classification": None,
            },
            issues,
        )
        assert decision["label"] == "duplicate-open #9"

    def test_a_marker_must_be_the_whole_fingerprint(self):
        fp = _finding().fingerprint
        issues = report.parse_known_issues(
            json.dumps([_issue(5, fp + "ff"), _issue(6, fp[:-1])])
        )
        finding = {
            "fingerprint": fp,
            "category": "tool-contract",
            "severity": "high",
            "classification": None,
        }
        assert report.decide(finding, issues)["kind"] == "new"


# ---------------------------------------------------------------------------
# Security verdict
# ---------------------------------------------------------------------------


class TestSecurityVerdict:
    def write(self, res, *findings):
        _write(
            res, "security/security.json", _result("security", "pip-audit", findings)
        )

    def test_untracked_fixable_advisory_exits_1_with_all_outputs(self, tmp_path):
        res = tmp_path / "results"
        advisory = _advisory("pyjwt", "PYSEC-2026-4140")
        self.write(res, advisory)
        code, out = _run(tmp_path, res, expect="security")
        assert code == 1
        for name in ("audit-report.json", "audit-report.md", "issue-candidates.md"):
            assert (out / name).is_file(), name
        doc = _report_json(out)
        assert doc["verdict"]["exit_code"] == 1
        assert advisory.fingerprint in _md(out, "issue-candidates.md")

    def test_fixable_advisory_tracked_by_an_open_issue_exits_0(self, tmp_path):
        res = tmp_path / "results"
        advisory = _advisory("pyjwt", "PYSEC-2026-4140")
        self.write(res, advisory, _advisory("pypdf", "PYSEC-2026-3910", fixable=False))
        code, out = _run(
            tmp_path, res, expect="security", known=[_issue(12, advisory.fingerprint)]
        )
        assert code == 0
        assert _by_fp(_report_json(out))[advisory.fingerprint]["decision"]["label"] == (
            "duplicate-open #12"
        )

    def test_closed_completed_does_not_silence_a_fixable_advisory(self, tmp_path):
        res = tmp_path / "results"
        advisory = _advisory("pyjwt", "PYSEC-2026-4140")
        self.write(res, advisory)
        code, _ = _run(
            tmp_path,
            res,
            expect="security",
            known=[_issue(12, advisory.fingerprint, "CLOSED", "COMPLETED")],
        )
        assert code == 1

    def test_not_planned_security_finding_is_still_a_candidate(self, tmp_path):
        res = tmp_path / "results"
        advisory = _advisory("pyjwt", "PYSEC-2026-4140")
        self.write(res, advisory)
        code, out = _run(
            tmp_path,
            res,
            expect="security",
            known=[_issue(17, advisory.fingerprint, "CLOSED", "NOT_PLANNED")],
        )
        item = _by_fp(_report_json(out))[advisory.fingerprint]
        assert item["candidate"] and item["decision"]["issue"] == 17
        assert item["decision"]["kind"] == "new"
        (block,) = _blocks(_md(out, "issue-candidates.md"))
        assert advisory.fingerprint in block and "#17" in block
        assert code == 1

    def test_advisories_for_one_package_share_one_block(self, tmp_path):
        res = tmp_path / "results"
        first = _advisory("pyjwt", "PYSEC-2026-4140")
        second = _advisory("pyjwt", "PYSEC-2026-4141")
        other = _advisory("pypdf", "PYSEC-2026-3910", version="6.1.0")
        self.write(res, second, first, other)
        _, out = _run(tmp_path, res, expect="security")
        blocks = _blocks(_md(out, "issue-candidates.md"))
        assert len(blocks) == 2
        (pyjwt,) = [b for b in blocks if "pyjwt" in b]
        assert (
            f"[Audit] pyjwt 2.10.1: 2 advisories (pmm-audit:{first.fingerprint})"
            in pyjwt
        )
        for advisory in (first, second):
            assert f"Audit fingerprint: pmm-audit:{advisory.fingerprint}" in pyjwt
        assert other.fingerprint not in pyjwt
        doc = _report_json(out)
        assert len(doc["candidates"]) == 3
        assert doc["counts"]["candidate_blocks"] == 2

    def test_security_result_from_the_security_normalizer(self, tmp_path):
        built = security.build_result(
            FIXTURES / "pip_audit_lock.json",
            FIXTURES / "pip_audit_fresh.json",
            deps.read_declared(REPO_ROOT / "pyproject.toml"),
        )
        res = tmp_path / "results"
        _write(res, "security/security.json", built)
        code, out = _run(tmp_path, res, expect="security")
        fixable = [
            f
            for f in built.findings
            if f.rule == "advisory" and f.evidence["fixable_in_range"]
        ]
        assert code == (1 if fixable else 0)
        doc = _report_json(out)
        assert {f["fingerprint"] for f in doc["findings"]} == {
            f.fingerprint for f in built.findings
        }


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


class TestFailsClosed:
    def test_missing_expected_check_exits_2_and_is_marked_missing(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        code, out = _run(tmp_path, res, expect="smoke,security")
        assert code == 2
        doc = _report_json(out)
        assert doc["missing_checks"] == ["security"]
        assert {"check": "security", "status": "missing"}.items() <= next(
            row for row in doc["checks"] if row["check"] == "security"
        ).items()
        assert "missing" in _md(out)
        assert (out / "issue-candidates.md").is_file()

    def test_the_default_expectation_is_the_weekly_checks(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        out = tmp_path / "out"
        code = audit_main.main(["report", "--results", str(res), "--out", str(out)])
        assert code == 2
        assert _report_json(out)["missing_checks"] == list(
            results.EXPECTED_WEEKLY_CHECKS
        )

    def test_empty_results_directory_exits_2(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        code, out = _run(tmp_path, res, expect="smoke")
        assert code == 2
        assert (out / "audit-report.json").is_file()
        doc = _report_json(out)
        assert doc["verdict"]["exit_code"] == 2
        # Named as its own problem, not only implied by the missing check.
        (problem,) = doc["problems"]
        assert problem["source"] == str(res)
        assert "no result JSON, JUnit XML or pip list" in problem["problem"]

    def test_missing_results_directory_exits_2(self, tmp_path):
        code, out = _run(tmp_path, tmp_path / "nope", expect="smoke")
        assert code == 2
        assert "does not exist" in _md(out)

    def test_malformed_json_exits_2_naming_the_file(self, tmp_path, capsys):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        broken = res / "upstream" / "upstream.json"
        broken.parent.mkdir(parents=True)
        broken.write_text('{"schema": "pmm-audit-result/1", ', encoding="utf-8")
        code, out = _run(tmp_path, res, expect="smoke")
        assert code == 2
        problems = _report_json(out)["problems"]
        assert any("upstream/upstream.json" in p["source"] for p in problems)
        assert "upstream/upstream.json" in capsys.readouterr().err

    def test_invalid_result_document_exits_2(self, tmp_path):
        res = tmp_path / "results"
        doc = _result("smoke", "checkout").to_dict()
        doc["status"] = "green"
        path = res / "smoke.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(doc), encoding="utf-8")
        code, out = _run(tmp_path, res, expect="smoke")
        assert code == 2
        assert "smoke.json" in _report_json(out)["problems"][0]["source"]

    def test_result_whose_facts_are_not_an_object_exits_2(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        doc = _result("package-smoke", "wheel-ubuntu").to_dict()
        doc["facts"] = ["not", "an", "object"]
        (res / "package-smoke.json").write_text(json.dumps(doc), encoding="utf-8")
        code, out = _run(tmp_path, res, expect="package-smoke")
        assert code == 2
        (problem,) = _report_json(out)["problems"]
        assert "facts must be an object" in problem["problem"]

    def test_junit_with_zero_test_cases_exits_2(self, tmp_path):
        res = tmp_path / "results"
        _leg(res, "suite-floors-py3.11", _junit())
        code, out = _run(tmp_path, res, expect="suite")
        assert code == 2
        assert "no test cases" in _md(out)

    def test_malformed_pip_list_exits_2(self, tmp_path):
        res = tmp_path / "results"
        _leg(res, "suite-floors-py3.11", PASSING_JUNIT, "[]")
        code, _ = _run(tmp_path, res, expect="suite")
        assert code == 2

    def test_leg_with_versions_but_no_junit_is_a_failed_leg(self, tmp_path):
        """A suite job records its installed versions even when the install
        or the test run failed, so a pip list without JUnit beside it is a leg
        that did not finish: the report must show it and fail closed."""
        res = tmp_path / "results"
        _leg(res, "suite-newest-py3.12")
        crashed = res / "suite-newest-py3.14"
        crashed.mkdir(parents=True)
        (crashed / "pip-list.json").write_text(
            _pip_list(numpy="2.5.4", mcp="2.3.0"), encoding="utf-8"
        )
        code, out = _run(tmp_path, res, expect="suite")
        assert code == 2
        doc = _report_json(out)
        assert any(
            "suite-newest-py3.14" in p["source"] and "no JUnit" in p["problem"]
            for p in doc["problems"]
        )
        assert "suite-newest-py3.14/pip-list.json" not in doc["ignored_inputs"]
        assert "suite-newest-py3.14" in _md(out)

    def test_leg_without_junit_makes_its_upstream_release_breaking(self):
        """The versions of a leg that did not finish still count: a release it
        installed is not shown harmless on the strength of the other legs."""
        inputs = report.Inputs()
        legs = [
            report.Leg("suite", "suite-newest-py3.12", "pass", {"numpy": "2.5.4"}),
            report.Leg("suite", "suite-newest-py3.14", "error", {"numpy": "2.5.4"}),
        ]
        inputs.legs.extend(legs)
        resolved = report.resolve_upstream(_test_gated(), inputs.legs)
        assert resolved.classification == "breaking-candidate"
        assert resolved.evidence["failing_legs"] == ["suite/suite-newest-py3.14"]

    def test_collected_orphan_pip_list_becomes_an_error_leg(self, tmp_path):
        res = tmp_path / "results"
        crashed = res / "suite-floors-py3.11"
        crashed.mkdir(parents=True)
        (crashed / "pip-list.json").write_text(
            _pip_list(numpy="2.3.3"), encoding="utf-8"
        )
        inputs = report.collect_inputs(res)
        assert [(leg.variant, leg.status) for leg in inputs.legs] == [
            ("suite-floors-py3.11", "error")
        ]
        assert inputs.legs[0].installed == {"numpy": "2.3.3"}
        assert [(run.check, run.variant, run.status) for run in inputs.runs] == [
            ("suite", "suite-floors-py3.11", "error")
        ]
        assert inputs.ignored == []
        assert len(inputs.problems) == 1

    def test_known_issues_at_the_limit_exit_2(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        known = [
            _issue(n, "0" * 12, "CLOSED", "COMPLETED")
            for n in range(results.KNOWN_ISSUES_LIMIT)
        ]
        code, out = _run(tmp_path, res, expect="smoke", known=known)
        assert code == 2
        assert "truncated" in _md(out)

    def test_known_issues_below_the_limit_are_accepted(self):
        text = json.dumps([_issue(1, "0" * 12)])
        assert len(report.parse_known_issues(text, limit=2)) == 1
        with pytest.raises(ValueError, match="limit"):
            report.parse_known_issues(
                json.dumps([_issue(1, "a"), _issue(2, "b")]), limit=2
            )

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "{}",
            '[{"number": "1", "state": "OPEN"}]',
            '[{"number": 1, "state": "MERGED"}]',
        ],
    )
    def test_malformed_known_issues_are_refused(self, text):
        with pytest.raises(ValueError):
            report.parse_known_issues(text)

    def test_missing_known_issues_file_exits_2(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        out = tmp_path / "out"
        code = audit_main.main(
            [
                "report",
                "--results",
                str(res),
                "--out",
                str(out),
                "--expect",
                "smoke",
                "--known-issues",
                str(tmp_path / "absent.json"),
            ]
        )
        assert code == 2
        assert (out / "audit-report.md").is_file()

    def test_unknown_expected_check_is_refused(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout"))
        code, _ = _run(tmp_path, res, expect="smoke,nonsense")
        assert code == 2


# ---------------------------------------------------------------------------
# Local-processing guard and untrusted text
# ---------------------------------------------------------------------------


class TestEvidenceHygiene:
    def test_long_excerpt_is_truncated_with_a_marker(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        finding = _finding(excerpt="x" * (results.EXCERPT_LIMIT * 3))
        doc = _result("smoke", "checkout", [finding]).to_dict()
        # Written without the canonical writer, so the excerpt is still long.
        (res / "smoke.json").write_text(json.dumps(doc), encoding="utf-8")
        code, out = _run(tmp_path, res, expect="smoke")
        (item,) = _report_json(out)["findings"]
        assert len(item["evidence"]["excerpt"]) < results.EXCERPT_LIMIT + 100
        assert "truncated" in item["evidence"]["excerpt"]
        assert "truncated" in _md(out)
        assert "x" * (results.EXCERPT_LIMIT + 1) not in _md(out, "issue-candidates.md")
        assert code == 0

    def test_result_with_a_waveform_sized_array_is_rejected(self, tmp_path):
        res = tmp_path / "results"
        res.mkdir()
        doc = _result("smoke", "checkout", [_finding()]).to_dict()
        doc["facts"]["samples"] = [0.1] * 1000
        (res / "smoke.json").write_text(json.dumps(doc), encoding="utf-8")
        code, out = _run(tmp_path, res, expect="smoke")
        assert code == 2
        (problem,) = _report_json(out)["problems"]
        assert "smoke.json" in problem["source"]
        assert "numeric array" in problem["problem"]
        assert "0.1, 0.1" not in (out / "audit-report.json").read_text(encoding="utf-8")

    def test_untrusted_excerpt_cannot_break_out_of_its_fence(self, tmp_path):
        res = tmp_path / "results"
        hostile = "line\n```\n## Candidate 99\n````\n~~~~~\n# injected heading\n~~~"
        finding = _finding(excerpt=hostile)
        _write(res, "smoke.json", _result("smoke", "checkout", [finding]))
        _, out = _run(tmp_path, res, expect="smoke")
        for text in (_md(out), _md(out, "issue-candidates.md")):
            outside = "\n".join(_outside_fences(text))
            assert "injected heading" not in outside
            assert "Candidate 99" not in outside
        # The pasted issue body (the inside of the outer fence) holds too.
        (block,) = _blocks(_md(out, "issue-candidates.md"))
        body = _fenced_bodies(block)[-1]
        assert "injected heading" not in "\n".join(_outside_fences(body))
        assert hostile.splitlines()[-2] in body

    def test_title_markup_only_appears_inside_code_spans(self, tmp_path):
        res = tmp_path / "results"
        nasty = _finding(
            title="<img src=x onerror=alert(1)> | [link](http://x) `tick` @someone"
        )
        _write(res, "smoke.json", _result("smoke", "checkout", [nasty]))
        _, out = _run(tmp_path, res, expect="smoke")
        for text in (_md(out), _md(out, "issue-candidates.md")):
            for line in _outside_fences(text):
                bare = re.sub(r"(`+)(?!`).*?(?<!`)\1(?!`)", "", line)
                assert "onerror" not in bare, line
                assert "](http://x)" not in bare, line


# ---------------------------------------------------------------------------
# Markdown, JSON and CLI wiring
# ---------------------------------------------------------------------------


class TestOutputs:
    def build(self, tmp_path):
        res = tmp_path / "results"
        _write(res, "smoke/smoke.json", _result("smoke", "checkout", [_finding()]))
        _leg(res, "suite-floors-py3.11", _junit_sample())
        return res

    def test_summary_has_the_per_check_table_and_candidate_count(self, tmp_path):
        res = self.build(tmp_path)
        summary = tmp_path / "step-summary.md"
        summary.write_text("previous step output\n", encoding="utf-8")
        code, out = _run(
            tmp_path,
            res,
            expect="smoke,suite",
            extra=["--github-summary", str(summary)],
        )
        md = _md(out)
        assert re.search(r"(?m)^\| Check \| Variant \| Status \|", md)
        assert re.search(r"(?m)^\| suite \| suite-floors-py3\.11 \| fail \|", md)
        assert re.search(r"(?m)^\| smoke \| checkout \| pass \|", md)
        doc = _report_json(out)
        assert f"Issue candidates: {len(doc['candidates'])}" in md
        assert len(doc["candidates"]) == 6

        appended = summary.read_text(encoding="utf-8")
        assert appended.startswith("previous step output\n")
        assert f"Issue candidates: {len(doc['candidates'])}" in appended
        assert "| Check | Variant | Status |" in appended
        # The summary is not the candidate bodies.
        assert "### Reproduction" not in appended
        assert code == 0

    def test_report_json_carries_run_metadata_from_the_environment(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
        monkeypatch.setenv("GITHUB_RUN_ID", "123")
        monkeypatch.setenv("GITHUB_SHA", "abc123")
        res = self.build(tmp_path)
        _, out = _run(tmp_path, res, expect="smoke,suite")
        doc = _report_json(out)
        assert doc["schema"] == results.REPORT_SCHEMA
        run = doc["run"]
        assert run["run_url"] == "https://github.com/owner/repo/actions/runs/123"
        assert run["sha"] == "abc123"
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", run["generated_at"])
        assert "https://github.com/owner/repo/actions/runs/123" in _md(
            out, "issue-candidates.md"
        )

    def test_local_run_has_no_github_metadata(self, tmp_path, monkeypatch):
        for name in (
            "GITHUB_SERVER_URL",
            "GITHUB_REPOSITORY",
            "GITHUB_RUN_ID",
            "GITHUB_SHA",
        ):
            monkeypatch.delenv(name, raising=False)
        res = self.build(tmp_path)
        _, out = _run(tmp_path, res, expect="smoke,suite")
        assert "run_url" not in _report_json(out)["run"]
        assert "local run" in _md(out, "issue-candidates.md")

    def test_outputs_are_deterministic(self, tmp_path):
        res = self.build(tmp_path)
        _, out = _run(tmp_path, res, expect="smoke,suite")
        first = _md(out, "issue-candidates.md")
        _, out = _run(tmp_path, res, expect="smoke,suite")
        assert _md(out, "issue-candidates.md") == first

    def test_suite_reproduction_names_the_test(self, tmp_path):
        res = self.build(tmp_path)
        _, out = _run(tmp_path, res, expect="smoke,suite")
        (block,) = [b for b in _blocks(_md(out, "issue-candidates.md")) if FP_DOCS in b]
        assert (
            'python -m pytest "tests/test_install_docs.py::'
            'test_extras_named_in_docs_exist"' in block
        )

    def test_cli_help_lists_the_report_flags(self, capsys):
        with pytest.raises(SystemExit) as exc:
            audit_main.main(["report", "--help"])
        assert exc.value.code == 0
        text = capsys.readouterr().out
        for flag in (
            "--results",
            "--expect",
            "--known-issues",
            "--out",
            "--github-summary",
        ):
            assert flag in text


# ---------------------------------------------------------------------------
# Issue template parity
# ---------------------------------------------------------------------------


def _template():
    return yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))


def _rendered_block_labels(tmp_path):
    res = tmp_path / "results"
    _write(res, "smoke.json", _result("smoke", "checkout", [_finding()]))
    _, out = _run(tmp_path, res, expect="smoke")
    (block,) = _blocks(_md(out, "issue-candidates.md"))
    return _headings(block)


class TestTemplateParity:
    def test_template_parses_and_declares_the_audit_title_and_label(self):
        template = _template()
        assert template["name"] == "Audit finding"
        assert template["title"] == "[Audit] "
        assert template["labels"] == ["audit"]

    def test_the_real_template_matches_the_candidate_blocks(self, tmp_path):
        labels = _rendered_block_labels(tmp_path)
        assert report.template_parity(_template(), labels) == []
        assert labels == CANDIDATE_LABELS

    def test_dropdowns_equal_the_vocabularies(self):
        fields = {item.get("id"): item for item in _template()["body"]}
        assert fields["category"]["attributes"]["options"] == list(results.CATEGORIES)
        assert fields["severity"]["attributes"]["options"] == list(results.SEVERITIES)

    def test_a_missing_fingerprint_field_is_reported(self):
        template = _template()
        template["body"] = [i for i in template["body"] if i.get("id") != "fingerprint"]
        problems = report.template_parity(template, CANDIDATE_LABELS)
        assert any("fingerprint" in p for p in problems), problems

    def test_a_fingerprint_field_that_is_optional_is_reported(self):
        template = _template()
        for item in template["body"]:
            if item.get("id") == "fingerprint":
                item["validations"] = {"required": False}
        problems = report.template_parity(template, CANDIDATE_LABELS)
        assert any("required" in p for p in problems)

    @pytest.mark.parametrize("field_id", ["category", "severity"])
    def test_dropdown_drift_is_reported_both_ways(self, field_id):
        vocabulary = (
            results.CATEGORIES if field_id == "category" else results.SEVERITIES
        )
        for options in (list(vocabulary)[1:], list(vocabulary) + ["extra"]):
            template = _template()
            for item in template["body"]:
                if item.get("id") == field_id:
                    item["attributes"]["options"] = options
            problems = report.template_parity(template, CANDIDATE_LABELS)
            assert any(field_id in p for p in problems), (options, problems)

    def test_fields_must_map_both_ways(self):
        extra_block_field = report.template_parity(
            _template(), CANDIDATE_LABELS + ["Notes"]
        )
        assert any("Notes" in p for p in extra_block_field)
        missing_block_field = report.template_parity(
            _template(),
            [label for label in CANDIDATE_LABELS if label != "Reproduction"],
        )
        assert any("Reproduction" in p for p in missing_block_field)

    def test_template_has_no_em_dashes(self):
        assert "\u2014" not in TEMPLATE.read_text(encoding="utf-8")
