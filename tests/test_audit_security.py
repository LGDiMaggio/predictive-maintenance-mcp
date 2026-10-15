"""Guards for the vulnerability-audit normalizer (``python -m tools.audit security``).

The weekly security job runs pip-audit twice, over the ``uv.lock`` pins and
over a fresh resolution, and ``security`` normalizes the two JSON files into
one ``pmm-audit-result/1`` result. pip-audit's exit code is not trusted (a
tool failure and a found vulnerability share non-zero codes), so the
normalizer reads the JSON and fails closed: a missing, unparseable or
truncated audit is an error, never a quiet pass.

These tests pin one finding per (package, advisory) across both audits, the
``fixable_in_range`` verdict against the declared range read live from
``pyproject.toml``, and every fail-closed path. Everything here is offline:
the pip-audit outputs are fixtures in ``tests/fixtures/audit``.
"""

import json
import shutil
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from tools.audit import __main__ as audit_main
from tools.audit import deps, results, security

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audit"
PYPROJECT = REPO_ROOT / "pyproject.toml"
AUDIT_LOCK = FIXTURES / "pip_audit_lock.json"
AUDIT_FRESH = FIXTURES / "pip_audit_fresh.json"


def _security(tmp_path, lock=AUDIT_LOCK, fresh=AUDIT_FRESH, pyproject=PYPROJECT):
    out = tmp_path / "security.json"
    code = audit_main.main(
        [
            "security",
            "--lock",
            str(lock),
            "--fresh",
            str(fresh),
            "--pyproject",
            str(pyproject),
            "--json-out",
            str(out),
        ]
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    results.validate_result(doc)
    return code, doc


def _by_subject(doc):
    return {item["subject"]: item for item in doc["findings"]}


class TestSecurity:
    def test_fixable_in_range_advisory_is_breaking_candidate(self, tmp_path):
        code, doc = _security(tmp_path)
        assert code == 0 and doc["status"] == "pass"
        assert doc["check"] == "security"
        finding = _by_subject(doc)["numpy GHSA-0000-test-nmpy"]
        assert finding["category"] == "security"
        assert finding["classification"] == "breaking-candidate"
        assert finding["severity"] == "high"
        assert finding["evidence"]["fixable_in_range"] is True
        assert finding["evidence"]["fix_versions"] == ["2.3.4"]
        assert finding["fingerprint"] == results.fingerprint(
            "security", finding["rule"], "numpy GHSA-0000-test-nmpy"
        )

    def test_fix_only_above_the_cap_is_not_fixable_in_range(self, tmp_path):
        _, doc = _security(tmp_path)
        finding = _by_subject(doc)["mcp GHSA-0000-test-mcp3"]
        assert finding["evidence"]["fixable_in_range"] is False
        assert finding["classification"] == "review"

    def test_no_fix_is_not_fixable_in_range(self, tmp_path):
        _, doc = _security(tmp_path)
        finding = _by_subject(doc)["pandas PYSEC-2099-1"]
        assert finding["evidence"]["fixable_in_range"] is False
        assert finding["classification"] == "review"
        assert finding["evidence"]["aliases"] == ["GHSA-0000-test-pnds"]

    def test_transitive_fix_counts_as_in_range(self, tmp_path):
        _, doc = _security(tmp_path)
        finding = _by_subject(doc)["urllib3 GHSA-0000-test-url3"]
        assert finding["evidence"]["fixable_in_range"] is True
        assert finding["evidence"]["declared"] is None
        assert finding["classification"] == "breaking-candidate"

    def test_advisory_seen_by_both_audits_is_one_finding(self, tmp_path):
        _, doc = _security(tmp_path)
        subjects = [item["subject"] for item in doc["findings"]]
        assert subjects.count("urllib3 GHSA-0000-test-url3") == 1
        evidence = _by_subject(doc)["urllib3 GHSA-0000-test-url3"]["evidence"]
        assert evidence["seen_in"] == ["fresh", "lock"]
        assert evidence["versions"] == {"fresh": "2.5.0", "lock": "2.5.0"}
        assert evidence["aliases"] == ["CVE-2099-0002", "PYSEC-2099-2"]

    def test_skipped_dependency_is_a_coverage_finding_except_the_project(
        self, tmp_path
    ):
        _, doc = _security(tmp_path)
        skipped = [f for f in doc["findings"] if f["rule"] == "audit-skipped"]
        assert [f["subject"] for f in skipped] == ["odd-transitive"]
        assert skipped[0]["classification"] == "review"

    def test_long_description_is_truncated(self, tmp_path):
        data = json.loads(AUDIT_LOCK.read_text(encoding="utf-8"))
        data["dependencies"][1]["vulns"][0]["description"] = "x" * 20000
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps(data), encoding="utf-8")
        _, doc = _security(tmp_path, lock=lock)
        excerpt = _by_subject(doc)["numpy GHSA-0000-test-nmpy"]["evidence"]["excerpt"]
        assert len(excerpt) < 20000
        assert "truncated" in excerpt

    def test_missing_declared_name_fails_closed(self, tmp_path):
        data = json.loads(AUDIT_LOCK.read_text(encoding="utf-8"))
        data["dependencies"] = [d for d in data["dependencies"] if d["name"] != "scipy"]
        lock = tmp_path / "lock.json"
        lock.write_text(json.dumps(data), encoding="utf-8")
        code, doc = _security(tmp_path, lock=lock)
        assert code == 2 and doc["status"] == "error"
        coverage = [f for f in doc["findings"] if f["rule"] == "audit-coverage"]
        assert coverage and coverage[0]["evidence"]["missing"] == ["scipy"]

    def test_names_are_compared_after_pep503_normalization(self, tmp_path):
        # The fresh fixture spells scikit-learn as "Scikit_Learn".
        code, doc = _security(tmp_path)
        assert code == 0
        assert not [f for f in doc["findings"] if f["rule"] == "audit-coverage"]

    def test_missing_file_fails_closed(self, tmp_path):
        code, doc = _security(tmp_path, fresh=tmp_path / "absent.json")
        assert code == 2 and doc["status"] == "error"
        invalid = [f for f in doc["findings"] if f["rule"] == "audit-input-invalid"]
        assert [f["subject"] for f in invalid] == ["pip-audit fresh"]
        assert invalid[0]["classification"] == "not-checked"

    @pytest.mark.parametrize(
        "content",
        ["{not json", "[]", '{"dependencies": "nope"}', '{"dependencies": [{}]}'],
    )
    def test_unparseable_file_fails_closed(self, tmp_path, content):
        bad = tmp_path / "bad.json"
        bad.write_text(content, encoding="utf-8")
        code, doc = _security(tmp_path, lock=bad)
        assert code == 2 and doc["status"] == "error"
        assert any(f["rule"] == "audit-input-invalid" for f in doc["findings"])

    def test_pure_normalizer_matches_cli(self):
        requirements = deps.read_declared(PYPROJECT)
        result = security.build_result(
            AUDIT_LOCK, AUDIT_FRESH, requirements, variant="unit"
        )
        assert result.status == "pass"
        assert result.check == "security"
        assert result.facts["advisories"] == 4
        assert result.facts["fixable_in_range"] == 2

    def test_fixable_in_range_helper(self):
        cap = SpecifierSet(">=2.0.0,<3.0.0")
        assert security.fixable_in_range(["2.3.1"], cap) is True
        assert security.fixable_in_range(["3.0.0"], cap) is False
        assert security.fixable_in_range([], cap) is False
        assert security.fixable_in_range(["not-a-version"], cap) is False
        assert security.fixable_in_range(["0.1"], None) is True

    def test_pyproject_copy_is_read_live(self, tmp_path):
        # A declared dependency the audits never saw must fail the check:
        # the declared names come from pyproject.toml, not a frozen list.
        pyproject = tmp_path / "pyproject.toml"
        shutil.copy(PYPROJECT, pyproject)
        text = pyproject.read_text(encoding="utf-8")
        (scipy,) = [
            entry
            for entry in tomllib.loads(text)["project"]["dependencies"]
            if Requirement(entry).name == "scipy"
        ]
        line = f'"{scipy}",'
        pyproject.write_text(
            text.replace(line, f'{line}\n    "newdep>=1.0",'), encoding="utf-8"
        )
        code, doc = _security(tmp_path, pyproject=pyproject)
        assert code == 2
        coverage = [f for f in doc["findings"] if f["rule"] == "audit-coverage"]
        assert {tuple(f["evidence"]["missing"]) for f in coverage} == {("newdep",)}
