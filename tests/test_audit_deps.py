"""Guards for the dependency helpers and the vulnerability-audit normalizer.

Three subcommands feed the weekly audit's dependency legs:

* ``floors`` turns the ``>=`` bounds of ``[project.dependencies]`` into a
  constraints file, so one suite leg runs on exactly the versions the project
  claims to support. A dependency without a floor is an error, never skipped:
  a skipped entry would let that leg resolve the newest release and still be
  reported as "floors".
* ``lock-reqs`` turns ``uv.lock`` into pinned requirements for pip-audit.
  uv.lock is a universal lock, so one name can appear once per Python range;
  each pin carries its fork's markers so exactly one applies on any
  interpreter (pip-audit refuses duplicate names after marker evaluation).
* ``security`` normalizes two pip-audit JSON files into one
  ``pmm-audit-result/1`` result and fails closed: a missing, unparseable or
  truncated audit is an error, never a quiet pass.

Everything here is offline: fixtures live in ``tests/fixtures/audit``.
"""

import json
import shutil
import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

from tools.audit import __main__ as audit_main
from tools.audit import deps, results, security

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audit"
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCK_SAMPLE = FIXTURES / "uv_lock_sample.toml"
AUDIT_LOCK = FIXTURES / "pip_audit_lock.json"
AUDIT_FRESH = FIXTURES / "pip_audit_fresh.json"

INTERPRETERS = ("3.11.9", "3.12.4", "3.13.1", "3.14.0", "3.15.0")


def _write_pyproject(tmp_path, dependencies):
    body = "\n".join(f'    "{dep}",' for dep in dependencies)
    path = tmp_path / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "demo"\nversion = "0"\ndependencies = [\n{body}\n]\n',
        encoding="utf-8",
    )
    return path


def _env(full_version):
    major, minor = full_version.split(".")[:2]
    return {"python_full_version": full_version, "python_version": f"{major}.{minor}"}


# --------------------------------------------------------------------------
# floors
# --------------------------------------------------------------------------


class TestFloors:
    def test_real_pyproject_pins_every_declared_floor(self):
        requirements = deps.read_declared(PYPROJECT)
        pins = deps.floor_pins(requirements)

        assert "mcp==2.0.0" in pins
        assert "numpy==2.3.3" in pins
        assert "scipy==1.16.2" in pins
        assert "pydantic==2.12.0" in pins
        assert "pypdf==4.0" in pins
        assert len(pins) == len(requirements)
        assert not any("[" in pin or ";" in pin for pin in pins), pins

    def test_every_pin_parses_as_an_exact_requirement(self):
        for pin in deps.floor_pins(deps.read_declared(PYPROJECT)):
            req = Requirement(pin)
            (spec,) = req.specifier
            assert spec.operator == "=="

    def test_extras_and_markers_are_stripped(self):
        reqs = [Requirement('mcp[cli]>=2.0.0,<3.0.0; python_version >= "3.10"')]
        assert deps.floor_pins(reqs) == ["mcp==2.0.0"]

    def test_dependency_without_a_floor_is_an_error_naming_it(self):
        reqs = [
            Requirement("numpy>=2.3.3"),
            Requirement("unbounded"),
            Requirement("pinned==1.0"),
            Requirement("compatible~=1.4"),
        ]
        with pytest.raises(ValueError) as excinfo:
            deps.floor_pins(reqs)
        message = str(excinfo.value)
        for name in ("unbounded", "pinned", "compatible"):
            assert name in message
        assert "numpy" not in message

    def test_cli_writes_constraints_file(self, tmp_path, capsys):
        out = tmp_path / "constraints.txt"
        code = audit_main.main(
            ["floors", "--pyproject", str(PYPROJECT), "--output", str(out)]
        )
        assert code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "mcp==2.0.0" in lines and "numpy==2.3.3" in lines

    def test_cli_prints_to_stdout_without_output(self, capsys):
        code = audit_main.main(["floors", "--pyproject", str(PYPROJECT)])
        assert code == 0
        assert "mcp==2.0.0" in capsys.readouterr().out.splitlines()

    def test_cli_refuses_a_dependency_without_floor(self, tmp_path, capsys):
        path = _write_pyproject(tmp_path, ["numpy>=2.3.3", "no-floor-pkg<5"])
        code = audit_main.main(["floors", "--pyproject", str(path)])
        assert code != 0
        assert "no-floor-pkg" in capsys.readouterr().err


# --------------------------------------------------------------------------
# lock-reqs
# --------------------------------------------------------------------------


class TestLockReqs:
    def _pins(self):
        return deps.lock_pins(tomllib.loads(LOCK_SAMPLE.read_text(encoding="utf-8")))

    def test_excludes_project_and_non_registry_sources(self):
        names = [Requirement(pin).name for pin in self._pins()]
        assert "predictive-maintenance-mcp" not in names
        for excluded in ("local-helper", "path-dep", "git-dep", "workspace-root"):
            assert excluded not in names
        assert "alpha" in names and "beta-pkg" in names

    def test_unforked_package_is_a_plain_pin(self):
        assert "alpha==1.0.0" in self._pins()
        assert "beta-pkg==2.5.0" in self._pins()

    def test_forked_package_gives_one_applicable_pin_per_interpreter(self):
        sphinx = [Requirement(p) for p in self._pins() if p.startswith("sphinx==")]
        assert len(sphinx) == 2
        assert all(req.marker is not None for req in sphinx)
        for full_version in INTERPRETERS:
            applicable = [
                str(req.specifier)
                for req in sphinx
                if req.marker.evaluate(_env(full_version))
            ]
            assert len(applicable) == 1, (full_version, applicable)
        on_311 = [r for r in sphinx if r.marker.evaluate(_env("3.11.9"))]
        assert str(on_311[0].specifier) == "==9.0.4"

    def test_marker_text_is_valid_pep508(self):
        for pin in self._pins():
            if ";" in pin:
                Marker(pin.split(";", 1)[1])

    def test_lock_without_registry_packages_is_an_error(self):
        lock = {
            "version": 1,
            "package": [
                {
                    "name": "predictive-maintenance-mcp",
                    "version": "0.13.0",
                    "source": {"editable": "."},
                }
            ],
        }
        with pytest.raises(ValueError):
            deps.lock_pins(lock)
        with pytest.raises(ValueError):
            deps.lock_pins({"version": 1})

    def test_real_lock_yields_registry_pins(self):
        lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
        pins = deps.lock_pins(lock)
        names = [Requirement(p).name for p in pins]
        assert "predictive-maintenance-mcp" not in names
        assert "mcp" in names and "numpy" in names
        for full_version in ("3.11.9", "3.12.4", "3.14.0"):
            applicable = [
                Requirement(p).name
                for p in pins
                if Requirement(p).marker is None
                or Requirement(p).marker.evaluate(_env(full_version))
            ]
            assert len(applicable) == len(set(applicable)), full_version

    def test_cli_writes_requirements(self, tmp_path):
        out = tmp_path / "lock-reqs.txt"
        code = audit_main.main(
            ["lock-reqs", "--lock", str(LOCK_SAMPLE), "--output", str(out)]
        )
        assert code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "alpha==1.0.0" in lines

    def test_cli_empty_lock_exits_non_zero(self, tmp_path, capsys):
        empty = tmp_path / "uv.lock"
        empty.write_text("version = 1\n", encoding="utf-8")
        code = audit_main.main(["lock-reqs", "--lock", str(empty)])
        assert code != 0
        assert "error" in capsys.readouterr().err


# --------------------------------------------------------------------------
# security
# --------------------------------------------------------------------------


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

    def test_pure_normalizer_matches_cli(self, tmp_path):
        requirements = deps.read_declared(PYPROJECT)
        result = security.build_result(
            AUDIT_LOCK, AUDIT_FRESH, requirements, variant="unit"
        )
        assert result.status == "pass"
        assert result.check == "security"
        assert result.facts["advisories"] == 4
        assert result.facts["fixable_in_range"] == 2

    def test_fixable_in_range_helper(self):
        from packaging.specifiers import SpecifierSet

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
        pyproject.write_text(
            text.replace('"scipy>=1.16.2",', '"scipy>=1.16.2",\n    "newdep>=1.0",'),
            encoding="utf-8",
        )
        code, doc = _security(tmp_path, pyproject=pyproject)
        assert code == 2
        coverage = [f for f in doc["findings"] if f["rule"] == "audit-coverage"]
        assert {tuple(f["evidence"]["missing"]) for f in coverage} == {("newdep",)}
