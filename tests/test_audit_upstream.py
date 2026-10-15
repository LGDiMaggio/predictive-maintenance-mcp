"""Guards for the compatibility inventory and the upstream checker.

``python -m tools.audit upstream`` compares ``tools/audit/compat_inventory.toml``
(what a maintainer has reviewed) with official release data (PyPI JSON,
GitHub releases, the installed SDK's protocol constants) and classifies each
difference with the table in the ``tools/audit/upstream.py`` docstring:
harmless, test-gated, review, breaking-candidate or blocked-by-policy.

The checker takes an injected fetcher, so nothing here touches the network.
PyPI and GitHub payloads in ``tests/fixtures/audit`` are real responses
recorded on 2026-10-15 and trimmed to the fields the checker reads; the
scenarios add synthetic future releases on top of them.

Two properties matter more than any single classification:

* an upstream source that cannot be read is a ``not-checked`` finding and an
  ``error`` status, never a silent pass, and it never stops the other
  components from being classified;
* the inventory and ``[project.dependencies]`` agree both ways, so a new
  runtime dependency cannot arrive unwatched.
"""

import copy
import json
import tomllib
from datetime import date
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from tools.audit import __main__ as audit_main
from tools.audit import deps, results, upstream

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audit"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TODAY = date(2026, 10, 15)

MCP_SPEC = SpecifierSet(">=2.0.0,<3.0.0")
REVIEWED_REVISIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
)


def _recorded(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _pypi(name, releases):
    """Synthetic PyPI JSON: ``{version: requires_python | (rp, yanked)}``."""
    out = {}
    for version, spec in releases.items():
        yanked = False
        if isinstance(spec, tuple):
            spec, yanked = spec
        out[version] = [{"requires_python": spec, "yanked": yanked}]
    return {"info": {"name": name}, "releases": out}


def _add(payload, releases):
    payload = copy.deepcopy(payload)
    payload["releases"].update(_pypi("x", releases)["releases"])
    return payload


def _component(**fields):
    base = {
        "id": "numpy",
        "kind": "pypi",
        "source": "https://pypi.org/project/numpy/",
        "package": "numpy",
        "rules": "library",
        "role": "runtime",
        "reviewed": "2.3.3",
        "reviewed_on": TODAY,
        "note": "test component",
    }
    base.update(fields)
    for key in [k for k, v in base.items() if v is None]:
        del base[key]
    doc = {
        "schema": upstream.INVENTORY_SCHEMA,
        "defaults": {"interpreters": ["3.11", "3.12", "3.14"]},
        "component": [base],
    }
    (component,) = upstream.parse_inventory(doc)
    return component


def _mcp_component():
    return _component(
        id="mcp",
        package="mcp",
        rules="mcp-sdk",
        reviewed="2.3.0",
        category="mcp-compat",
        source="https://pypi.org/project/mcp/",
    )


def _non_pypi(kind, **fields):
    return _component(kind=kind, package=None, rules=None, role=None, **fields)


def _release(tag, published, prerelease=False, draft=False):
    return {
        "tag_name": tag,
        "name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "published_at": published,
        "created_at": published,
    }


class FakeFetcher:
    """URL -> payload; an Exception value is raised instead of returned."""

    def __init__(self, responses):
        self.responses = dict(responses)
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url not in self.responses:
            raise AssertionError(f"unexpected fetch {url}")
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def _steady_responses(components):
    """Upstream that has published nothing newer than the inventory."""
    responses = {}
    for component in components:
        if component.kind == "pypi":
            url = upstream.pypi_url(component.package)
            responses[url] = _pypi(component.package, {component.reviewed: None})
        elif component.kind == "github-release":
            url = upstream.github_releases_url(component.repo)
            responses[url] = [_release(component.reviewed, "2026-01-01T00:00:00Z")]
    return responses


def _steady_probe():
    return upstream.ProtocolProbe(
        versions=REVIEWED_REVISIONS,
        source="test",
        sdk_version="2.3.0",
    )


def _real_inventory():
    return upstream.load_inventory(upstream.DEFAULT_INVENTORY)


def _real_requirements():
    return deps.read_declared(PYPROJECT)


def _only(findings, **match):
    hits = [f for f in findings if all(getattr(f, k) == v for k, v in match.items())]
    assert len(hits) == 1, (match, findings)
    return hits[0]


# --------------------------------------------------------------------------
# PyPI classification
# --------------------------------------------------------------------------


class TestMcpClassification:
    def _classify(self, extra=None):
        payload = _recorded("pypi_mcp")
        if extra:
            payload = _add(payload, extra)
        return upstream.classify_pypi(_mcp_component(), payload, MCP_SPEC)

    def test_recorded_snapshot_latest_equals_reviewed(self):
        outcome = self._classify()
        assert outcome.findings == []
        assert outcome.facts["outcome"] == "harmless"
        assert set(outcome.facts["latest"].values()) == {"2.3.0"}
        # 2.0.0rc1 predates 2.3.0: an old pre-release is not news.
        assert outcome.facts["latest_prerelease"] is None

    def test_new_minor_is_review_with_major_minor_in_subject(self):
        (finding,) = self._classify({"2.4.0": ">=3.10"}).findings
        assert finding.classification == "review"
        assert "2.4" in finding.subject
        assert finding.fingerprint == results.fingerprint(
            "mcp-compat", finding.rule, "mcp 2.4"
        )
        (later,) = self._classify({"2.5.0": ">=3.10"}).findings
        assert later.fingerprint != finding.fingerprint

    def test_new_patch_is_test_gated_and_unresolved(self):
        (finding,) = self._classify({"2.3.1": ">=3.10"}).findings
        assert finding.classification == "test-gated"
        assert finding.subject == "mcp"
        assert finding.evidence["candidates"] == ["2.3.1"]
        assert finding.evidence["package"] == "mcp"

    @pytest.mark.parametrize("version", ["3.0.0", "3.0.0a1"])
    def test_next_major_final_or_prerelease_is_blocked_by_policy(self, version):
        (finding,) = self._classify({version: ">=3.10"}).findings
        assert finding.classification == "blocked-by-policy"
        assert finding.subject == "mcp 3.0"

    def test_1x_maintenance_release_is_ignored(self):
        assert self._classify({"1.31.0": ">=3.10"}).findings == []

    def test_yanked_newest_release_is_ignored(self):
        outcome = self._classify({"2.4.0": (">=3.10", True)})
        assert outcome.findings == []
        assert set(outcome.facts["latest"].values()) == {"2.3.0"}

    def test_lockstep_partner_is_named_in_evidence(self):
        component = _component(
            id="mcp",
            package="mcp",
            rules="mcp-sdk",
            reviewed="2.3.0",
            category="mcp-compat",
            lockstep=["mcp-types"],
        )
        payload = _add(_recorded("pypi_mcp"), {"2.3.1": ">=3.10"})
        (finding,) = upstream.classify_pypi(component, payload, MCP_SPEC).findings
        assert finding.evidence["lockstep"] == ["mcp-types"]


class TestLibraryClassification:
    def test_numpy_requires_python_gives_per_interpreter_latest_and_split(self):
        outcome = upstream.classify_pypi(
            _component(), _recorded("pypi_numpy"), SpecifierSet(">=2.3.3")
        )
        # numpy 2.4.0 is really yanked on PyPI: the 3.11 line is 2.4.6.
        assert outcome.facts["latest"] == {
            "3.11": "2.4.6",
            "3.12": "2.5.4",
            "3.14": "2.5.4",
        }
        gated = _only(outcome.findings, classification="test-gated")
        assert gated.subject == "numpy"
        assert gated.evidence["candidates"] == ["2.4.6", "2.5.4"]
        split = _only(outcome.findings, rule="matrix-split")
        assert split.classification == "review"
        assert split.subject == "numpy 2.5"
        assert split.evidence["excluded_interpreters"] == ["3.11"]
        assert len(outcome.findings) == 2

    def test_new_major_reachable_from_floor_is_breaking_candidate(self):
        payload = _add(_recorded("pypi_pandas"), {"4.0.0": ">=3.11"})
        component = _component(id="pandas", package="pandas", reviewed="3.0.6")
        outcome = upstream.classify_pypi(component, payload, SpecifierSet(">=2.3.3"))
        (finding,) = outcome.findings
        assert finding.classification == "breaking-candidate"
        assert finding.subject == "pandas 4.0"

    def test_recorded_pandas_3_is_breaking_candidate_from_2x_review(self):
        component = _component(id="pandas", package="pandas", reviewed="2.3.3")
        outcome = upstream.classify_pypi(
            component, _recorded("pypi_pandas"), SpecifierSet(">=2.3.3")
        )
        finding = _only(outcome.findings, classification="breaking-candidate")
        assert finding.subject == "pandas 3.0"
        # The 3.1.0rc0 pre-release is recorded, never classified.
        assert outcome.facts["latest_prerelease"] == "3.1.0rc0"

    def test_pydantic_minor_is_test_gated(self):
        payload = _pypi("pydantic", {"2.12.0": ">=3.9", "2.13.0": ">=3.9"})
        component = _component(id="pydantic", package="pydantic", reviewed="2.12.0")
        outcome = upstream.classify_pypi(component, payload, SpecifierSet(">=2.12.0"))
        (finding,) = outcome.findings
        assert finding.classification == "test-gated"
        assert finding.rule == "upstream-minor"
        assert finding.subject == "pydantic"

    def test_latest_equal_to_reviewed_is_no_finding(self):
        payload = _pypi("pydantic", {"2.11.0": ">=3.9", "2.12.0": ">=3.9"})
        component = _component(id="pydantic", package="pydantic", reviewed="2.12.0")
        outcome = upstream.classify_pypi(component, payload, SpecifierSet(">=2.12.0"))
        assert outcome.findings == []
        assert outcome.facts["outcome"] == "harmless"

    def test_missing_or_invalid_requires_python_admits_every_interpreter(self):
        payload = _pypi("legacy", {"1.0": None, "1.1": "", "1.2": ">=2.7.*"})
        component = _component(id="legacy", package="legacy", reviewed="1.0")
        outcome = upstream.classify_pypi(component, payload, None)
        assert set(outcome.facts["latest"].values()) == {"1.2"}

    def test_malformed_payload_raises_value_error(self):
        with pytest.raises(ValueError):
            upstream.classify_pypi(_component(), {"releases": "nope"}, None)
        with pytest.raises(ValueError):
            upstream.classify_pypi(_component(), {"info": {}}, None)


class TestBuildToolClassification:
    def _tool(self, **fields):
        base = {
            "id": "setuptools",
            "package": "setuptools",
            "rules": "build-tool",
            "role": "tooling",
            "reviewed": "84.0.0",
        }
        base.update(fields)
        return _component(**base)

    def test_new_minor_is_recorded_harmless_without_finding(self):
        payload = _pypi("setuptools", {"84.0.0": ">=3.10", "84.1.0": ">=3.10"})
        outcome = upstream.classify_pypi(self._tool(), payload, None)
        assert outcome.findings == []
        assert outcome.facts["outcome"] == "harmless"

    def test_new_major_is_test_gated(self):
        payload = _pypi("setuptools", {"84.0.0": ">=3.10", "85.0.0": ">=3.10"})
        (finding,) = upstream.classify_pypi(self._tool(), payload, None).findings
        assert finding.classification == "test-gated"
        assert finding.subject == "setuptools"

    def test_zero_major_minor_counts_as_major(self):
        tool = self._tool(id="uv", package="uv", reviewed="0.13.1")
        patch = _pypi("uv", {"0.13.1": ">=3.8", "0.13.2": ">=3.8"})
        assert upstream.classify_pypi(tool, patch, None).findings == []
        minor = _pypi("uv", {"0.13.1": ">=3.8", "0.14.0": ">=3.8"})
        (finding,) = upstream.classify_pypi(tool, minor, None).findings
        assert finding.classification == "test-gated"

    def test_deprecation_due_within_120_days_is_low_review(self):
        tool = self._tool(
            deprecations=[
                {
                    "id": "license-table",
                    "due": date(2027, 2, 18),
                    "note": "project.license as a TOML table",
                }
            ]
        )
        assert upstream.check_deprecations(tool, date(2026, 10, 15)) == []
        (finding,) = upstream.check_deprecations(tool, date(2026, 10, 21))
        assert finding.classification == "review"
        assert finding.severity == "low"
        assert finding.subject == "setuptools license-table"
        (overdue,) = upstream.check_deprecations(tool, date(2027, 3, 1))
        assert overdue.evidence["days_left"] < 0


# --------------------------------------------------------------------------
# Protocol, GitHub and manual kinds
# --------------------------------------------------------------------------


def _protocol_component():
    return _non_pypi(
        "sdk-protocol",
        id="mcp-protocol",
        reviewed=list(REVIEWED_REVISIONS),
        category="mcp-compat",
        source="https://modelcontextprotocol.io/specification/versioning",
    )


class TestProtocol:
    def test_known_equal_to_reviewed_is_no_finding(self):
        outcome = upstream.classify_protocol(_protocol_component(), _steady_probe())
        assert outcome.findings == []

    def test_new_revision_in_sdk_is_review(self):
        probe = upstream.ProtocolProbe(
            versions=REVIEWED_REVISIONS + ("2027-01-15",),
            source="mcp.types.version.KNOWN_PROTOCOL_VERSIONS",
            sdk_version="2.9.0",
        )
        (finding,) = upstream.classify_protocol(_protocol_component(), probe).findings
        assert finding.classification == "review"
        assert finding.subject == "mcp-protocol 2027-01-15"
        assert finding.evidence["sdk_version"] == "2.9.0"

    def test_missing_constants_are_a_finding_not_a_crash(self):
        probe = upstream.ProtocolProbe(
            versions=None, source="tried x", sdk_version=None
        )
        (finding,) = upstream.classify_protocol(_protocol_component(), probe).findings
        assert finding.classification == "not-checked"
        assert finding.severity == "medium"

    def test_probe_reads_the_installed_sdk(self):
        probe = upstream.sdk_protocol_versions()
        assert probe.versions is not None
        assert "2025-11-25" in probe.versions

    def test_probe_survives_moved_constants(self, monkeypatch):
        monkeypatch.setattr(
            upstream,
            "PROTOCOL_LOCATIONS",
            (("no_such_module_for_audit.version", "KNOWN_PROTOCOL_VERSIONS"),),
        )
        probe = upstream.sdk_protocol_versions()
        assert probe.versions is None
        assert "no_such_module_for_audit" in probe.source


def _spec_component(**fields):
    base = {
        "id": "mcp-spec",
        "repo": "modelcontextprotocol/modelcontextprotocol",
        "reviewed": "2026-07-28",
        "category": "mcp-compat",
        "source": "https://github.com/modelcontextprotocol/modelcontextprotocol",
    }
    base.update(fields)
    return _non_pypi("github-release", **base)


class TestGithubRelease:
    def test_recorded_spec_releases_match_reviewed(self):
        outcome = upstream.classify_github(
            _spec_component(), _recorded("github_spec_releases")
        )
        assert outcome.findings == []
        assert outcome.facts["newest_final"] == "2026-07-28"

    def test_rc_tag_is_low_review(self):
        payload = [
            _release("2027-01-15-RC", "2026-11-01T00:00:00Z", prerelease=True)
        ] + _recorded("github_spec_releases")
        (finding,) = upstream.classify_github(_spec_component(), payload).findings
        assert finding.classification == "review"
        assert finding.severity == "low"
        assert finding.rule == "release-candidate"
        assert finding.subject == "mcp-spec 2027-01-15"

    def test_draft_is_ignored(self):
        payload = [
            _release("2027-01-15", "2026-11-01T00:00:00Z", draft=True)
        ] + _recorded("github_spec_releases")
        assert upstream.classify_github(_spec_component(), payload).findings == []

    def test_newer_final_release_is_review(self):
        payload = [_release("2027-01-15", "2027-01-15T00:00:00Z")] + _recorded(
            "github_spec_releases"
        )
        (finding,) = upstream.classify_github(_spec_component(), payload).findings
        assert finding.classification == "review"
        assert finding.rule == "release-newer"
        assert finding.subject == "mcp-spec 2027-01-15"

    def test_floating_ref_moving_is_harmless(self):
        component = _spec_component(
            id="pypi-publish-action",
            repo="pypa/gh-action-pypi-publish",
            reviewed="v1.14.2",
            floating=True,
            category="dependency",
        )
        payload = [
            _release("v1.14.3", "2026-11-01T00:00:00Z"),
            _release("v1.14.2", "2026-07-29T00:00:00Z"),
        ]
        (finding,) = upstream.classify_github(component, payload).findings
        assert finding.classification == "harmless"
        assert finding.severity == "info"

    def test_malformed_payload_raises_value_error(self):
        with pytest.raises(ValueError):
            upstream.classify_github(_spec_component(), {"message": "Not Found"})


class TestManual:
    def _manual(self, **fields):
        base = {
            "id": "transports",
            "reviewed_on": date(2026, 1, 1),
            "review_interval_days": 90,
            "source": "https://modelcontextprotocol.io/specification",
            "category": "mcp-compat",
        }
        base.update(fields)
        return _non_pypi("manual", **base)

    def test_overdue_review_is_low_review(self):
        (finding,) = upstream.classify_manual(self._manual(), TODAY).findings
        assert finding.classification == "review"
        assert finding.severity == "low"
        assert finding.subject == "transports"

    def test_review_within_interval_is_no_finding(self):
        component = self._manual(reviewed_on=date(2026, 9, 1))
        assert upstream.classify_manual(component, TODAY).findings == []

    def test_without_interval_is_no_finding(self):
        component = self._manual(review_interval_days=None)
        assert upstream.classify_manual(component, TODAY).findings == []


# --------------------------------------------------------------------------
# Whole check: failures, parity, real inventory
# --------------------------------------------------------------------------


class TestCheckUpstream:
    def test_one_fetch_failure_is_not_checked_and_others_still_classified(self):
        components = [_mcp_component(), _component()]
        fetcher = FakeFetcher(
            {
                upstream.pypi_url("mcp"): _add(
                    _recorded("pypi_mcp"), {"2.4.0": ">=3.10"}
                ),
                upstream.pypi_url("numpy"): OSError("connection reset"),
            }
        )
        result = upstream.check_upstream(
            components,
            requirements=[
                Requirement("mcp[cli]>=2.0.0,<3.0.0"),
                Requirement("numpy>=2.3.3"),
            ],
            fetcher=fetcher,
            today=TODAY,
            protocol_probe=_steady_probe,
        )
        assert result.status == "error"
        assert result.exit_code == 2
        missed = _only(result.findings, classification="not-checked")
        assert missed.subject == "numpy"
        assert missed.severity == "medium"
        assert "connection reset" in missed.evidence["error"]
        assert _only(result.findings, classification="review").subject == "mcp 2.4"

    def test_malformed_upstream_payload_is_not_checked(self):
        fetcher = FakeFetcher({upstream.pypi_url("numpy"): {"releases": []}})
        result = upstream.check_upstream(
            [_component()],
            requirements=[Requirement("numpy>=2.3.3")],
            fetcher=fetcher,
            today=TODAY,
            protocol_probe=_steady_probe,
        )
        assert result.status == "error"
        assert _only(result.findings, classification="not-checked").subject == "numpy"

    def test_real_inventory_loads(self):
        components = _real_inventory()
        kinds = {component.kind for component in components}
        assert kinds == set(upstream.KINDS)
        ids = [component.id for component in components]
        assert len(ids) == len(set(ids))
        for component in components:
            if component.kind == "pypi":
                assert "3.11" in component.interpreters

    def test_real_inventory_matches_pyproject_both_ways(self):
        missing, extra = upstream.inventory_parity(
            _real_inventory(), _real_requirements()
        )
        assert missing == [] and extra == []

    def test_inventory_missing_scipy_is_reported(self):
        doc = tomllib.loads(upstream.DEFAULT_INVENTORY.read_text(encoding="utf-8"))
        doc["component"] = [c for c in doc["component"] if c.get("package") != "scipy"]
        components = upstream.parse_inventory(doc)
        missing, extra = upstream.inventory_parity(components, _real_requirements())
        assert missing == ["scipy"] and extra == []

        result = upstream.check_upstream(
            components,
            requirements=_real_requirements(),
            fetcher=FakeFetcher(_steady_responses(components)),
            today=TODAY,
            protocol_probe=_steady_probe,
        )
        assert result.status == "error"
        parity = _only(result.findings, rule="inventory-parity")
        assert parity.subject == "scipy"

    def test_undeclared_runtime_entry_is_reported(self):
        components = _real_inventory()
        requirements = [r for r in _real_requirements() if r.name not in {"pypdf"}]
        missing, extra = upstream.inventory_parity(components, requirements)
        assert missing == [] and extra == ["pypdf"]

    def test_inventory_rejects_unknown_keys_and_copied_ranges(self):
        doc = tomllib.loads(upstream.DEFAULT_INVENTORY.read_text(encoding="utf-8"))
        doc["component"][0]["declared"] = ">=1.0"
        with pytest.raises(ValueError, match="declared"):
            upstream.parse_inventory(doc)

    def test_inventory_rejects_unknown_kind(self):
        with pytest.raises(ValueError, match="kind"):
            _component(kind="npm")

    def test_steady_state_real_inventory_passes_without_findings(self, tmp_path):
        components = _real_inventory()
        result = upstream.check_upstream(
            components,
            requirements=_real_requirements(),
            fetcher=FakeFetcher(_steady_responses(components)),
            today=TODAY,
            protocol_probe=_steady_probe,
        )
        assert result.status == "pass", [f.to_dict() for f in result.findings]
        assert result.findings == []
        assert result.check == "upstream"
        out = results.write_json(result, tmp_path / "upstream.json")
        results.validate_result(json.loads(out.read_text(encoding="utf-8")))

    def test_recorded_snapshot_classification(self):
        """The payloads recorded on 2026-10-15: mcp 2.3.0 reviewed, pandas 3 and
        plotly 7 reachable, numpy split."""
        components = _real_inventory()
        responses = _steady_responses(components)
        for name in ("mcp", "numpy", "pandas", "plotly"):
            responses[upstream.pypi_url(name)] = _recorded(f"pypi_{name}")
        responses[
            upstream.github_releases_url("modelcontextprotocol/modelcontextprotocol")
        ] = _recorded("github_spec_releases")
        result = upstream.check_upstream(
            components,
            requirements=_real_requirements(),
            fetcher=FakeFetcher(responses),
            today=TODAY,
            protocol_probe=_steady_probe,
        )
        assert result.status == "pass"
        by_subject = {(f.rule, f.subject): f for f in result.findings}
        assert by_subject[("upstream-major", "pandas 3.0")].classification == (
            "breaking-candidate"
        )
        assert by_subject[("upstream-major", "plotly 7.1")].classification == (
            "breaking-candidate"
        )
        assert by_subject[("matrix-split", "numpy 2.5")].classification == "review"
        assert not [f for f in result.findings if f.subject.startswith("mcp")]
        assert result.facts["components"]["mcp"]["outcome"] == "harmless"

    def test_cli_writes_a_valid_result(self, tmp_path, monkeypatch, capsys):
        components = _real_inventory()
        monkeypatch.setattr(
            upstream, "http_fetcher", FakeFetcher(_steady_responses(components))
        )
        monkeypatch.setattr(upstream, "sdk_protocol_versions", _steady_probe)
        monkeypatch.setattr(upstream, "_today", lambda: TODAY)
        out = tmp_path / "upstream.json"
        code = audit_main.main(["upstream", "--json-out", str(out)])
        assert code == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        results.validate_result(doc)
        assert doc["check"] == "upstream" and doc["status"] == "pass"
        assert "upstream pass" in capsys.readouterr().out

    def test_github_token_is_sent_only_to_the_github_api(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-token-value")
        api = upstream.request_headers("https://api.github.com/repos/a/b/releases")
        assert api["Authorization"] == "Bearer test-token-value"
        pypi = upstream.request_headers("https://pypi.org/pypi/mcp/json")
        assert "Authorization" not in pypi
        assert pypi["User-Agent"].startswith("predictive-maintenance-mcp")
