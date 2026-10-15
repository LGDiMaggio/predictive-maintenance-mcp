"""Guards for the audit result contract and the ``python -m tools.audit`` CLI.

Every audit check writes a ``pmm-audit-result/1`` document through
``tools/audit/results.py``, and the report builder reads nothing else. These
tests pin that contract:

* the fingerprint: twelve hex characters that stay the same when only the
  version, timestamp, line number or matrix variant changes, and change when
  the subject does;
* the closed vocabularies: a category, severity, classification, check or
  status outside them is refused, so a typo cannot create a status the
  report builder treats as neither pass nor fail;
* the writer: canonical JSON with one trailing newline, no NaN, a visible
  truncation marker on long evidence, and no numeric array long enough to be
  a waveform (raw vibration data never leaves the machine through a report);
* the subcommand dispatch, which imports only the selected check's module so
  one check's missing dependency cannot break the others.
"""

import json
import math

import pytest

from tools.audit import __main__ as audit_main
from tools.audit import results


def _finding(**overrides):
    fields = {
        "rule": "tool-missing",
        "category": "tool-contract",
        "severity": "high",
        "subject": "analyze_fft",
        "title": "A tool from the frozen inventory is not registered",
        "evidence": {"versions": {"mcp": "2.0.0"}, "excerpt": "line 12"},
    }
    fields.update(overrides)
    return results.Finding(**fields)


def _result(**overrides):
    fields = {
        "check": "smoke",
        "variant": "checkout-win32-py3.11",
        "status": "fail",
        "started_at": "2026-10-15T10:00:00Z",
        "finished_at": "2026-10-15T10:00:30Z",
        "environment": {"python": "3.11.9"},
        "facts": {"tool_count": {"legacy": 38}},
        "findings": [_finding()],
    }
    fields.update(overrides)
    return results.Result(**fields)


class TestFingerprint:
    def test_is_twelve_lowercase_hex_characters(self):
        value = results.fingerprint("tool-contract", "tool-missing", "analyze_fft")
        assert len(value) == 12
        assert set(value) <= set("0123456789abcdef")

    def test_unchanged_when_only_version_timestamp_line_or_variant_differ(self):
        first = _result(
            variant="checkout-win32-py3.11",
            started_at="2026-10-15T10:00:00Z",
            findings=[_finding(evidence={"versions": {"mcp": "2.0.0"}, "line": 12})],
        )
        second = _result(
            variant="wheel-ubuntu-py3.14",
            started_at="2026-10-22T03:00:00Z",
            findings=[_finding(evidence={"versions": {"mcp": "2.3.0"}, "line": 90})],
        )
        assert first.findings[0].fingerprint == second.findings[0].fingerprint

    def test_changes_when_the_subject_changes(self):
        assert (
            _finding(subject="analyze_fft").fingerprint
            != _finding(subject="analyze_envelope").fingerprint
        )

    def test_subject_case_and_whitespace_are_normalized(self):
        assert results.fingerprint(
            "install", "bearing-catalog-not-shipped", "  Search_Bearing_Catalog   6205 "
        ) == results.fingerprint(
            "install", "bearing-catalog-not-shipped", "search_bearing_catalog 6205"
        )

    def test_the_finding_carries_the_function_value(self):
        finding = _finding()
        assert finding.fingerprint == results.fingerprint(
            "tool-contract", "tool-missing", "analyze_fft"
        )


class TestVocabularies:
    @pytest.mark.parametrize(
        "field, value",
        [("category", "style"), ("severity", "urgent"), ("classification", "meh")],
    )
    def test_a_finding_outside_the_closed_vocabulary_is_refused(self, field, value):
        with pytest.raises(ValueError, match=field):
            _finding(**{field: value})

    @pytest.mark.parametrize("field, value", [("check", "lint"), ("status", "ok")])
    def test_a_result_outside_the_closed_vocabulary_is_refused(self, field, value):
        with pytest.raises(ValueError, match=field):
            _result(**{field: value})


class TestResultWriter:
    def test_writes_canonical_json_with_a_single_trailing_newline(self, tmp_path):
        target = results.write_json(_result(), tmp_path / "out" / "smoke.json")
        raw = target.read_bytes()
        assert raw.endswith(b"}\n") and not raw.endswith(b"\n\n")
        assert b"\r\n" not in raw
        doc = json.loads(raw)
        assert doc["schema"] == results.RESULT_SCHEMA
        assert list(doc) == sorted(doc)
        assert doc["findings"][0]["fingerprint"] == _finding().fingerprint

    def test_rejects_nan(self, tmp_path):
        with pytest.raises(ValueError, match="non-finite"):
            results.write_json(_result(facts={"rms": math.nan}), tmp_path / "nan.json")
        assert not (tmp_path / "nan.json").exists()

    def test_truncates_a_long_evidence_excerpt_with_a_visible_marker(self, tmp_path):
        long = "x" * (results.EXCERPT_LIMIT * 3)
        result = _result(findings=[_finding(evidence={"excerpt": long})])
        doc = json.loads(results.write_json(result, tmp_path / "t.json").read_text())
        excerpt = doc["findings"][0]["evidence"]["excerpt"]
        assert len(excerpt) < len(long)
        assert "truncated" in excerpt

    def test_truncate_evidence_keeps_the_tail_on_request(self):
        text = "head-" + "x" * 500 + "-the-last-line"
        out = results.truncate_evidence(text, 100, keep="tail")
        assert out.endswith("-the-last-line")
        assert "truncated" in out
        assert results.truncate_evidence("short", 100) == "short"

    def test_refuses_a_long_numeric_array_anywhere(self, tmp_path):
        waveform = [0.1] * 1000
        result = _result(facts={"nested": {"samples": waveform}})
        with pytest.raises(ValueError, match="numeric array"):
            results.write_json(result, tmp_path / "w.json")

    def test_refuses_a_long_array_of_numeric_pairs(self, tmp_path):
        spectrum = [[float(i), 1.0] for i in range(200)]
        with pytest.raises(ValueError, match="numeric array"):
            results.write_json(_result(facts={"spectrum": spectrum}), tmp_path / "s")

    def test_accepts_a_short_numeric_array(self, tmp_path):
        peaks = [10.0 * i for i in range(results.MAX_NUMERIC_ARRAY)]
        results.write_json(_result(facts={"peaks": peaks}), tmp_path / "ok.json")


class TestCli:
    def test_every_subcommand_has_a_dispatch_entry(self):
        assert audit_main.SUBCOMMANDS == (
            "smoke",
            "package-smoke",
            "upstream",
            "floors",
            "lock-reqs",
            "security",
            "report",
        )
        assert set(audit_main.DISPATCH) == set(audit_main.SUBCOMMANDS)

    @pytest.fixture
    def absent_report(self, monkeypatch):
        """Point ``report`` at a module that does not exist, whatever the tree has."""
        monkeypatch.setitem(
            audit_main.DISPATCH,
            "report",
            audit_main.Entry(
                module="tools.audit._absent_for_test",
                add_arguments="add_arguments",
                run="run",
                help="absent on purpose",
            ),
        )

    def test_a_subcommand_whose_module_is_absent_exits_2(self, absent_report, capsys):
        assert audit_main.main(["report"]) == 2
        err = capsys.readouterr().err.strip().splitlines()
        assert len(err) == 1 and err[0].startswith("error: ")
        assert "tools.audit._absent_for_test" in err[0]

    def test_an_absent_module_does_not_break_the_other_subcommands(self, absent_report):
        """Dispatch is lazy: a subcommand's module is imported only when that
        subcommand is selected."""
        assert "report" in audit_main.build_parser().format_help()
        smoke = audit_main.build_parser("smoke")
        args = smoke.parse_args(["smoke", "--variant", "x"])
        assert args.variant == "x"
