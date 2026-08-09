"""Tests for the integrated diagnostic report rendering.

The rendering layer places strings; it never composes them. These tests are
the enforcement of that: every evaluative sentence in the rendered document
must be findable, verbatim, in the advisory payload that produced it.
"""

import html
import json
import re
from pathlib import Path

import numpy as np
import pytest

from predictive_maintenance_mcp.decision_support.advisory import (
    build_advisory,
    collect_statements,
)
from predictive_maintenance_mcp.figures import build_annotated_envelope_figure
from predictive_maintenance_mcp.integrated_report import (
    create_integrated_diagnostic_report,
)
from predictive_maintenance_mcp.report_generator import (
    HAS_PDF,
    REPORTS_DIR,
    save_integrated_diagnostic_report,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import THRESHOLD_PROVENANCE

# Sibling-module import, not `from tests.…`: pytest puts the tests directory
# on sys.path, but only `python -m pytest` also puts the repo root there. CI
# invokes the console script, where the packaged form would not resolve.
from test_advisory import (  # noqa: E402
    _anomaly,
    _bearing_faults,
    _diagnosis,
    _iso_assessed,
    _iso_refused,
)


@pytest.fixture
def advisory():
    return build_advisory(
        _diagnosis(
            iso=_iso_assessed(zone="B", rms=1.64),
            bearing_faults=_bearing_faults(),
            anomaly=_anomaly(health="Faulty", ratio=1.0),
        )
    )


@pytest.fixture
def figure():
    freqs = np.linspace(0.0, 500.0, 1500)
    mags = 0.001 + 0.05 * np.exp(-(((freqs - 80.5) / 0.8) ** 2))
    return build_annotated_envelope_figure(
        freqs,
        mags,
        {"BPFO": 81.125, "BPFI": 118.875, "BSF": 63.91, "FTF": 14.84},
        matched={"fault_type": "BPFO", "measured_hz": 80.5, "deviation_pct": 0.77},
    )


class TestRenderedContent:
    def test_document_carries_the_verdict_and_the_iso_caveat(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert html.escape(advisory["verdict"]["statement"]) in rendered
        assert html.escape(THRESHOLD_PROVENANCE) in rendered

    def test_every_authored_statement_appears_verbatim(self, advisory, figure):
        """Covers AE4."""
        rendered = create_integrated_diagnostic_report(advisory, figure)
        missing = [
            statement
            for statement in collect_statements(advisory)
            if html.escape(statement) not in rendered
        ]
        assert missing == [], f"rendering dropped authored statements: {missing}"

    def test_disagreement_statement_is_rendered(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert advisory["disagreements"], "fixture should produce a disagreement"
        assert html.escape(advisory["disagreements"][0]["statement"]) in rendered

    def test_every_recommendation_is_rendered_with_its_motivation(
        self, advisory, figure
    ):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        for rec in advisory["recommendations"]:
            assert html.escape(rec["action"]) in rendered
            assert html.escape(rec["motivation"]) in rendered

    def test_absence_statements_render_as_sections_not_omissions(self, figure):
        sparse = build_advisory(
            _diagnosis(iso=_iso_refused(), bearing_faults=None, anomaly=None)
        )
        rendered = create_integrated_diagnostic_report(sparse, None)
        for key in ("iso", "bearing_match", "anomaly", "baseline_comparison"):
            assert html.escape(sparse[key]["statement"]) in rendered

    def test_rendering_escapes_content_rather_than_interpolating_it_raw(self, figure):
        hostile = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        hostile["signal_id"] = "<script>alert(1)</script>"
        hostile["provenance"]["signal_id"] = "<script>alert(1)</script>"
        rendered = create_integrated_diagnostic_report(hostile, figure)
        assert "<script>alert(1)</script>" not in rendered
        assert "&lt;script&gt;" in rendered

    def test_metadata_block_cannot_be_closed_early_by_signal_content(self, figure):
        """The embedded JSON is tokenised as HTML before it is parsed as JSON."""
        hostile = build_advisory(
            _diagnosis(bearing_faults=_bearing_faults(), anomaly=_anomaly())
        )
        hostile["provenance"]["signal_id"] = "</script><script>alert(1)</script>"
        rendered = create_integrated_diagnostic_report(hostile, figure)
        _, _, tail = rendered.partition(
            '<script type="application/json" id="report-metadata">'
        )
        json_block, closed, _ = tail.partition("</script>")

        # The block survives to its own closing tag rather than being cut
        # short by the payload, and what it holds is still valid JSON
        # carrying the hostile string inertly as data.
        assert closed, "metadata block was never closed"
        assert "</script>" not in json_block
        parsed = json.loads(json_block.replace("<\\/", "</"))
        assert parsed["provenance"]["signal_id"] == (
            "</script><script>alert(1)</script>"
        )

    def test_figure_caption_is_rendered(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        assert html.escape(figure["caption"]) in rendered

    def test_missing_figure_states_the_absence(self, advisory):
        rendered = create_integrated_diagnostic_report(advisory, None)
        assert "no envelope spectrum" in rendered.lower()


class TestProvenanceAndDeterminism:
    def test_provenance_names_signal_parameters_and_version(self, advisory, figure):
        rendered = create_integrated_diagnostic_report(advisory, figure)
        prov = advisory["provenance"]
        assert html.escape(str(prov["signal_id"])) in rendered
        assert str(prov["rpm"]) in rendered
        assert html.escape(str(prov["bearing_id"])) in rendered
        assert html.escape(prov["server_version"]) in rendered

    def test_same_payload_and_timestamp_render_identically(self, advisory, figure):
        """Covers AE5."""
        first = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        second = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        assert first == second

    def test_two_renders_differ_only_in_the_generation_timestamp(
        self, advisory, figure
    ):
        """Covers AE5.

        Determinism is a claim about content, not filenames — report names
        carry a timestamp and a sequence on purpose so a re-run never
        overwrites its predecessor.
        """
        first = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2026-08-06T10:00:00Z"
        )
        second = create_integrated_diagnostic_report(
            advisory, figure, generated_at="2027-01-01T23:59:59Z"
        )
        strip = lambda doc: re.sub(r"20\d\d-\d\d-\d\dT[\d:]+Z", "", doc)  # noqa: E731
        assert strip(first) == strip(second)


class TestSaving:
    def test_report_is_written_under_the_reports_directory(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure)
        written = Path(result["file_path"])
        assert written.exists()
        assert written.parent.resolve() == REPORTS_DIR.resolve()
        written.unlink()

    def test_traversal_shaped_signal_id_cannot_escape_the_reports_directory(
        self, advisory, figure
    ):
        advisory["signal_id"] = "../../../etc/passwd"
        advisory["provenance"]["signal_id"] = "../../../etc/passwd"
        result = save_integrated_diagnostic_report(advisory, figure)
        written = Path(result["file_path"])
        assert written.parent.resolve() == REPORTS_DIR.resolve()
        written.unlink()

    def test_consecutive_saves_never_overwrite(self, advisory, figure):
        first = save_integrated_diagnostic_report(advisory, figure)
        second = save_integrated_diagnostic_report(advisory, figure)
        assert first["file_name"] != second["file_name"]
        Path(first["file_path"]).unlink()
        Path(second["file_path"]).unlink()

    def test_result_carries_the_authored_statements(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure)
        assert result["statements"] == collect_statements(advisory)
        assert result["report_type"] == "integrated_diagnostic"
        Path(result["file_path"]).unlink()


# ---------------------------------------------------------------------------
# PDF rendering and cross-rendering parity
#
# The parity guarantee is what makes "two renderings, one authored source"
# a property rather than an intention. Both files are produced from the same
# rendered HTML string, so this test would only fail if that stopped being
# true — which is exactly when someone needs to hear about it.
# ---------------------------------------------------------------------------


def _normalise(text):
    """Collapse whitespace so line breaks introduced by PDF layout do not
    make an identical sentence look different."""
    return " ".join(text.split())


def _pdf_text(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return _normalise(" ".join(page.extract_text() or "" for page in reader.pages))


@pytest.mark.skipif(not HAS_PDF, reason="PDF extra not installed")
class TestPdfRendering:
    def test_pdf_is_written_and_non_empty(self, advisory, figure):
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        written = Path(result["files"][0]["file_path"])
        assert written.exists()
        assert written.read_bytes()[:5] == b"%PDF-"
        assert result["files"][0]["file_size_kb"] > 0
        written.unlink()

    def test_both_renderings_are_produced_from_one_request(self, advisory, figure):
        result = save_integrated_diagnostic_report(
            advisory, figure, formats=["html", "pdf"]
        )
        assert [f["format"] for f in result["files"]] == ["html", "pdf"]
        for entry in result["files"]:
            path = Path(entry["file_path"])
            assert path.exists()
            path.unlink()

    def test_every_authored_statement_survives_into_the_pdf(self, advisory, figure):
        """Covers AE5 across renderings."""
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        path = Path(result["files"][0]["file_path"])
        try:
            extracted = _pdf_text(path)
            missing = [
                s
                for s in collect_statements(advisory)
                if _normalise(s) not in extracted
            ]
            assert missing == [], f"PDF dropped authored statements: {missing}"
        finally:
            path.unlink()

    def test_the_iso_caveat_survives_into_the_pdf(self, advisory, figure):
        """The caveat is the single sentence most likely to be lost in a
        rendering that silently truncates."""
        result = save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
        path = Path(result["files"][0]["file_path"])
        try:
            assert _normalise(THRESHOLD_PROVENANCE) in _pdf_text(path)
        finally:
            path.unlink()

    def test_pdf_and_html_agree_on_every_statement(self, advisory, figure):
        result = save_integrated_diagnostic_report(
            advisory, figure, formats=["html", "pdf"]
        )
        html_path = Path(result["files"][0]["file_path"])
        pdf_path = Path(result["files"][1]["file_path"])
        try:
            rendered_html = html_path.read_text(encoding="utf-8")
            extracted_pdf = _pdf_text(pdf_path)
            for statement in collect_statements(advisory):
                in_html = html.escape(statement) in rendered_html
                in_pdf = _normalise(statement) in extracted_pdf
                assert (
                    in_html == in_pdf is True
                ), f"statement present in one rendering only: {statement}"
        finally:
            html_path.unlink()
            pdf_path.unlink()


class TestFormatValidation:
    def test_unknown_format_raises(self, advisory, figure):
        with pytest.raises(ValueError, match="Unknown report format"):
            save_integrated_diagnostic_report(advisory, figure, formats=["docx"])

    @pytest.mark.skipif(HAS_PDF, reason="PDF extra installed")
    def test_missing_pdf_dependency_raises_with_an_install_hint(self, advisory, figure):
        with pytest.raises(ValueError, match=r"\[pdf\]"):
            save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])

    def test_missing_pdf_dependency_message_names_the_extra(self, advisory, figure):
        """Exercised regardless of whether the extra is installed locally."""
        import predictive_maintenance_mcp.report_generator as rg

        original = rg.HAS_PDF
        rg.HAS_PDF = False
        try:
            with pytest.raises(ValueError) as excinfo:
                save_integrated_diagnostic_report(advisory, figure, formats=["pdf"])
            assert "[pdf]" in str(excinfo.value)
        finally:
            rg.HAS_PDF = original
