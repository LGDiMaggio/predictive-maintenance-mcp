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
    REPORTS_DIR,
    save_integrated_diagnostic_report,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import THRESHOLD_PROVENANCE

from tests.test_advisory import (
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
