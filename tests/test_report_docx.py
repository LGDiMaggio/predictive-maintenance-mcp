"""
Tests for DOCX diagnostic report generation.

Covers:
- Basic DOCX creation with all sections
- Partial sections (some missing)
- Empty sections
- File path sanitization (no path traversal)
- Missing python-docx graceful error
"""

import pytest
from pathlib import Path
from unittest.mock import patch

# Check availability before importing
try:
    from docx import Document as DocxDocument

    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from predictive_maintenance_mcp.report_generator import (
    save_diagnostic_report_docx,
    REPORTS_DIR,
)

# ── DOCX generation ───────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_DOCX, reason="python-docx not installed")
class TestDocxReportGeneration:

    def _full_sections(self):
        return {
            "statistics": {
                "RMS": 5.234,
                "Crest Factor": 3.2,
                "Kurtosis": 4.1,
                "Skewness": 0.05,
            },
            "fft_peaks": [
                {"frequency": 120.5, "magnitude_db": 5.2, "note": "shaft freq"},
                {"frequency": 241.0, "magnitude_db": 3.1, "note": "2x shaft"},
            ],
            "envelope_peaks": [
                {"frequency": 107.36, "magnitude_db": 2.5, "match": "BPFO"},
                {"frequency": 214.72, "magnitude_db": 1.8, "match": "2x BPFO"},
            ],
            "bearing_frequencies": {
                # 6205 (CWRU geometry) at 1797 RPM
                "BPFO": 107.36,
                "BPFI": 162.19,
                "BSF": 70.58,
                "FTF": 11.93,
            },
            "iso": {
                "zone": "B",
                "severity_level": "Acceptable",
                "rms_velocity": 3.5,
                "thresholds": "2.3 - 4.5 mm/s",
            },
            "diagnosis": "Bearing shows early-stage outer race fault. Monitor closely.",
        }

    def test_full_report(self, tmp_path):
        """Generate report with all sections."""
        with patch.object(
            type(REPORTS_DIR),
            "__fspath__",
            return_value=str(tmp_path),
            create=True,
        ):
            pass  # We'll use the real REPORTS_DIR but clean up

        result = save_diagnostic_report_docx(
            signal_file="test_signal.csv",
            sections=self._full_sections(),
        )
        assert "error" not in result
        assert "file_path" in result
        assert result["file_path"].endswith(".docx")
        output = Path(result["file_path"])
        assert output.exists()
        assert output.stat().st_size > 0

        # Clean up
        output.unlink(missing_ok=True)

    def test_empty_sections(self):
        """Report with no sections should still create a file (just title)."""
        result = save_diagnostic_report_docx(
            signal_file="minimal.csv",
            sections={},
        )
        assert "error" not in result
        output = Path(result["file_path"])
        assert output.exists()
        output.unlink(missing_ok=True)

    def test_partial_sections(self):
        """Report with only statistics and diagnosis."""
        result = save_diagnostic_report_docx(
            signal_file="partial_test.csv",
            sections={
                "statistics": {"RMS": 2.5, "Kurtosis": 3.0},
                "diagnosis": "No issues detected.",
            },
        )
        assert "error" not in result
        output = Path(result["file_path"])
        assert output.exists()
        output.unlink(missing_ok=True)

    def test_custom_title(self):
        """Report with custom title."""
        result = save_diagnostic_report_docx(
            signal_file="titled.csv",
            sections={"diagnosis": "All clear."},
            title="Custom Report Title",
        )
        assert "error" not in result
        output = Path(result["file_path"])
        assert output.exists()
        output.unlink(missing_ok=True)

    def test_filename_sanitized(self):
        """Path traversal in signal_file should be sanitized."""
        result = save_diagnostic_report_docx(
            signal_file="../../etc/passwd.csv",
            sections={"diagnosis": "test"},
        )
        assert "error" not in result
        output = Path(result["file_path"])
        # Should NOT be outside REPORTS_DIR
        assert output.parent == REPORTS_DIR
        assert ".." not in output.name
        output.unlink(missing_ok=True)

    def test_result_metadata(self):
        """Result dict should contain metadata."""
        result = save_diagnostic_report_docx(
            signal_file="meta_test.csv",
            sections={"statistics": {"RMS": 1.0}},
        )
        assert "file_name" in result
        assert "file_size_kb" in result
        assert result["file_size_kb"] > 0
        Path(result["file_path"]).unlink(missing_ok=True)


# ── Missing python-docx ───────────────────────────────────────────────────


def test_missing_docx_raises():
    """When HAS_DOCX is False, raise ValueError (never an error dict)."""
    with patch("predictive_maintenance_mcp.report_generator.HAS_DOCX", False):
        with pytest.raises(ValueError, match="python-docx"):
            save_diagnostic_report_docx(
                signal_file="test.csv",
                sections={"statistics": {"RMS": 1.0}},
            )
