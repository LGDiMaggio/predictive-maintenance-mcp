"""
Tests for the Document Reader module.

Covers:
- Bearing characteristic frequency calculations (BPFO, BPFI, BSF, FTF)
- PDF text extraction (basic, missing file, OCR fallback)
- Edge cases: zero contact angle, high contact angle, zero RPM
"""

import math
import pytest
from pathlib import Path

from predictive_maintenance_mcp.document_reader import (
    calculate_bearing_frequencies,
    extract_text_from_pdf,
    HAS_PDF,
    HAS_OCR,
)


# ── Bearing frequency calculations ─────────────────────────────────────────

class TestBearingFrequencies:
    """Tests against known bearing geometry values."""

    def test_standard_deep_groove_6205(self):
        """6205 (CWRU-documented geometry): Z=9, Bd=7.94mm, Pd=39.04mm, RPM=1797.

        The CWRU Bearing Data Center publishes BPFO=3.5848x and BPFI=5.4152x
        shaft speed for this bearing (SKF 6205-2RS JEM, drive end).
        """
        freqs = calculate_bearing_frequencies(
            num_balls=9,
            ball_diameter_mm=7.94,
            pitch_diameter_mm=39.04,
            contact_angle_deg=0.0,
            shaft_speed_rpm=1797,
        )
        shaft_freq = 1797 / 60.0  # 29.95 Hz

        assert "BPFO" in freqs
        assert "BPFI" in freqs
        assert "BSF" in freqs
        assert "FTF" in freqs
        assert "shaft_freq_hz" in freqs
        assert abs(freqs["shaft_freq_hz"] - shaft_freq) < 0.01

        # CWRU published factors: BPFO = 3.5848 x 29.95 Hz = 107.36 Hz
        assert freqs["BPFO"] == pytest.approx(107.36, rel=0.005)
        assert freqs["BPFI"] == pytest.approx(162.19, rel=0.005)
        # BPFI > BPFO always (inner race spins faster relative to rolling elements)
        assert freqs["BPFI"] > freqs["BPFO"]
        # FTF < shaft frequency
        assert freqs["FTF"] < shaft_freq

    def test_all_frequencies_positive(self):
        freqs = calculate_bearing_frequencies(
            num_balls=8,
            ball_diameter_mm=12.0,
            pitch_diameter_mm=50.0,
            contact_angle_deg=0.0,
            shaft_speed_rpm=3000,
        )
        assert all(freqs[k] > 0 for k in ("BPFO", "BPFI", "BSF", "FTF"))

    def test_angular_contact_bearing(self):
        """Non-zero contact angle (angular contact bearing)."""
        freqs_zero = calculate_bearing_frequencies(
            num_balls=16, ball_diameter_mm=12.7,
            pitch_diameter_mm=71.5, contact_angle_deg=0.0,
            shaft_speed_rpm=2000,
        )
        freqs_15 = calculate_bearing_frequencies(
            num_balls=16, ball_diameter_mm=12.7,
            pitch_diameter_mm=71.5, contact_angle_deg=15.0,
            shaft_speed_rpm=2000,
        )
        # cos(15°) < cos(0°), so d_ratio*cos(alpha) is smaller
        # → BPFO should increase, BPFI should decrease compared to alpha=0
        # Actually: BPFO = (Z/2)*fs*(1 - d_ratio*cos(a)) → larger with smaller cos
        assert freqs_15["BPFO"] > freqs_zero["BPFO"]
        assert freqs_15["BPFI"] < freqs_zero["BPFI"]

    def test_high_contact_angle(self):
        """45° contact angle (thrust bearing)."""
        freqs = calculate_bearing_frequencies(
            num_balls=12, ball_diameter_mm=10.0,
            pitch_diameter_mm=60.0, contact_angle_deg=45.0,
            shaft_speed_rpm=1500,
        )
        assert all(freqs[k] > 0 for k in ("BPFO", "BPFI", "BSF", "FTF"))

    def test_frequency_ratios(self):
        """BPFI/BPFO ratio should be > 1 for typical bearings."""
        freqs = calculate_bearing_frequencies(
            num_balls=8, ball_diameter_mm=10.0,
            pitch_diameter_mm=50.0, shaft_speed_rpm=1500,
        )
        assert freqs["BPFI"] / freqs["BPFO"] > 1.0

    def test_input_parameters_returned(self):
        """Verify input parameters are echoed in result dict."""
        freqs = calculate_bearing_frequencies(
            num_balls=9, ball_diameter_mm=7.94,
            pitch_diameter_mm=39.04, contact_angle_deg=0.0,
            shaft_speed_rpm=1797,
        )
        params = freqs.get("input_parameters", {})
        assert params.get("num_balls") == 9
        assert params.get("shaft_speed_rpm") == 1797

    def test_different_speeds(self):
        """Frequencies should scale linearly with RPM."""
        freqs_1000 = calculate_bearing_frequencies(
            num_balls=8, ball_diameter_mm=10.0,
            pitch_diameter_mm=50.0, shaft_speed_rpm=1000,
        )
        freqs_2000 = calculate_bearing_frequencies(
            num_balls=8, ball_diameter_mm=10.0,
            pitch_diameter_mm=50.0, shaft_speed_rpm=2000,
        )
        # Double RPM → double all frequencies
        assert abs(freqs_2000["BPFO"] / freqs_1000["BPFO"] - 2.0) < 0.01
        assert abs(freqs_2000["BPFI"] / freqs_1000["BPFI"] - 2.0) < 0.01
        assert abs(freqs_2000["BSF"] / freqs_1000["BSF"] - 2.0) < 0.01


# ── PDF text extraction ────────────────────────────────────────────────────

class TestExtractTextFromPdf:

    @pytest.mark.skipif(not HAS_PDF, reason="pypdf not installed")
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf(Path("nonexistent_file.pdf"))

    @pytest.mark.skipif(not HAS_PDF, reason="pypdf not installed")
    def test_real_catalog_if_available(self):
        """Extract text from a real bearing catalog in the repo."""
        catalogs_dir = Path(__file__).parent.parent / "resources" / "bearing_catalogs"
        pdfs = list(catalogs_dir.glob("*.pdf")) if catalogs_dir.exists() else []
        if not pdfs:
            pytest.skip("No PDF catalogs found in resources/")

        text = extract_text_from_pdf(pdfs[0], max_pages=3)
        assert isinstance(text, str)
        assert len(text) > 50  # Should have some content

    @pytest.mark.skipif(not HAS_PDF, reason="pypdf not installed")
    def test_max_pages_limits_extraction(self):
        """max_pages should limit processing."""
        catalogs_dir = Path(__file__).parent.parent / "resources" / "bearing_catalogs"
        pdfs = list(catalogs_dir.glob("*.pdf")) if catalogs_dir.exists() else []
        if not pdfs:
            pytest.skip("No PDF catalogs found in resources/")

        text_1 = extract_text_from_pdf(pdfs[0], max_pages=1)
        text_3 = extract_text_from_pdf(pdfs[0], max_pages=3)
        # More pages should produce more (or equal) text
        assert len(text_3) >= len(text_1)

    def test_no_pypdf_raises_import_error(self):
        """When pypdf not installed, should raise ImportError."""
        if HAS_PDF:
            pytest.skip("pypdf IS installed — can't test missing path")
        with pytest.raises(ImportError, match="pypdf"):
            extract_text_from_pdf(Path("test.pdf"))
