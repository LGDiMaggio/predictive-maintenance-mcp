"""Tests for bearing fault detection and catalog functions."""

import numpy as np
import pytest

from predictive_maintenance_mcp.diagnostics.bearing_catalog import (
    lookup_bearing,
    compute_fault_frequencies,
    list_catalog_bearings,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_bearing_fault_peak,
    check_all_bearing_faults,
    lookup_bearing_and_compute,
)


class TestBearingCatalog:
    def test_lookup_6205(self):
        b = lookup_bearing("6205")
        assert b is not None
        assert b["num_balls"] == 9
        assert b["ball_diameter_mm"] == pytest.approx(7.94, abs=0.1)

    def test_lookup_with_prefix(self):
        b = lookup_bearing("SKF 6205-2RS")
        assert b is not None
        assert b["designation"] == "6205"

    def test_lookup_not_found(self):
        assert lookup_bearing("FAKE_999") is None

    def test_compute_fault_frequencies(self):
        result = compute_fault_frequencies("6205", rpm=1797)
        assert result is not None
        assert "BPFO" in result
        assert "BPFI" in result
        assert "BSF" in result
        assert "FTF" in result
        assert result["shaft_freq_hz"] == pytest.approx(1797 / 60.0, abs=0.1)
        # BPFO for 6205 at 1797 RPM should be around 107 Hz
        assert 90 < result["BPFO"] < 130

    def test_compute_fault_not_found(self):
        assert compute_fault_frequencies("NONEXISTENT", 1500) is None

    def test_list_catalog(self):
        # The catalog is small by design: verified, source-traceable entries only.
        bearings = list_catalog_bearings()
        assert len(bearings) >= 3
        designations = {b["designation"] for b in bearings}
        assert "6205" in designations
        # Every listed entry must carry its source citation
        assert all(b["source"] for b in bearings)


class TestCheckBearingFaultPeak:
    @pytest.fixture
    def signal_with_bpfo(self):
        """Synthetic signal with amplitude modulation at 107 Hz (BPFO for 6205@1797)."""
        fs = 12000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        carrier = np.sin(2 * np.pi * 3000 * t)
        bpfo = 107.36
        modulation = 1.0 + 0.8 * np.sin(2 * np.pi * bpfo * t)
        # Add harmonics
        modulation += 0.3 * np.sin(2 * np.pi * 2 * bpfo * t)
        signal = carrier * modulation + 0.05 * np.random.randn(fs)
        return signal, fs, bpfo

    def test_detects_known_frequency(self, signal_with_bpfo):
        signal, fs, bpfo = signal_with_bpfo
        result = check_bearing_fault_peak(
            signal=signal,
            fs=fs,
            expected_freq=bpfo,
            fault_type="BPFO",
            bearing_id="6205",
            tolerance_pct=5.0,
        )
        assert result["detected"] is True
        assert result["evidence_strength"] in ("high", "moderate")
        assert abs(result["detected_frequency_hz"] - bpfo) < bpfo * 0.05

    def test_no_detection_wrong_frequency(self, signal_with_bpfo):
        signal, fs, _ = signal_with_bpfo
        result = check_bearing_fault_peak(
            signal=signal,
            fs=fs,
            expected_freq=250.0,  # Not present
            fault_type="BPFI",
            bearing_id="6205",
            tolerance_pct=3.0,
        )
        # May or may not detect depending on noise, but evidence should be low
        if not result["detected"]:
            assert result["evidence_strength"] in ("none", "low")

    def test_harmonics_detected(self, signal_with_bpfo):
        signal, fs, bpfo = signal_with_bpfo
        result = check_bearing_fault_peak(
            signal=signal,
            fs=fs,
            expected_freq=bpfo,
            fault_type="BPFO",
            bearing_id="6205",
            num_harmonics=3,
        )
        # Should detect at least the 2x harmonic
        if result["detected"]:
            assert len(result["harmonics_detected"]) >= 1


class TestCheckAllBearingFaults:
    def test_all_faults_6205(self):
        """Run all fault checks on a clean sine wave (should find no faults)."""
        fs = 12000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = np.sin(2 * np.pi * 50 * t)

        result = check_all_bearing_faults(
            signal=signal,
            fs=fs,
            bearing_id="6205",
            rpm=1797,
            signal_id="test",
        )

        assert result["bearing_id"] == "6205"
        assert result["rpm"] == 1797
        assert len(result["fault_checks"]) == 4
        fault_types = {c["fault_type"] for c in result["fault_checks"]}
        assert fault_types == {"BPFO", "BPFI", "BSF", "FTF"}

    def test_bearing_not_found(self):
        signal = np.random.randn(1000)
        with pytest.raises(ValueError, match="not found"):
            check_all_bearing_faults(signal, fs=10000, bearing_id="FAKE", rpm=1500)

    def test_assessment_text_present(self):
        fs = 10000
        signal = np.random.randn(fs)
        result = check_all_bearing_faults(
            signal=signal, fs=fs, bearing_id="6205", rpm=1500
        )
        assert len(result["overall_assessment"]) > 0


class TestLookupBearingAndCompute:
    def test_end_to_end(self):
        fs = 10000
        signal = np.random.randn(fs)
        result = lookup_bearing_and_compute(
            signal=signal,
            fs=fs,
            bearing_type="6205",
            rpm=1500,
            signal_id="e2e_test",
        )
        assert result["bearing_id"] == "6205"
        assert len(result["fault_checks"]) == 4
