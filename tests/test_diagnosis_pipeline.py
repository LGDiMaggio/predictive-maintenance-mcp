"""Tests for the integrated diagnosis pipeline."""

import numpy as np
import pytest

from predictive_maintenance_mcp.diagnosis_pipeline import diagnose_vibration


@pytest.fixture
def healthy_signal():
    """Low-amplitude sine wave simulating healthy machine."""
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Small vibration at shaft frequency (25 Hz = 1500 RPM)
    signal = 0.01 * np.sin(2 * np.pi * 25 * t) + 0.001 * np.random.randn(fs)
    return signal, fs


@pytest.fixture
def faulty_signal():
    """High-amplitude signal with bearing fault modulation."""
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Shaft at 1500 RPM = 25 Hz
    shaft = 0.5 * np.sin(2 * np.pi * 25 * t)
    # Carrier at 3000 Hz modulated by BPFO (~107 Hz for 6205 at 1797 RPM)
    carrier = np.sin(2 * np.pi * 3000 * t)
    modulation = 1.0 + 0.8 * np.sin(2 * np.pi * 107 * t)
    bearing = carrier * modulation
    signal = shaft + 0.3 * bearing + 0.05 * np.random.randn(fs)
    return signal, fs


class TestDiagnoseVibration:
    def test_basic_diagnosis(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(
            signal=signal,
            fs=fs,
            rpm=1500,
            signal_id="healthy_test",
            machine_class="II",
            signal_unit="g",
        )

        assert result["signal_id"] == "healthy_test"
        assert result["rpm"] == 1500
        assert result["machine_class"] == "II"
        assert "fft_summary" in result
        assert "psd_summary" in result
        assert "stft_summary" in result
        assert "iso_severity" in result
        assert "overall_diagnosis" in result
        assert "confidence" in result
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_diagnosis_with_bearing(self, faulty_signal):
        signal, fs = faulty_signal
        result = diagnose_vibration(
            signal=signal,
            fs=fs,
            rpm=1797,
            signal_id="faulty_test",
            bearing_id="6205",
            machine_class="II",
            signal_unit="g",
        )

        assert result["bearing_id"] == "6205"
        assert result["bearing_faults"] is not None
        assert len(result["bearing_faults"]["fault_checks"]) == 4

    def test_diagnosis_without_bearing(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(
            signal=signal,
            fs=fs,
            rpm=1500,
            bearing_id=None,
            signal_unit="g",
        )
        assert result["bearing_faults"] is None

    def test_diagnosis_bad_bearing_id(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(
            signal=signal,
            fs=fs,
            rpm=1500,
            bearing_id="NONEXISTENT_999",
            signal_unit="g",
        )
        # Should handle gracefully (bearing analysis skipped)
        assert result["bearing_faults"] is None


class TestFFTSummary:
    def test_fft_summary_structure(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        fft = result["fft_summary"]
        assert "peak_frequency_hz" in fft
        assert "peak_magnitude" in fft
        assert "rms_spectral" in fft
        assert "top_peaks" in fft


class TestPSDSummary:
    def test_psd_summary_structure(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        psd = result["psd_summary"]
        assert "total_power" in psd
        assert "top_peaks" in psd


class TestISOSeverity:
    def test_iso_severity_in_diagnosis(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        iso = result["iso_severity"]
        assert "zone" in iso
        assert iso["zone"] in ("A", "B", "C", "D")
        assert "rms_velocity_mm_s" in iso


class TestRecommendations:
    def test_always_has_recommendations(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        assert len(result["recommendations"]) >= 1

    def test_confidence_level(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        assert result["confidence"] in ("high", "moderate", "low")
