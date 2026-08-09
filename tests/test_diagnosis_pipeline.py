"""Tests for the integrated diagnosis pipeline."""

import json
import pickle
import numpy as np
import pytest

from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    diagnose_vibration,
    _run_anomaly_detection,
    _extract_time_domain_features,
)


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
            machine_group=2,
            support_type="rigid",
            signal_unit="g",
        )

        assert result["signal_id"] == "healthy_test"
        assert result["rpm"] == 1500
        assert result["machine_group"] == 2
        assert result["support_type"] == "rigid"
        assert "fft_summary" in result
        assert "psd_summary" in result
        assert "stft_summary" in result
        assert "iso_severity" in result
        assert "overall_diagnosis" in result
        assert "evidence_strength" in result
        assert "confidence" not in result
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
            machine_group=2,
            support_type="rigid",
            signal_unit="g",
        )

        assert result["bearing_id"] == "6205"
        assert result["bearing_faults"] is not None
        assert len(result["bearing_faults"]["fault_checks"]) == 4

    def test_machine_class_parameter_removed(self, healthy_signal):
        """The invented machine_class I-IV vocabulary no longer exists."""
        signal, fs = healthy_signal
        with pytest.raises(TypeError):
            diagnose_vibration(
                signal=signal,
                fs=fs,
                rpm=1500,
                machine_class="II",
                signal_unit="g",
            )

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


class TestISOSeverityRefusal:
    """U5: the ISO block is refused (schema-level) instead of guessed/failed."""

    def _velocity_signal(self, rms_mm_s=4.0, freq=50.0, fs=10000):
        """Velocity signal with a given RMS in mm/s (the audit case: today's
        heuristic would guess RMS 4 > 0.5 as 'g' acceleration)."""
        t = np.linspace(0, 1.0, fs, endpoint=False)
        amp = rms_mm_s * np.sqrt(2)
        return amp * np.sin(2 * np.pi * freq * t), fs

    def test_undeclared_unit_refuses_iso_block(self, healthy_signal):
        """No unit → status='refused' with reason + remedy; other blocks run."""
        signal, fs = healthy_signal
        result = diagnose_vibration(
            signal,
            fs,
            rpm=1500,
            signal_id="no_unit",
            anomaly_model_name="nonexistent_model_u5",
        )
        iso = result["iso_severity"]
        assert iso["status"] == "refused"
        assert "unit" in iso["reason"].lower()
        assert "load_signal" in iso["remedy"]
        assert "signal_unit=" in iso["remedy"]
        # Refusal carries no verdict fields
        assert "zone" not in iso
        # Other diagnosis blocks still ran
        assert "peak_frequency_hz" in result["fft_summary"]
        assert "total_power" in result["psd_summary"]
        assert "num_time_bins" in result["stft_summary"]
        # Remedy surfaces in the recommendations, refusal in the text
        assert any("load_signal" in r for r in result["recommendations"])
        assert "not assessed" in result["overall_diagnosis"]

    def test_no_heuristic_on_velocity_amplitude(self):
        """RMS 4 mm/s velocity signal: the old heuristic guessed 'g' and
        integrated it into a wrong-but-plausible zone. Now: refusal without a
        declared unit, correct verdict only WITH the declared unit."""
        signal, fs = self._velocity_signal(rms_mm_s=4.0)

        # Without declared unit → refused, no verdict at all
        result = diagnose_vibration(
            signal, fs, rpm=3000, anomaly_model_name="nonexistent_model_u5"
        )
        assert result["iso_severity"]["status"] == "refused"

        # With declared unit 'mm/s' → assessed without integration, zone C
        # (group 2 rigid: 2.8 < 4.0 <= 4.5)
        result2 = diagnose_vibration(
            signal,
            fs,
            rpm=3000,
            signal_unit="mm/s",
            anomaly_model_name="nonexistent_model_u5",
        )
        iso2 = result2["iso_severity"]
        assert iso2["status"] == "assessed"
        assert iso2["unit_conversion_performed"] is False
        assert iso2["zone"] == "C"
        assert iso2["rms_velocity_mm_s"] == pytest.approx(4.0, rel=0.05)

    def test_declared_g_integrates_without_confirmation(self, healthy_signal):
        """Unit 'g' declared → acc→vel integration and verdict, no fake step."""
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        iso = result["iso_severity"]
        assert iso["status"] == "assessed"
        assert iso["unit_conversion_performed"] is True
        assert iso["zone"] in ("A", "B", "C", "D")

    def test_nyquist_below_iso_band_refuses_iso_only(self):
        """fs too low for the ISO band (U2 ValueError) → refused ISO block,
        while spectral blocks still run."""
        fs = 1600  # Nyquist 800 Hz < 1000 Hz ISO band upper edge
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.5 * np.sin(2 * np.pi * 50 * t)
        result = diagnose_vibration(
            signal,
            fs,
            rpm=3000,
            signal_unit="g",
            anomaly_model_name="nonexistent_model_u5",
        )
        iso = result["iso_severity"]
        assert iso["status"] == "refused"
        assert "Nyquist" in iso["reason"]
        assert "re-acquire" in iso["remedy"] or "fs >=" in iso["remedy"]
        assert "peak_frequency_hz" in result["fft_summary"]

    def test_invalid_unit_string_raises(self, healthy_signal):
        """A provided-but-unrecognized unit is caller misuse → ValueError."""
        signal, fs = healthy_signal
        with pytest.raises(ValueError, match="signal_unit"):
            diagnose_vibration(signal, fs, rpm=1500, signal_unit="furlongs")

    def test_assessed_block_has_status(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        assert result["iso_severity"]["status"] == "assessed"


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

    def test_evidence_strength_vocabulary(self, healthy_signal):
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        assert result["evidence_strength"] in ("none", "weak", "moderate", "strong")

    def test_quiet_machine_evidence_not_high(self, healthy_signal):
        """A very quiet machine must NOT get strong fault evidence.

        The old code derived 'confidence' from the severity score and
        returned 'high' for a machine with zero findings.
        """
        signal, fs = healthy_signal
        result = diagnose_vibration(
            signal,
            fs,
            rpm=1500,
            signal_unit="g",
            anomaly_model_name="nonexistent_model_u4",
        )
        assert result["iso_severity"]["zone"] == "A"
        assert result["evidence_strength"] in ("none", "weak")

    def test_quiet_machine_no_findings_evidence_none(self):
        """Quiet machine with no shaft signature → no fault evidence at all."""
        fs = 10000
        rpm = 1500  # shaft = 25 Hz; dominant peak at 60 Hz is 2.4x — no match
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.01 * np.sin(2 * np.pi * 60 * t)
        result = diagnose_vibration(
            signal,
            fs,
            rpm=rpm,
            signal_unit="g",
            anomaly_model_name="nonexistent_model_u4",
        )
        assert result["iso_severity"]["zone"] == "A"
        assert result["evidence_strength"] == "none"


class TestAnomalyDetection:
    def test_no_model_returns_none(self, healthy_signal):
        """When no trained model exists, anomaly_detection should be None."""
        signal, fs = healthy_signal
        result = _run_anomaly_detection(signal, fs, model_name="nonexistent_model")
        assert result is None

    def test_anomaly_in_diagnosis_result(self, healthy_signal):
        """Diagnosis result should include anomaly_detection key."""
        signal, fs = healthy_signal
        result = diagnose_vibration(signal, fs, rpm=1500, signal_unit="g")
        assert "anomaly_detection" in result

    def test_with_real_model(self, healthy_signal):
        """If bearing_health_model exists, anomaly detection should run."""
        from pathlib import Path

        model_path = (
            Path(__file__).parent.parent / "models" / "bearing_health_model_model.pkl"
        )
        if not model_path.exists():
            pytest.skip("No trained model available")
        signal, fs = healthy_signal
        result = _run_anomaly_detection(signal, fs, model_name="bearing_health_model")
        # May be None if fs mismatch with model, that's OK
        if result is not None:
            assert "anomaly_ratio" in result
            assert "overall_health" in result
            assert result["overall_health"] in ("Healthy", "Suspicious", "Faulty")


class TestFeatureExtraction:
    def test_17_features(self):
        """Feature extractor should return 17 features."""
        segment = np.random.randn(1000)
        features = _extract_time_domain_features(segment)
        assert len(features) == 17
        expected_keys = {
            "mean",
            "std",
            "var",
            "rms",
            "kurtosis",
            "skewness",
            "crest_factor",
            "entropy",
            "zero_crossing_rate",
        }
        assert expected_keys.issubset(set(features.keys()))

    def test_features_deterministic(self):
        """Same input should give same features."""
        np.random.seed(42)
        segment = np.random.randn(500)
        f1 = _extract_time_domain_features(segment)
        f2 = _extract_time_domain_features(segment)
        for k in f1:
            assert f1[k] == pytest.approx(f2[k])


class TestSynthesisEdgeCases:
    def _make_signal(self, rms, freq, fs=10000):
        t = np.linspace(0, 1.0, fs, endpoint=False)
        amp = rms * np.sqrt(2)
        return amp * np.sin(2 * np.pi * freq * t), fs

    def test_zone_b_diagnosis(self):
        """Zone B alone is not corroborated fault evidence."""
        signal, fs = self._make_signal(2.0, 50)
        result = diagnose_vibration(signal, fs, rpm=3000, signal_unit="mm/s")
        assert result["iso_severity"]["zone"] == "B"

    def test_zone_d_urgent_recommendation(self):
        """Zone D should produce IMMEDIATE ACTION recommendation."""
        signal, fs = self._make_signal(10.0, 50)
        result = diagnose_vibration(signal, fs, rpm=3000, signal_unit="mm/s")
        assert result["iso_severity"]["zone"] == "D"
        assert any("IMMEDIATE" in r for r in result["recommendations"])

    def test_zone_d_alone_not_strong_evidence(self):
        """Severity alone (even zone D) must never reach 'strong' evidence.

        Elevated broadband vibration is one finding; 'strong' requires
        corroboration from independent analyses.
        """
        signal, fs = self._make_signal(10.0, 70)  # 70 Hz ≠ 1x/2x of 50 Hz shaft
        result = diagnose_vibration(
            signal,
            fs,
            rpm=3000,
            signal_unit="mm/s",
            anomaly_model_name="nonexistent_model_u4",
        )
        assert result["iso_severity"]["zone"] == "D"
        assert result["evidence_strength"] != "strong"

    def test_1x_unbalance_detection(self):
        """Dominant peak at 1x shaft speed should suggest unbalance."""
        fs = 10000
        rpm = 3000  # shaft = 50 Hz
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = 0.01 * np.sin(2 * np.pi * 50 * t)
        result = diagnose_vibration(signal, fs, rpm=rpm, signal_unit="g")
        # Assert on the detected 1x peak (deterministic) rather than the
        # FFT-resolution-flaky diagnosis text, mirroring the 2x test.
        peak_f = result["fft_summary"]["peak_frequency_hz"]
        assert abs(peak_f - 50) < 5
        assert "unbalance" in result["overall_diagnosis"].lower()

    def test_2x_misalignment_detection(self):
        """Dominant peak at 2x shaft speed should suggest misalignment."""
        fs = 10000
        rpm = 3000  # shaft = 50 Hz, 2x = 100 Hz
        t = np.linspace(0, 1.0, fs, endpoint=False)
        # Make 100 Hz dominant
        signal = 0.001 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.sin(2 * np.pi * 100 * t)
        result = diagnose_vibration(signal, fs, rpm=rpm, signal_unit="g")
        # Check that 2x is detected in FFT
        peak_f = result["fft_summary"]["peak_frequency_hz"]
        assert abs(peak_f - 100) < 5


class TestEvidenceStrengthUpperBranches:
    """Exercise the moderate (>=2.0) and strong (>=3.0) evidence branches of
    _synthesize_diagnosis, which the other tests only touch at the low end.

    Both cases use an undeclared unit so the ISO block is REFUSED (contributes
    NO points) — proving the upper branches are reached by CORROBORATION from
    independent spectral analyses, not by severity.
    """

    #: 6205 BPFO at 1800 RPM (CWRU geometry): 3.5848 x 30 Hz. Matches the
    #: value used by the golden fixtures (tests/_golden_signals.py).
    BPFO_6205_1800 = 107.54
    RPM = 1800  # shaft = 30 Hz

    def _modulated_carrier(self, t):
        """3 kHz carrier AM-modulated at BPFO + 2nd + 3rd harmonic.

        Modulation stays strictly positive (min 0.2), so the Hilbert envelope
        recovers BPFO with two harmonics -> "high" bearing evidence (+2.0).
        """
        f = self.BPFO_6205_1800
        modulation = (
            1.0
            + 0.40 * np.sin(2 * np.pi * f * t)
            + 0.25 * np.sin(2 * np.pi * 2 * f * t)
            + 0.15 * np.sin(2 * np.pi * 3 * f * t)
        )
        return np.sin(2 * np.pi * 3000.0 * t) * modulation

    def test_single_corroboration_is_moderate(self):
        """One independent finding (a high-evidence bearing fault, +2.0) and
        nothing else -> evidence_strength == 'moderate'.

        The carrier dominates the FFT (no 1x/2x shaft match), the unit is
        undeclared (ISO refused), and no anomaly model exists -> the ONLY
        contribution is the bearing check.
        """
        fs = 10000
        t = np.arange(int(fs)) / fs
        rng = np.random.default_rng(123)
        signal = self._modulated_carrier(t) + 0.02 * rng.standard_normal(t.size)

        result = diagnose_vibration(
            signal,
            fs,
            rpm=self.RPM,
            signal_id="moderate_case",
            bearing_id="6205",
            anomaly_model_name="nonexistent_model_evidence",
        )

        assert result["iso_severity"]["status"] == "refused"  # no ISO points
        strengths = {
            c["evidence_strength"] for c in result["bearing_faults"]["fault_checks"]
        }
        assert "high" in strengths  # bearing check contributes +2.0
        assert result["evidence_strength"] == "moderate"

    def test_two_corroborations_are_strong(self):
        """Two independent findings — a high-evidence bearing fault (+2.0) AND
        a dominant 1x shaft peak / unbalance (+1.0) — reach 'strong' (>=3.0),
        with the ISO block still refused (contributing nothing)."""
        fs = 10000
        t = np.arange(int(fs)) / fs
        rng = np.random.default_rng(123)
        # Dominant shaft component at 1x (30 Hz) so the FFT peak lands on the
        # shaft frequency (+1.0), on top of the modulated bearing carrier.
        shaft = 1.5 * np.sin(2 * np.pi * 30.0 * t)
        signal = shaft + self._modulated_carrier(t) + 0.02 * rng.standard_normal(t.size)

        result = diagnose_vibration(
            signal,
            fs,
            rpm=self.RPM,
            signal_id="strong_case",
            bearing_id="6205",
            anomaly_model_name="nonexistent_model_evidence",
        )

        assert result["iso_severity"]["status"] == "refused"  # no ISO points
        peak_f = result["fft_summary"]["peak_frequency_hz"]
        assert abs(peak_f - 30.0) < 5  # dominant 1x shaft peak
        strengths = {
            c["evidence_strength"] for c in result["bearing_faults"]["fault_checks"]
        }
        assert "high" in strengths  # bearing check contributes +2.0
        assert result["evidence_strength"] == "strong"
