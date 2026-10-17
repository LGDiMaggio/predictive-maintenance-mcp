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
    check_frequency_set,
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

    def test_low_fs_signal_no_raise(self):
        """A bearing check on an 8 kHz signal (Nyquist 4 kHz) must run: the
        fs-aware envelope default analyzes 500-3999 Hz instead of raising on
        the old fixed 5000 Hz upper edge (500-5000 exceeded Nyquist)."""
        fs = 8000  # Nyquist 4000 Hz < old fixed 5000 Hz upper edge
        signal = np.random.randn(fs)
        result = check_all_bearing_faults(
            signal=signal, fs=fs, bearing_id="6205", rpm=1500
        )
        assert len(result["fault_checks"]) == 4


class TestNonFiniteInputRefused:
    """Issue #82: no bearing verdict from corrupt input on a direct call.

    Before the guard a signal with one NaN sample matched no fault frequency,
    so the bearing checks returned "Bearing appears healthy"; a zero, NaN or
    infinite fs failed with an unrelated filter-design message.
    """

    FS = 12000

    def _nan_signal(self) -> np.ndarray:
        t = np.arange(self.FS) / self.FS
        signal = np.sin(2 * np.pi * 50 * t)
        signal[100] = np.nan
        return signal

    def test_peak_check_nan_sample_is_refused(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_bearing_fault_peak(
                signal=self._nan_signal(),
                fs=self.FS,
                expected_freq=107.36,
                fault_type="BPFO",
                bearing_id="6205",
            )

    @pytest.mark.parametrize("bad_fs", [0, np.nan, np.inf], ids=["zero", "nan", "inf"])
    def test_peak_check_bad_sampling_rate_is_refused(self, bad_fs):
        signal = np.sin(2 * np.pi * 50 * np.arange(self.FS) / self.FS)
        with pytest.raises(ValueError, match="fs must be positive"):
            check_bearing_fault_peak(
                signal=signal,
                fs=bad_fs,
                expected_freq=107.36,
                fault_type="BPFO",
                bearing_id="6205",
            )

    def test_all_faults_nan_sample_is_refused_not_healthy(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_all_bearing_faults(
                signal=self._nan_signal(), fs=self.FS, bearing_id="6205", rpm=1797
            )

    def test_frequency_set_nan_sample_is_refused_not_healthy(self):
        with pytest.raises(ValueError, match="non-finite"):
            check_frequency_set(
                signal=self._nan_signal(),
                fs=self.FS,
                frequencies={"BPFO": 107.36},
                rpm=1797,
            )


RPM_PHRASE = "rpm must be positive and finite"
BAD_SPEEDS = [0.0, -1797.0, np.nan, np.inf, -np.inf]
BAD_SPEED_IDS = ["zero", "negative", "nan", "inf", "-inf"]


class TestInvalidShaftSpeedRefused:
    """Issue #83: no bearing verdict from a speed that is not positive and
    finite, nor from an expected frequency that is not.

    Before the guard a zero, negative or NaN speed gave 0 Hz, negative or
    NaN fault frequencies that matched no peak, so the verdict read "Bearing
    appears healthy"; an infinite one matched every peak and read as a fault.
    """

    FS = 12000

    def _signal(self) -> np.ndarray:
        t = np.arange(self.FS) / self.FS
        return np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.default_rng(
            0
        ).standard_normal(self.FS)

    @pytest.mark.parametrize("rpm", BAD_SPEEDS, ids=BAD_SPEED_IDS)
    def test_catalog_check_refuses_the_speed(self, rpm):
        with pytest.raises(ValueError, match=RPM_PHRASE):
            check_all_bearing_faults(
                signal=self._signal(), fs=self.FS, bearing_id="6205", rpm=rpm
            )

    def test_speed_is_refused_before_the_catalog_lookup(self):
        with pytest.raises(ValueError, match=RPM_PHRASE):
            check_all_bearing_faults(
                signal=self._signal(), fs=self.FS, bearing_id="FAKE_999", rpm=0.0
            )

    def test_lookup_and_compute_refuses_the_speed(self):
        with pytest.raises(ValueError, match=RPM_PHRASE):
            lookup_bearing_and_compute(
                signal=self._signal(), fs=self.FS, bearing_type="6205", rpm=np.nan
            )

    def test_tiny_speed_is_refused_where_a_frequency_rounds_to_zero(self):
        """0.5 rpm is positive and finite, but the 6205's FTF rounds to 0.0 Hz:
        the expected-frequency backstop refuses it instead of reading healthy."""
        with pytest.raises(ValueError, match="expected FTF frequency must be positive"):
            check_all_bearing_faults(
                signal=self._signal(), fs=self.FS, bearing_id="6205", rpm=0.5
            )

    @pytest.mark.parametrize("rpm", BAD_SPEEDS, ids=BAD_SPEED_IDS)
    def test_frequency_set_refuses_the_speed_it_echoes(self, rpm):
        with pytest.raises(ValueError, match=RPM_PHRASE):
            check_frequency_set(
                signal=self._signal(), fs=self.FS, frequencies={"GMF": 800.0}, rpm=rpm
            )

    def test_frequency_set_refuses_every_bad_label_in_one_message(self):
        with pytest.raises(ValueError) as excinfo:
            check_frequency_set(
                signal=self._signal(),
                fs=self.FS,
                frequencies={"BPFO": 107.4, "BPFI": np.nan, "BSF": -1.0},
                rpm=1797,
            )

        message = str(excinfo.value)
        assert "non-positive or non-finite" in message
        assert "BPFI" in message
        assert "BSF" in message
        assert "BPFO" not in message.split(":", 1)[1].split("—")[0]

    @pytest.mark.parametrize(
        "frequencies",
        [{"GMF": -10.0}, {"X": np.inf}, {"BPFO": True}, {"BPFO": "107.4"}],
        ids=["negative", "inf", "bool", "string"],
    )
    def test_frequency_set_refuses_unusable_values(self, frequencies):
        with pytest.raises(ValueError, match="non-positive"):
            check_frequency_set(
                signal=self._signal(), fs=self.FS, frequencies=frequencies, rpm=1797
            )

    def test_nan_geometry_frequencies_are_refused_not_healthy(self):
        from predictive_maintenance_mcp.document_reader import (
            calculate_bearing_frequencies,
        )

        freqs = calculate_bearing_frequencies(
            num_balls=9,
            ball_diameter_mm=np.nan,
            pitch_diameter_mm=39.04,
            shaft_speed_rpm=1797,
        )
        with pytest.raises(ValueError, match="non-finite"):
            check_frequency_set(
                signal=self._signal(),
                fs=self.FS,
                frequencies={k: freqs[k] for k in ("BPFO", "BPFI", "BSF", "FTF")},
                rpm=1797,
            )

    @pytest.mark.parametrize("expected", [0.0, -5.0, np.nan, np.inf])
    def test_peak_check_refuses_the_expected_frequency(self, expected):
        with pytest.raises(
            ValueError, match="expected BPFO frequency must be positive"
        ):
            check_bearing_fault_peak(
                signal=self._signal(),
                fs=self.FS,
                expected_freq=expected,
                fault_type="BPFO",
                bearing_id="6205",
            )

    def test_valid_speed_echo_keeps_the_callers_value(self):
        result = check_all_bearing_faults(
            signal=self._signal(), fs=self.FS, bearing_id="6205", rpm=1797
        )

        assert result["rpm"] == 1797
        assert type(result["rpm"]) is int


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
