"""Tests for spectral analysis functions (amplitude spectrum, PSD, STFT, envelope)."""

import numpy as np
import pytest
from scipy.fft import fft, fftfreq

from predictive_maintenance_mcp.signal_processing.spectral import (
    amplitude_near_frequency,
    amplitude_spectrum,
    compute_psd,
    compute_stft_spectrogram,
    compute_envelope_spectrum,
    select_leading_segment,
    validate_bandpass_band,
)


@pytest.fixture
def sine_signal():
    """50 Hz sine wave at 10 kHz sampling, 1 second."""
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t)
    return signal, fs


@pytest.fixture
def multi_sine_signal():
    """Signal with 50 Hz + 150 Hz components."""
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)
    return signal, fs


@pytest.fixture
def bearing_fault_signal():
    """Synthetic signal with amplitude modulation at 81 Hz (simulated BPFO)."""
    fs = 10000
    t = np.linspace(0, 1.0, fs, endpoint=False)
    # Carrier at 2000 Hz, modulated by 81 Hz (BPFO)
    carrier = np.sin(2 * np.pi * 2000 * t)
    modulation = 1.0 + 0.5 * np.sin(2 * np.pi * 81 * t)
    signal = carrier * modulation + 0.1 * np.random.randn(fs)
    return signal, fs


class TestSelectLeadingSegment:
    """The deterministic segment rule shared by analyze_fft and the ledger."""

    def test_none_returns_the_whole_signal(self):
        signal = np.arange(100.0)
        assert select_leading_segment(signal, 1000.0, None) is signal

    def test_duration_covering_the_signal_returns_it_whole(self):
        signal = np.arange(100.0)
        assert select_leading_segment(signal, 100.0, 1.0) is signal
        assert select_leading_segment(signal, 100.0, 5.0) is signal

    def test_leading_samples(self):
        signal = np.arange(1000.0)
        segment = select_leading_segment(signal, 1000.0, 0.25)
        assert len(segment) == 250
        np.testing.assert_array_equal(segment, signal[:250])

    def test_same_samples_on_every_call(self):
        rng = np.random.default_rng(1)
        signal = rng.standard_normal(5000)
        a = select_leading_segment(signal, 1000.0, 0.5)
        b = select_leading_segment(signal, 1000.0, 0.5)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_non_positive_duration_raises(self, duration):
        with pytest.raises(ValueError, match="duration_s must be positive"):
            select_leading_segment(np.arange(10.0), 10.0, duration)


class TestAmplitudeSpectrum:
    """Hamming window, scipy.fft.fft, positive frequencies, 2|X|/N."""

    def test_matches_the_historical_inline_formula_exactly(self, multi_sine_signal):
        """The extraction preserved the operation order: bit-identical to
        the block analyze_fft, the diagnosis pipeline and the FFT report
        used to inline."""
        signal, fs = multi_sine_signal
        N = len(signal)
        expected_values = fft(signal * np.hamming(N))
        expected_freqs = fftfreq(N, 1 / fs)
        positive = expected_freqs > 0
        expected_mags = 2.0 * np.abs(expected_values[positive]) / N

        freqs, mags = amplitude_spectrum(signal, fs)
        np.testing.assert_array_equal(freqs, expected_freqs[positive])
        np.testing.assert_array_equal(mags, expected_mags)

    def test_sine_reads_amplitude_times_hamming_coherent_gain(self):
        """A 2.0-amplitude sine on an exact bin reads 2.0 x mean(hamming):
        no coherent-gain correction, by design (the numbers stay comparable
        with every FFT amplitude the server ever reported)."""
        fs = 10000
        t = np.arange(fs) / fs
        signal = 2.0 * np.sin(2 * np.pi * 25.0 * t)
        freqs, mags = amplitude_spectrum(signal, fs)
        peak = int(np.argmax(mags))
        assert freqs[peak] == pytest.approx(25.0)
        assert mags[peak] == pytest.approx(2.0 * np.mean(np.hamming(fs)), rel=1e-6)

    def test_positive_frequencies_only_and_nyquist(self, sine_signal):
        signal, fs = sine_signal
        freqs, mags = amplitude_spectrum(signal, fs)
        assert len(freqs) == len(mags) == (len(signal) - 1) // 2
        assert freqs[0] > 0
        assert freqs[-1] <= fs / 2
        assert np.all(np.diff(freqs) > 0)

    def test_too_short_signal_raises(self):
        with pytest.raises(ValueError, match="at least 3 samples"):
            amplitude_spectrum(np.array([1.0, 2.0]), 100.0)

    def test_two_dimensional_signal_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            amplitude_spectrum(np.zeros((4, 4)), 100.0)

    @pytest.mark.parametrize("fs", [0.0, -100.0])
    def test_non_positive_fs_raises(self, fs):
        with pytest.raises(ValueError, match="fs > 0"):
            amplitude_spectrum(np.arange(10.0), fs)


class TestAmplitudeNearFrequency:
    """Bin search around a target: max inside the window, explicit absence."""

    def test_returns_the_largest_bin_inside_the_window(self):
        freqs = np.arange(1.0, 101.0)  # 1..100 Hz, 1 Hz bins
        mags = np.zeros_like(freqs)
        mags[48] = 0.5  # 49 Hz
        mags[49] = 0.2  # 50 Hz (the target itself, smaller)
        mags[59] = 9.0  # 60 Hz, outside +/-5 % of 50 Hz
        result = amplitude_near_frequency(freqs, mags, 50.0, 5.0)
        assert result == {
            "target_hz": 50.0,
            "amplitude": 0.5,
            "frequency_hz": 49.0,
            "tolerance_pct": 5.0,
            "bins_searched": 5,  # 48, 49, 50, 51, 52
        }

    def test_window_edges_are_inclusive(self):
        freqs = np.array([95.0, 100.0, 105.0])
        mags = np.array([1.0, 0.0, 2.0])
        result = amplitude_near_frequency(freqs, mags, 100.0, 5.0)
        assert result["bins_searched"] == 3
        assert result["amplitude"] == 2.0
        assert result["frequency_hz"] == 105.0

    def test_no_bin_inside_the_window_is_reported_not_raised(self):
        """Coarse resolution or a target beyond the axis: amplitude 0.0,
        frequency None, bins_searched 0 (never the nearest outside bin)."""
        freqs = np.array([10.0, 20.0, 30.0])
        mags = np.array([1.0, 1.0, 1.0])
        result = amplitude_near_frequency(freqs, mags, 15.0, 5.0)
        assert result == {
            "target_hz": 15.0,
            "amplitude": 0.0,
            "frequency_hz": None,
            "tolerance_pct": 5.0,
            "bins_searched": 0,
        }
        beyond = amplitude_near_frequency(freqs, mags, 500.0, 5.0)
        assert beyond["bins_searched"] == 0 and beyond["frequency_hz"] is None

    def test_zero_tolerance_needs_an_exact_bin(self):
        freqs = np.array([10.0, 20.0])
        mags = np.array([3.0, 4.0])
        assert amplitude_near_frequency(freqs, mags, 20.0, 0.0)["amplitude"] == 4.0
        assert amplitude_near_frequency(freqs, mags, 21.0, 0.0)["amplitude"] == 0.0

    def test_tie_goes_to_the_first_bin(self):
        freqs = np.array([49.0, 50.0, 51.0])
        mags = np.array([1.0, 1.0, 1.0])
        assert amplitude_near_frequency(freqs, mags, 50.0, 5.0)["frequency_hz"] == 49.0

    def test_values_are_plain_python_types(self):
        freqs = np.arange(1.0, 11.0)
        mags = np.linspace(0.1, 1.0, 10)
        result = amplitude_near_frequency(freqs, mags, 5.0, 20.0)
        assert type(result["amplitude"]) is float
        assert type(result["frequency_hz"]) is float
        assert type(result["bins_searched"]) is int

    def test_on_a_real_amplitude_spectrum(self):
        fs = 10000
        t = np.arange(fs) / fs
        signal = 2.0 * np.sin(2 * np.pi * 25.0 * t)
        freqs, mags = amplitude_spectrum(signal, fs)
        hit = amplitude_near_frequency(freqs, mags, 25.0, 5.0)
        miss = amplitude_near_frequency(freqs, mags, 50.0, 5.0)
        assert hit["frequency_hz"] == pytest.approx(25.0)
        assert hit["amplitude"] == pytest.approx(
            2.0 * np.mean(np.hamming(fs)), rel=1e-6
        )
        assert miss["amplitude"] < 0.01

    @pytest.mark.parametrize(
        "target, tolerance, message",
        [
            (0.0, 5.0, "target_hz must be positive"),
            (-10.0, 5.0, "target_hz must be positive"),
            (float("nan"), 5.0, "target_hz must be positive"),
            (10.0, -1.0, "tolerance_pct must be >= 0"),
        ],
    )
    def test_invalid_arguments_raise(self, target, tolerance, message):
        freqs = np.arange(1.0, 11.0)
        with pytest.raises(ValueError, match=message):
            amplitude_near_frequency(freqs, freqs, target, tolerance)

    def test_mismatched_arrays_raise(self):
        with pytest.raises(ValueError, match="same length"):
            amplitude_near_frequency(np.arange(5.0), np.arange(4.0), 2.0, 5.0)


class TestComputePSD:
    def test_psd_detects_dominant_frequency(self, sine_signal):
        signal, fs = sine_signal
        result = compute_psd(signal, fs, nperseg=1024)
        # The top peak should be near 50 Hz
        top_freq = result["top_peaks"][0]["frequency_hz"]
        assert abs(top_freq - 50) < 5, f"Expected ~50 Hz, got {top_freq}"

    def test_psd_total_power_positive(self, sine_signal):
        signal, fs = sine_signal
        result = compute_psd(signal, fs)
        assert result["total_power"] > 0

    def test_psd_freq_range(self, sine_signal):
        signal, fs = sine_signal
        result = compute_psd(signal, fs)
        assert result["freq_range_hz"][0] >= 0
        assert result["freq_range_hz"][1] <= fs / 2

    def test_psd_returns_peaks(self, multi_sine_signal):
        signal, fs = multi_sine_signal
        result = compute_psd(signal, fs, nperseg=1024)
        freqs = [p["frequency_hz"] for p in result["top_peaks"]]
        # Should detect both 50 and 150 Hz
        has_50 = any(abs(f - 50) < 5 for f in freqs)
        has_150 = any(abs(f - 150) < 5 for f in freqs)
        assert has_50, "Should detect 50 Hz"
        assert has_150, "Should detect 150 Hz"

    def test_psd_handles_short_signal(self):
        signal = np.random.randn(100)
        result = compute_psd(signal, fs=1000, nperseg=256)
        assert len(result["top_peaks"]) > 0

    def test_psd_frequency_resolution(self, sine_signal):
        signal, fs = sine_signal
        result = compute_psd(signal, fs, nperseg=1024)
        assert result["frequency_resolution"] > 0


class TestComputeSTFT:
    def test_stft_dimensions(self, sine_signal):
        signal, fs = sine_signal
        result = compute_stft_spectrogram(signal, fs, nperseg=256)
        assert result["num_time_bins"] > 0
        assert result["num_freq_bins"] > 0

    def test_stft_freq_range(self, sine_signal):
        signal, fs = sine_signal
        result = compute_stft_spectrogram(signal, fs)
        assert result["freq_range_hz"][0] >= 0
        assert result["freq_range_hz"][1] <= fs / 2

    def test_stft_max_power_location(self, sine_signal):
        signal, fs = sine_signal
        result = compute_stft_spectrogram(signal, fs, nperseg=256)
        # Max power should be near 50 Hz
        assert abs(result["max_power_freq_hz"] - 50) < 20

    def test_stft_energy_per_band(self, sine_signal):
        signal, fs = sine_signal
        result = compute_stft_spectrogram(signal, fs)
        assert len(result["energy_per_band"]) > 0
        for band in result["energy_per_band"]:
            assert "band" in band
            assert "energy" in band
            assert band["energy"] >= 0

    def test_stft_handles_short_signal(self):
        signal = np.random.randn(100)
        result = compute_stft_spectrogram(signal, fs=1000, nperseg=256)
        assert result["num_time_bins"] > 0


class TestComputeEnvelopeSpectrum:
    def test_envelope_detects_modulation(self, bearing_fault_signal):
        signal, fs = bearing_fault_signal
        result = compute_envelope_spectrum(
            signal, fs, frequency_range=(1000, 4000), num_peaks=10
        )
        # Should detect modulation at ~81 Hz
        freqs = [p["frequency_hz"] for p in result["top_peaks"]]
        has_81 = any(abs(f - 81) < 10 for f in freqs)
        assert has_81, f"Expected peak near 81 Hz, got {freqs[:5]}"

    def test_envelope_returns_peaks(self, sine_signal):
        signal, fs = sine_signal
        result = compute_envelope_spectrum(signal, fs)
        assert len(result["top_peaks"]) > 0

    def test_envelope_diagnosis_text(self, sine_signal):
        signal, fs = sine_signal
        result = compute_envelope_spectrum(signal, fs)
        assert "Envelope Spectrum Analysis" in result["diagnosis"]

    def test_envelope_handles_narrow_band(self, bearing_fault_signal):
        signal, fs = bearing_fault_signal
        result = compute_envelope_spectrum(signal, fs, frequency_range=(1500, 2500))
        assert len(result["top_peaks"]) > 0

    def test_envelope_num_samples(self, sine_signal):
        signal, fs = sine_signal
        result = compute_envelope_spectrum(signal, fs)
        assert result["num_envelope_samples"] == len(signal)


class TestBandValidation:
    """U9 (audit 2.8): invalid bands RAISE — the silent clamp/fallback that
    could quietly analyze a quasi-full band is gone."""

    def test_low_above_high_raises(self, sine_signal):
        signal, fs = sine_signal
        with pytest.raises(ValueError, match="filter_high"):
            compute_envelope_spectrum(signal, fs, frequency_range=(4000, 500))

    def test_high_above_nyquist_raises(self, sine_signal):
        signal, fs = sine_signal  # fs = 10 kHz, Nyquist 5 kHz
        with pytest.raises(ValueError, match="Nyquist"):
            compute_envelope_spectrum(signal, fs, frequency_range=(500, 6000))

    def test_non_positive_low_raises(self, sine_signal):
        signal, fs = sine_signal
        with pytest.raises(ValueError, match="filter_low"):
            compute_envelope_spectrum(signal, fs, frequency_range=(0, 2000))

    def test_band_at_nyquist_is_realizable(self, sine_signal):
        """An upper edge exactly AT Nyquist is allowed (realized 1 Hz
        below — a digital filter corner cannot sit at Nyquist)."""
        signal, fs = sine_signal
        result = compute_envelope_spectrum(signal, fs, frequency_range=(500, fs / 2))
        assert len(result["top_peaks"]) > 0

    def test_validator_direct(self):
        validate_bandpass_band(500, 4000, 10000)  # valid: no raise
        with pytest.raises(ValueError):
            validate_bandpass_band(500, 5001, 10000)

    def test_default_band_fs_aware_low_fs_no_raise(self):
        """The DEFAULT band (frequency_range omitted) must NOT raise on a
        legitimate low-fs signal: at fs=8 kHz (Nyquist 4 kHz) the old fixed
        (500, 5000) default raised because 5000 > 4000, even though
        500-3999 Hz is perfectly analyzable."""
        fs = 8000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = np.sin(2 * np.pi * 1500 * t)  # inside the 500-3999 band
        result = compute_envelope_spectrum(signal, fs)  # default band
        assert len(result["top_peaks"]) > 0

    def test_explicit_band_above_nyquist_still_raises_low_fs(self):
        """An EXPLICIT band above Nyquist is still a hard error (only the
        default is fs-aware; explicit bands are never silently clamped)."""
        fs = 8000  # Nyquist 4000 Hz
        t = np.linspace(0, 1.0, fs, endpoint=False)
        signal = np.sin(2 * np.pi * 1500 * t)
        with pytest.raises(ValueError, match="Nyquist"):
            compute_envelope_spectrum(signal, fs, frequency_range=(500, 5000))

    def test_default_band_matches_explicit_500_5000_at_10khz(self):
        """No-op guarantee: at fs=10000 Hz the fs-aware default resolves to
        the exact same band as the historical explicit (500, 5000), so the
        envelope peaks are byte-identical."""
        fs = 10000
        t = np.linspace(0, 1.0, fs, endpoint=False)
        carrier = np.sin(2 * np.pi * 2000 * t)
        signal = carrier * (1.0 + 0.5 * np.sin(2 * np.pi * 81 * t))
        default = compute_envelope_spectrum(signal, fs)
        explicit = compute_envelope_spectrum(signal, fs, frequency_range=(500, 5000))
        assert default["top_peaks"] == explicit["top_peaks"]


class TestEnvelopeDetrendWindow:
    """U9 INTENTIONAL CHANGE (audit 2.8): envelope mean subtraction + Hann
    window before the FFT. Expected-value tests, not golden — the old
    rectangular-window DC skirt buried the FTF zone."""

    def _am_signal(self, mod_freq, duration, depth=0.1, fs=10000):
        # Non-integer number of periods (duration 0.95 s) so the envelope's
        # DC component leaks across bins under a rectangular window.
        n = int(duration * fs)
        t = np.arange(n) / fs
        carrier = np.sin(2 * np.pi * 3000 * t)
        modulation = 1.0 + depth * np.sin(2 * np.pi * mod_freq * t)
        rng = np.random.default_rng(99)
        return carrier * modulation + 0.01 * rng.standard_normal(n), fs

    def test_ftf_zone_modulation_detected(self):
        """A weak 11 Hz (FTF-zone) modulation on a non-integer-period
        record is now visible: mean subtraction kills the DC skirt that
        used to dominate the low-frequency bins."""
        signal, fs = self._am_signal(mod_freq=11.0, duration=0.95, depth=0.1)
        result = compute_envelope_spectrum(
            signal, fs, frequency_range=(1000, 4000), num_peaks=5
        )
        freqs = [p["frequency_hz"] for p in result["top_peaks"]]
        assert any(abs(f - 11.0) < 2.0 for f in freqs), (
            f"Expected the 11 Hz FTF-zone modulation in the top peaks, " f"got {freqs}"
        )

    def test_unmodulated_carrier_no_low_freq_peaks(self):
        """With a constant envelope there is no genuine low-frequency
        content: the top peaks must not report DC-skirt artifacts."""
        fs = 10000
        n = int(0.95 * fs)  # non-integer periods -> worst case for leakage
        t = np.arange(n) / fs
        signal = np.sin(2 * np.pi * 3000 * t)
        result = compute_envelope_spectrum(
            signal, fs, frequency_range=(1000, 4000), num_peaks=5
        )
        # Any reported peak below 30 Hz must be far below full scale.
        for p in result["top_peaks"]:
            if p["frequency_hz"] < 30.0:
                assert p["magnitude_db"] < -20.0, (
                    f"DC-skirt artifact at {p['frequency_hz']} Hz "
                    f"({p['magnitude_db']} dB)"
                )
