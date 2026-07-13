"""Tests for spectral analysis functions (PSD, STFT, envelope spectrum)."""

import numpy as np
import pytest

from predictive_maintenance_mcp.signal_processing.spectral import (
    compute_psd,
    compute_stft_spectrogram,
    compute_envelope_spectrum,
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
        result = compute_envelope_spectrum(
            signal, fs, frequency_range=(1500, 2500)
        )
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
        explicit = compute_envelope_spectrum(
            signal, fs, frequency_range=(500, 5000)
        )
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
            f"Expected the 11 Hz FTF-zone modulation in the top peaks, "
            f"got {freqs}"
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
