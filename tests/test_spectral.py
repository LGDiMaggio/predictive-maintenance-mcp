"""Tests for spectral analysis functions (PSD, STFT, envelope spectrum)."""

import numpy as np
import pytest

from predictive_maintenance_mcp.spectral import (
    compute_psd,
    compute_stft_spectrogram,
    compute_envelope_spectrum,
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
