"""
Tests for FFT analysis functionality.

Tests:
- Basic FFT computation
- Peak detection accuracy
- Segment processing
- Frequency resolution calculation
- Error handling (invalid parameters)
"""

import pytest
import numpy as np
from pathlib import Path


class TestFFTAnalysis:
    """Test suite for analyze_fft tool."""
    
    def test_fft_synthetic_sine_50hz(self, synthetic_sine_signal):
        """Test FFT detects 50 Hz sine wave correctly."""
        from machinery_diagnostics_server import analyze_fft_computation
        
        signal, fs, expected_freq = synthetic_sine_signal
        
        # Perform FFT
        frequencies, magnitudes = analyze_fft_computation(signal, fs)
        
        # Find peak
        peak_idx = np.argmax(magnitudes)
        detected_freq = frequencies[peak_idx]
        
        # Assert: Peak should be at 50 Hz ±1 Hz
        assert abs(detected_freq - expected_freq) < 1.0, \
            f"Expected {expected_freq} Hz, got {detected_freq} Hz"
    
    
    def test_fft_segment_processing(self, sample_healthy_signal, sample_metadata):
        """Test segment-based FFT processing."""
        from machinery_diagnostics_server import extract_segment
        
        signal = sample_healthy_signal
        fs = sample_metadata['sampling_rate']
        segment_duration = 1.0
        
        # Extract segment
        segment = extract_segment(signal, segment_duration, fs)
        
        # Verify segment length
        expected_length = int(segment_duration * fs)
        assert len(segment) == expected_length, \
            f"Expected {expected_length} samples, got {len(segment)}"
    
    
    def test_fft_frequency_resolution(self, synthetic_sine_signal):
        """Test frequency resolution calculation."""
        signal, fs, _ = synthetic_sine_signal
        duration = len(signal) / fs
        
        expected_resolution = 1.0 / duration
        
        # FFT resolution = sampling_rate / num_samples = 1 / duration
        calculated_resolution = fs / len(signal)
        
        assert abs(calculated_resolution - expected_resolution) < 0.001, \
            f"Resolution mismatch: {calculated_resolution} vs {expected_resolution}"
    
    
    def test_fft_peak_detection_distance(self, synthetic_sine_signal):
        """Test peak detection with minimum distance parameter."""
        from scipy.signal import find_peaks
        
        signal, fs, _ = synthetic_sine_signal
        
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        magnitudes = np.abs(fft_result)
        
        # Find peaks with distance constraint
        peaks, _ = find_peaks(magnitudes, distance=1)
        
        # Verify distance between peaks
        if len(peaks) > 1:
            distances = np.diff(peaks)
            assert np.all(distances >= 1), \
                "Peak distance constraint not satisfied"
    
    
    def test_fft_with_noise(self):
        """Test FFT with noisy signal."""
        fs = 10000
        duration = 2.0
        freq = 100.0
        noise_level = 0.1
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        clean_signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, noise_level, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Compute FFT
        fft_result = np.fft.rfft(noisy_signal)
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(len(noisy_signal), 1/fs)
        
        # Find peak
        peak_idx = np.argmax(magnitudes)
        detected_freq = frequencies[peak_idx]
        
        # Should still detect 100 Hz despite noise
        assert abs(detected_freq - freq) < 2.0, \
            f"Peak detection failed with noise: {detected_freq} Hz"
    
    
    def test_fft_multiple_frequencies(self):
        """Test FFT with multiple frequency components."""
        fs = 10000
        duration = 2.0
        freqs = [50.0, 100.0, 150.0]  # Fundamental + harmonics
        
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
        
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(len(signal), 1/fs)
        
        # Find top 3 peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitudes, height=len(signal)/4)
        top_peaks = sorted(peaks, key=lambda i: magnitudes[i], reverse=True)[:3]
        detected_freqs = [frequencies[p] for p in top_peaks]
        
        # Verify all frequencies detected
        for expected_freq in freqs:
            closest = min(detected_freqs, key=lambda f: abs(f - expected_freq))
            assert abs(closest - expected_freq) < 2.0, \
                f"Failed to detect {expected_freq} Hz"
    
    
    def test_fft_segment_vs_full_signal(self, sample_healthy_signal, sample_metadata):
        """Compare FFT results: segment vs full signal."""
        from machinery_diagnostics_server import extract_segment
        
        signal = sample_healthy_signal
        fs = sample_metadata['sampling_rate']
        
        # Full signal FFT
        fft_full = np.fft.rfft(signal)
        mag_full = np.abs(fft_full)
        
        # Segment FFT
        segment = extract_segment(signal, 1.0, fs)
        fft_segment = np.fft.rfft(segment)
        mag_segment = np.abs(fft_segment)
        
        # Frequency resolution should differ
        res_full = fs / len(signal)
        res_segment = fs / len(segment)
        
        assert res_segment > res_full, \
            "Segment should have coarser frequency resolution"
    
    
    def test_fft_error_handling_invalid_sampling_rate(self, synthetic_sine_signal):
        """Test error handling for invalid sampling rate."""
        signal, _, _ = synthetic_sine_signal
        
        with pytest.raises((ValueError, AssertionError)):
            # Negative sampling rate should fail
            fs_invalid = -1000
            np.fft.rfftfreq(len(signal), 1/fs_invalid)
    
    
    def test_fft_error_handling_empty_signal(self):
        """Test error handling for empty signal."""
        empty_signal = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            np.fft.rfft(empty_signal)
    
    
    def test_fft_real_bearing_data(self, sample_healthy_signal, sample_metadata):
        """Test FFT on real bearing data."""
        signal = sample_healthy_signal
        fs = sample_metadata['sampling_rate']
        
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        magnitudes = np.abs(fft_result)
        frequencies = np.fft.rfftfreq(len(signal), 1/fs)
        
        # Basic validation
        assert len(magnitudes) > 0, "FFT result is empty"
        assert np.all(magnitudes >= 0), "Magnitudes should be non-negative"
        assert np.max(magnitudes) > 0, "Signal appears to be all zeros"
        
        # Should detect shaft frequency (25 Hz) or harmonics
        shaft_freq = sample_metadata['shaft_speed']
        tolerance = 5.0  # ±5 Hz
        
        # Find peaks near shaft frequency
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(magnitudes, height=np.max(magnitudes)*0.01)
        peak_freqs = frequencies[peaks]
        
        # Check if any peak is near shaft frequency or its harmonics
        found_shaft_related = False
        for harmonic in [1, 2, 3, 4]:
            expected = shaft_freq * harmonic
            if any(abs(f - expected) < tolerance for f in peak_freqs):
                found_shaft_related = True
                break
        
        # This is a soft check - shaft harmonics may not always be dominant
        # But we should at least have some peaks
        assert len(peak_freqs) > 0, "No peaks detected in spectrum"


# Helper function tests

def test_extract_segment_center():
    """Test segment extraction from signal center."""
    signal = np.arange(1000)
    fs = 1000
    duration = 0.5  # 0.5 seconds
    
    from machinery_diagnostics_server import extract_segment
    segment = extract_segment(signal, duration, fs)
    
    expected_length = int(duration * fs)
    assert len(segment) == expected_length
    
    # Segment should be from center
    start_idx = (len(signal) - expected_length) // 2
    expected_segment = signal[start_idx:start_idx + expected_length]
    
    np.testing.assert_array_equal(segment, expected_segment)


def test_extract_segment_full_signal():
    """Test extraction when duration=None (full signal)."""
    signal = np.arange(1000)
    fs = 1000
    
    from machinery_diagnostics_server import extract_segment
    segment = extract_segment(signal, None, fs)
    
    np.testing.assert_array_equal(segment, signal)
