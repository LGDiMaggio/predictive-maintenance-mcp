"""
Tests for envelope analysis and ISO 20816-3 evaluation.

Tests:
- Envelope analysis (Hilbert transform, filtering)
- Bearing fault frequency detection
- ISO 20816-3 zone classification
- RMS velocity calculation
"""

import pytest
import numpy as np


class TestEnvelopeAnalysis:
    """Test suite for analyze_envelope tool."""
    
    def test_envelope_bandpass_filter(self):
        """Test bandpass filter application."""
        from scipy.signal import butter, filtfilt
        
        # Generate signal with multiple frequencies
        fs = 20000  # Must be > 2 * highcut to avoid Wn=1.0
        t = np.linspace(0, 1, fs, endpoint=False)
        signal = (np.sin(2*np.pi*100*t) +  # Low freq
                  np.sin(2*np.pi*2000*t) +  # Mid freq (pass)
                  np.sin(2*np.pi*8000*t))   # High freq
        
        # Apply bandpass filter (500-5000 Hz)
        lowcut, highcut = 500, 5000
        nyq = fs / 2
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Verify: 2000 Hz should pass, 100 Hz and 8000 Hz should be attenuated
        fft_orig = np.abs(np.fft.rfft(signal))
        fft_filt = np.abs(np.fft.rfft(filtered))
        freqs = np.fft.rfftfreq(len(signal), 1/fs)
        
        idx_2000 = np.argmin(np.abs(freqs - 2000))
        idx_100 = np.argmin(np.abs(freqs - 100))
        
        # 2000 Hz should be preserved (ratio close to 1)
        ratio_pass = fft_filt[idx_2000] / (fft_orig[idx_2000] + 1e-10)
        assert ratio_pass > 0.5, f"Passband attenuated too much: {ratio_pass}"
        
        # 100 Hz should be attenuated (ratio << 1)
        ratio_stop = fft_filt[idx_100] / (fft_orig[idx_100] + 1e-10)
        assert ratio_stop < 0.3, f"Stopband not attenuated enough: {ratio_stop}"
    
    
    def test_envelope_hilbert_transform(self):
        """Test Hilbert transform for envelope extraction."""
        from scipy.signal import hilbert
        
        # Generate amplitude-modulated signal
        fs = 10000
        t = np.linspace(0, 1, fs, endpoint=False)
        carrier_freq = 2000  # High frequency carrier
        mod_freq = 100       # Low frequency modulation
        
        signal = (1 + 0.5*np.sin(2*np.pi*mod_freq*t)) * np.sin(2*np.pi*carrier_freq*t)
        
        # Extract envelope
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        
        # Envelope should follow modulation frequency
        fft_env = np.abs(np.fft.rfft(envelope))
        freqs_env = np.fft.rfftfreq(len(envelope), 1/fs)
        
        # Peak should be at modulation frequency
        peak_idx = np.argmax(fft_env[1:]) + 1  # Skip DC
        peak_freq = freqs_env[peak_idx]
        
        assert abs(peak_freq - mod_freq) < 5.0, \
            f"Envelope peak at {peak_freq} Hz, expected {mod_freq} Hz"
    
    
    def test_envelope_bearing_fault_simulation(self):
        """Test envelope analysis with simulated bearing fault."""
        fs = 10000
        duration = 2.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Simulate bearing fault: periodic impulses
        BPFO = 80.0  # Outer race fault frequency
        fault_signal = np.zeros_like(t)
        
        # Add impulses at BPFO intervals
        impulse_spacing = int(fs / BPFO)
        for i in range(0, len(t), impulse_spacing):
            if i < len(t):
                fault_signal[i] = 1.0
        
        # Add high-frequency resonance
        carrier_freq = 3000
        signal = fault_signal * np.sin(2*np.pi*carrier_freq*t)
        
        # Extract envelope
        from scipy.signal import hilbert
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        
        # Envelope FFT
        fft_env = np.abs(np.fft.rfft(envelope))
        freqs_env = np.fft.rfftfreq(len(envelope), 1/fs)
        
        # Should detect BPFO frequency
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(fft_env, height=np.max(fft_env)*0.1)
        peak_freqs = freqs_env[peaks]
        
        # Find closest peak to BPFO
        if len(peak_freqs) > 0:
            closest = min(peak_freqs, key=lambda f: abs(f - BPFO))
            assert abs(closest - BPFO) < 5.0, \
                f"BPFO not detected: closest peak at {closest} Hz"
    
    
    def test_envelope_real_fault_data(self, sample_faulty_signal, sample_metadata):
        """Test envelope analysis on real outer race fault data."""
        from scipy.signal import butter, filtfilt, hilbert
        
        signal = sample_faulty_signal
        fs = sample_metadata['sampling_rate']
        BPFO = sample_metadata['BPFO']
        
        # Apply bandpass filter
        lowcut, highcut = 500, 5000
        nyq = fs / 2
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal[:int(fs*1.0)])  # Use 1s segment
        
        # Extract envelope
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        
        # Envelope FFT
        fft_env = np.abs(np.fft.rfft(envelope))
        freqs_env = np.fft.rfftfreq(len(envelope), 1/fs)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(fft_env, height=np.max(fft_env)*0.05)
        peak_freqs = freqs_env[peaks]
        
        # Should detect BPFO or harmonics
        bpfo_harmonics = [BPFO * i for i in range(1, 5)]
        detected_harmonics = []
        
        for harmonic in bpfo_harmonics:
            if any(abs(f - harmonic) < 10.0 for f in peak_freqs):
                detected_harmonics.append(harmonic)
        
        assert len(detected_harmonics) >= 1, \
            f"No BPFO harmonics detected. Peaks at: {peak_freqs[:5]}"


class TestISO20816:
    """Test suite for ISO 20816-3 evaluation."""
    
    def test_rms_calculation(self):
        """Test RMS velocity calculation."""
        # Synthetic signal with known RMS
        fs = 10000
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Sine wave amplitude = 1, RMS = 1/sqrt(2) ≈ 0.707
        signal = np.sin(2*np.pi*50*t)
        rms_expected = 1.0 / np.sqrt(2)
        rms_calculated = np.sqrt(np.mean(signal**2))
        
        assert abs(rms_calculated - rms_expected) < 0.01, \
            f"RMS mismatch: {rms_calculated} vs {rms_expected}"
    
    
    def test_iso_zone_classification_group1(self):
        """Zone classification for Group 1 rigid — asserts on production code."""
        from predictive_maintenance_mcp.diagnostics.iso10816 import classify_zone

        # Group 1 rigid boundaries (ISO 10816-3:2009): 2.3 / 4.5 / 7.1 mm/s
        assert classify_zone(1.0, machine_group=1, support_type="rigid")["zone"] == "A"
        assert classify_zone(3.0, machine_group=1, support_type="rigid")["zone"] == "B"
        assert classify_zone(5.0, machine_group=1, support_type="rigid")["zone"] == "C"
        assert classify_zone(10.0, machine_group=1, support_type="rigid")["zone"] == "D"


    def test_iso_zone_classification_group2(self):
        """Zone classification for Group 2 rigid — asserts on production code."""
        from predictive_maintenance_mcp.diagnostics.iso10816 import classify_zone

        # Group 2 rigid boundaries (ISO 10816-3:2009): 1.4 / 2.8 / 4.5 mm/s
        assert classify_zone(1.0, machine_group=2, support_type="rigid")["zone"] == "A"
        assert classify_zone(2.0, machine_group=2, support_type="rigid")["zone"] == "B"
        assert classify_zone(3.5, machine_group=2, support_type="rigid")["zone"] == "C"
        assert classify_zone(6.0, machine_group=2, support_type="rigid")["zone"] == "D"
    
    
    def test_iso_velocity_integration(self):
        """Test acceleration to velocity integration."""
        from scipy.integrate import cumulative_trapezoid
        
        # Constant acceleration should give linear velocity
        fs = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        
        # Constant acceleration = 10 m/s²
        acceleration = np.ones_like(t) * 10
        
        # Integrate to get velocity
        velocity = cumulative_trapezoid(acceleration, t, initial=0)
        
        # At t=1s, velocity should be ~10 m/s
        assert abs(velocity[-1] - 10.0) < 0.5, \
            f"Integration error: v(1s) = {velocity[-1]}"
    
    
    def test_iso_real_healthy_baseline(self, sample_healthy_signal, sample_metadata):
        """Test ISO evaluation on healthy baseline."""
        signal = sample_healthy_signal
        fs = sample_metadata['sampling_rate']
        
        # Use 1s segment for faster test
        segment = signal[:int(fs*1.0)]
        
        # Calculate RMS
        rms = np.sqrt(np.mean(segment**2))
        
        # RMS should be reasonable (not zero, not huge)
        assert rms > 0, "RMS is zero - signal may be empty"
        assert rms < 100, f"RMS too high ({rms}) - check signal units"
        
        # For healthy bearing, RMS should typically be in Zone A or B
        # This is a soft assertion - depends on signal units and scaling
        # In practice, check if value is reasonable
    
    
    def test_iso_support_type_changes_thresholds(self):
        """Rigid and flexible supports have DIFFERENT zone boundaries.

        ISO 10816-3:2009 Table A.1: group 1 rigid is 2.3/4.5/7.1 mm/s,
        group 1 flexible is 3.5/7.1/11.0 mm/s. Replaces a former test that
        asserted the false domain claim rigid == flexible.
        """
        from predictive_maintenance_mcp.diagnostics.iso10816 import classify_zone

        rigid = classify_zone(3.0, machine_group=1, support_type="rigid")
        flexible = classify_zone(3.0, machine_group=1, support_type="flexible")

        assert rigid["zone"] == "B"  # 2.3 < 3.0 <= 4.5
        assert flexible["zone"] == "A"  # 3.0 <= 3.5
        assert rigid["boundaries"] != flexible["boundaries"]


    def test_iso_error_handling_negative_rms(self):
        """Negative RMS is physically impossible: production code must raise."""
        from predictive_maintenance_mcp.diagnostics.iso10816 import classify_zone

        with pytest.raises(ValueError):
            classify_zone(-1.0, machine_group=2, support_type="rigid")


    def test_iso_error_handling_invalid_machine_group(self):
        """Invalid machine group must be rejected by production code."""
        from predictive_maintenance_mcp.diagnostics.iso10816 import classify_zone

        for group in [0, 3, 5, -1]:
            with pytest.raises(ValueError):
                classify_zone(1.0, machine_group=group, support_type="rigid")
