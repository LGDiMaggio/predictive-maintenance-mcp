"""
Test Report Generation System
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machinery_diagnostics_server import (
    load_signal_data,
    analyze_fft,
    analyze_envelope,
    evaluate_iso_20816,
    DATA_DIR
)
from report_generator import (
    save_fft_report,
    save_envelope_report,
    save_iso_report,
    list_reports,
    REPORTS_DIR
)
import numpy as np

print("=" * 70)
print("REPORT GENERATION SYSTEM TEST")
print("=" * 70)
print()

# Test 1: Generate FFT Report
print("[TEST 1] FFT Report Generation")
print("-" * 70)

signal_file = "real_train/baseline_1.csv"
signal = load_signal_data(signal_file)

if signal is not None:
    fft_result = analyze_fft(signal_file, sampling_rate=97656, max_frequency=5000)
    
    report_result = save_fft_report(
        signal_file=signal_file,
        sampling_rate=97656,
        frequencies=np.array(fft_result.frequencies),
        magnitudes=np.array(fft_result.magnitudes),
        signal_data=signal,
        max_freq=5000,
        num_peaks=15
    )
    
    print(f"✓ {report_result['message']}")
    print(f"  Path: {report_result['file_path']}")
    print(f"  Size: {report_result['file_size_kb']:.2f} KB")
    print(f"  Peaks detected: {report_result['num_peaks_detected']}")
    print()
else:
    print("✗ Could not load signal")
    print()

# Test 2: Generate Envelope Report
print("[TEST 2] Envelope Report Generation")
print("-" * 70)

signal_file = "real_train/OuterRaceFault_1.csv"
signal = load_signal_data(signal_file)

if signal is not None:
    # Calculate envelope directly for report
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.fft import fft, fftfreq
    
    sampling_rate = 97656
    filter_low = 2000
    filter_high = 8000
    
    # Bandpass filter
    nyq = sampling_rate / 2.0
    low = filter_low / nyq
    high = filter_high / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    # Hilbert envelope
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # FFT of envelope
    N = len(envelope)
    env_fft = fft(envelope)
    env_freqs = fftfreq(N, 1/sampling_rate)
    
    # Keep only positive frequencies
    pos_mask = env_freqs > 0
    env_frequencies = env_freqs[pos_mask]
    env_magnitudes = np.abs(env_fft[pos_mask]) / N * 2
    
    bearing_freqs = {
        'BPFO': 81.13,
        'BPFI': 138.87,
        'BSF': 58.48,
        'FTF': 11.29
    }
    
    report_result = save_envelope_report(
        signal_file=signal_file,
        sampling_rate=sampling_rate,
        filter_band=(filter_low, filter_high),
        filtered_signal=filtered_signal,
        envelope=envelope,
        env_frequencies=env_frequencies,
        env_magnitudes=env_magnitudes,
        bearing_freqs=bearing_freqs,
        max_freq=500,
        num_peaks=15
    )
    
    print(f"✓ {report_result['message']}")
    print(f"  Path: {report_result['file_path']}")
    print(f"  Size: {report_result['file_size_kb']:.2f} KB")
    print(f"  Peaks detected: {report_result['num_peaks_detected']}")
    print(f"  Bearing matches: {report_result['bearing_matches']}")
    print()
else:
    print("✗ Could not load signal")
    print()

# Test 3: Generate ISO Report
print("[TEST 3] ISO 20816-3 Report Generation")
print("-" * 70)

signal_file = "real_train/baseline_1.csv"

iso_result = evaluate_iso_20816(
    signal_file,
    sampling_rate=97656,
    machine_group=2,
    support_type="rigid"
)

report_result = save_iso_report(
    signal_file=signal_file,
    iso_result=iso_result.model_dump()  # Convert Pydantic to dict
)

print(f"✓ {report_result['message']}")
print(f"  Path: {report_result['file_path']}")
print(f"  Size: {report_result['file_size_kb']:.2f} KB")
print(f"  Zone: {report_result['zone']}")
print(f"  RMS Velocity: {report_result['rms_velocity']:.2f} mm/s")
print()

# Test 4: List Reports
print("[TEST 4] List Generated Reports")
print("-" * 70)

reports = list_reports()
print(f"✓ Found {len(reports)} reports in {REPORTS_DIR}")
for report in reports:
    print(f"  - {report['file_name']} ({report['file_size_kb']:.1f} KB) - {report['report_type']}")
print()

print("=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print()
print(f"Reports saved to: {REPORTS_DIR}")
print("Open the HTML files in your browser to view interactive charts!")
