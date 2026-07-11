"""
Characterization test with real bearing fault data, running against the
module-level MCP tools (importable since the U6 refactor).

NOTE: Sampling rates are read from metadata JSON files.
"""

from pathlib import Path
import json
import numpy as np

from predictive_maintenance_mcp.mcp_tools.analysis_tools import analyze_statistics
from predictive_maintenance_mcp.signal_acquisition.loaders import load_signal_data

# Paths relative to data/signals/
BASELINE_TRAIN = [
    "real_train/baseline_1.csv",
    "real_train/baseline_2.csv"
]

INNER_FAULT_TRAIN = [
    "real_train/InnerRaceFault_vload_1.csv"
]

OUTER_FAULT_TRAIN = [
    "real_train/OuterRaceFault_1.csv"
]


def get_sampling_rate(csv_file):
    """Read sampling rate from metadata JSON"""
    metadata_file = csv_file.replace('.csv', '_metadata.json')
    metadata_path = Path("data/signals") / metadata_file

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            return metadata.get('sampling_rate', 25000.0)
    return 25000.0  # fallback


def test_real_data():
    """Test complete workflow with real data (statistic analysis only - sync)."""
    print("="*70)
    print("MODULE-LEVEL TOOL TEST WITH REAL BEARING DATA")
    print("="*70)
    print("\nDataset: Rolling Element Bearing Fault Diagnosis")
    print("License: CC BY-NC-SA 4.0\n")

    # Test 1: Statistical Analysis (only sync tool)
    print("\n[TEST 1] STATISTICAL ANALYSIS")
    print("-"*70)

    for name, file in [("Baseline", BASELINE_TRAIN[0]),
                       ("Inner Fault", INNER_FAULT_TRAIN[0]),
                       ("Outer Fault", OUTER_FAULT_TRAIN[0])]:
        print(f"\n{name}: {file}")
        result = analyze_statistics(file)
        assert result.rms > 0
        assert result.crest_factor > 0
        print(f"  RMS: {result.rms:.4f}")
        print(f"  Crest Factor: {result.crest_factor:.4f}")
        print(f"  Kurtosis: {result.kurtosis:.4f}")

    # Test 2: Signal loading and FFT computation (non-MCP)
    print("\n\n[TEST 2] FFT SPECTRUM ANALYSIS (direct computation)")
    print("-"*70)

    print(f"\nBaseline: {BASELINE_TRAIN[0]}")
    sr_base = get_sampling_rate(BASELINE_TRAIN[0])
    signal_data = load_signal_data(BASELINE_TRAIN[0])
    assert signal_data is not None, "Failed to load baseline signal"

    fft_result = np.fft.rfft(signal_data)
    frequencies = np.fft.rfftfreq(len(signal_data), 1/sr_base)
    magnitudes = np.abs(fft_result)
    peak_idx = np.argmax(magnitudes[1:]) + 1  # Skip DC
    print(f"  Sampling rate: {sr_base:.0f} Hz")
    print(f"  Peak freq: {frequencies[peak_idx]:.2f} Hz")
    print(f"  Samples: {len(signal_data)}")

    print("\n\n" + "="*70)
    print("ALL TESTS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    test_real_data()
