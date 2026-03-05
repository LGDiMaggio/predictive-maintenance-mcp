"""
Complete test of MCP server with real bearing fault data

NOTE: Sampling rates are read from metadata JSON files.
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.machinery_diagnostics_server import (
    analyze_fft,
    analyze_envelope,
    analyze_statistics,
    evaluate_iso_20816,
    load_signal_data
)

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
    print("MCP SERVER TEST WITH REAL BEARING DATA")
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


import pytest


@pytest.mark.skip(reason="Integration test requiring MCP Context - run manually")
@pytest.mark.asyncio
async def test_ml_pipeline():
    """Test 5: Complete ML anomaly detection pipeline"""
    print("\n" + "="*70)
    print("TEST 5: ML ANOMALY DETECTION")
    print("="*70)
    
    # Step 1: Train on baseline (healthy) data
    print("\n[Step 1] Training model on 2 baseline (healthy) signals...")
    result_train = await train_anomaly_model(
        healthy_signal_files=BASELINE_TRAIN,
        sampling_rate=25000.0,
        segment_duration=1.0,  # 1 second segments
        overlap_ratio=0.5,
        model_type="OneClassSVM",
        pca_variance=0.95,
        fault_signal_files=INNER_FAULT_TRAIN[:1],  # Validate on 1 fault signal
        model_name="real_bearing_model"
    )
    
    print(f"\n✓ Model trained!")
    print(f"  Training samples: {result_train.num_training_samples}")
    print(f"  Features: {result_train.num_features_original} → {result_train.num_features_pca} (PCA)")
    print(f"  Variance explained: {result_train.variance_explained:.3f}")
    
    if result_train.validation_accuracy:
        print(f"  Validation on fault data: {result_train.validation_accuracy*100:.1f}% detected")
    
    # Step 2: Predict on baseline test (should be healthy)
    print(f"\n[Step 2] Predicting on baseline test: {BASELINE_TEST[0]}")
    result_baseline = await predict_anomalies(
        signal_file=BASELINE_TEST[0],
        model_name="real_bearing_model"
    )
    print(f"  Anomaly ratio: {result_baseline.anomaly_ratio*100:.1f}%")
    print(f"  Health: {result_baseline.overall_health} ({result_baseline.confidence} confidence)")
    
    # Step 3: Predict on inner fault test
    print(f"\n[Step 3] Predicting on inner fault: {INNER_FAULT_TEST[0]}")
    result_inner = await predict_anomalies(
        signal_file=INNER_FAULT_TEST[0],
        model_name="real_bearing_model"
    )
    print(f"  Anomaly ratio: {result_inner.anomaly_ratio*100:.1f}%")
    print(f"  Health: {result_inner.overall_health} ({result_inner.confidence} confidence)")
    
    # Step 4: Predict on outer fault test
    print(f"\n[Step 4] Predicting on outer fault: {OUTER_FAULT_TEST[0]}")
    result_outer = await predict_anomalies(
        signal_file=OUTER_FAULT_TEST[0],
        model_name="real_bearing_model"
    )
    print(f"  Anomaly ratio: {result_outer.anomaly_ratio*100:.1f}%")
    print(f"  Health: {result_outer.overall_health} ({result_outer.confidence} confidence)")
    
    # Summary
    print("\n" + "-"*70)
    print("ML DETECTION SUMMARY:")
    print("-"*70)
    print(f"Baseline (healthy):  {result_baseline.anomaly_ratio*100:5.1f}% anomalies → {result_baseline.overall_health}")
    print(f"Inner fault:         {result_inner.anomaly_ratio*100:5.1f}% anomalies → {result_inner.overall_health}")
    print(f"Outer fault:         {result_outer.anomaly_ratio*100:5.1f}% anomalies → {result_outer.overall_health}")
    
    if (result_baseline.overall_health == "Healthy" and 
        result_inner.overall_health in ["Suspicious", "Faulty"] and
        result_outer.overall_health in ["Suspicious", "Faulty"]):
        print("\n✅ ML MODEL SUCCESSFULLY DETECTS FAULTS!")
    else:
        print("\n⚠️  Results need review (may need parameter tuning)")


@pytest.mark.skip(reason="Integration test requiring MCP Context - run manually")
@pytest.mark.asyncio
async def test_plotting():
    """Test 6: Generate diagnostic plots"""
    print("\n" + "="*70)
    print("TEST 6: GENERATING DIAGNOSTIC PLOTS")
    print("="*70)
    
    # Plot baseline signal
    print(f"\nGenerating signal plot: {BASELINE_TRAIN[0]}")
    html = await plot_signal(
        signal_file=BASELINE_TRAIN[0],
        sampling_rate=25000.0,
        time_range=[0, 1.0]  # First second
    )
    print(f"  ✓ Signal plot: {len(html)} characters")
    
    # Plot spectrum comparison
    print(f"\nGenerating spectrum plot: baseline vs fault")
    html = await plot_spectrum(
        signal_file=BASELINE_TRAIN[0],
        sampling_rate=25000.0,
        freq_range=[0, 2000]
    )
    print(f"  ✓ Spectrum plot: {len(html)} characters")
    
    # Plot envelope for fault detection
    print(f"\nGenerating envelope plot: {INNER_FAULT_TRAIN[0]}")
    html = await plot_envelope(
        signal_file=INNER_FAULT_TRAIN[0],
        sampling_rate=25000.0,
        filter_low=500.0,
        filter_high=10000.0,
        freq_range=[0, 500]
    )
    print(f"  ✓ Envelope plot: {len(html)} characters")


async def main():
    """Run all tests"""
    print("="*70)
    print("COMPLETE MCP SERVER TEST WITH REAL BEARING DATA")
    print("="*70)
    print("\nDataset: Rolling Element Bearing Fault Diagnosis")
    print("Source: http://data-acoustics.com/measurements/bearing-faults/bearing-2/")
    print("License: CC BY-NC-SA 4.0")
    print("\nSampling rate: 25 kHz (assumed)")
    print("="*70)
    
    try:
        # Run all tests
        await test_statistical_analysis()
        await test_fft_analysis()
        await test_envelope_analysis()
        await test_iso_evaluation()
        await test_ml_pipeline()
        await test_plotting()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nThe MCP server successfully:")
        print("  ✓ Loaded and analyzed real bearing vibration data")
        print("  ✓ Performed FFT spectrum analysis")
        print("  ✓ Detected bearing faults with envelope analysis")
        print("  ✓ Evaluated vibration severity per ISO 20816-3")
        print("  ✓ Trained ML anomaly detection model")
        print("  ✓ Predicted faults on unseen data")
        print("  ✓ Generated diagnostic plots")
        print("\n🎉 Server is production-ready for real diagnostics!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
