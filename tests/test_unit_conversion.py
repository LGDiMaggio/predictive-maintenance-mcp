"""
Test unit conversion warning in ISO 20816-3 evaluation.

This test verifies that:
1. Acceleration signals are detected and converted to velocity
2. User receives warning about the conversion
3. Zone description includes conversion notice
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from machinery_diagnostics_server import evaluate_iso_20816

def test_unit_conversion_warning():
    """Test that acceleration→velocity conversion shows warning."""
    
    print("=" * 70)
    print("UNIT CONVERSION TEST")
    print("=" * 70)
    
    # Create a synthetic acceleration signal (in g units)
    # Typical acceleration: RMS ~ 1-5 g
    print("\n[TEST 1] Creating synthetic acceleration signal (RMS ~ 2.5g)...")
    sampling_rate = 10000  # Hz
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Acceleration signal: multiple frequency components
    freq1, freq2 = 50, 120  # Hz
    signal_accel = (
        2.0 * np.sin(2 * np.pi * freq1 * t) +  # 2g @ 50 Hz
        1.5 * np.sin(2 * np.pi * freq2 * t) +  # 1.5g @ 120 Hz
        0.3 * np.random.randn(len(t))          # Noise
    )
    
    rms_accel = np.sqrt(np.mean(signal_accel**2))
    print(f"✓ Signal RMS: {rms_accel:.2f} g (should be > 0.5 to trigger conversion)")
    
    # Save to temporary CSV
    test_file = Path("data/signals/test_accel_signal.csv")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(test_file, signal_accel, delimiter=',')
    print(f"✓ Saved to: {test_file}")
    
    # Evaluate with ISO 20816-3
    print("\n[TEST 2] Evaluating with ISO 20816-3...")
    print("Expected: Warning message about unit conversion")
    print("-" * 70)
    
    result = evaluate_iso_20816(
        signal_file="test_accel_signal.csv",
        sampling_rate=sampling_rate,
        machine_group=2,
        support_type="rigid"
    )
    
    print("-" * 70)
    print(f"\n[RESULTS]")
    print(f"  RMS Velocity: {result.rms_velocity:.2f} mm/s")
    print(f"  Zone: {result.zone} ({result.severity_level})")
    print(f"  Zone Description: {result.zone_description}")
    
    # Verify conversion happened
    if "⚠️ SIGNAL CONVERTED" in result.zone_description:
        print("\n✓✓ PASS: Conversion warning present in zone description")
    else:
        print("\n✗✗ FAIL: Conversion warning NOT found!")
        return False
    
    # Test 3: Velocity signal (should NOT trigger conversion)
    print("\n" + "=" * 70)
    print("[TEST 3] Creating velocity signal (RMS ~ 0.002 mm/s)...")
    signal_velocity = 0.002 * np.sin(2 * np.pi * 50 * t)
    rms_vel = np.sqrt(np.mean(signal_velocity**2))
    print(f"✓ Signal RMS: {rms_vel:.4f} (should be < 0.5, no conversion)")
    
    test_file_vel = Path("data/signals/test_velocity_signal.csv")
    np.savetxt(test_file_vel, signal_velocity, delimiter=',')
    
    result_vel = evaluate_iso_20816(
        signal_file="test_velocity_signal.csv",
        sampling_rate=sampling_rate,
        machine_group=2,
        support_type="rigid"
    )
    
    print(f"\n[RESULTS]")
    print(f"  RMS Velocity: {result_vel.rms_velocity:.4f} mm/s")
    print(f"  Zone: {result_vel.zone} ({result_vel.severity_level})")
    print(f"  Zone Description: {result_vel.zone_description}")
    
    if "⚠️ SIGNAL CONVERTED" not in result_vel.zone_description:
        print("\n✓✓ PASS: No conversion warning (signal already velocity)")
    else:
        print("\n✗✗ FAIL: Unexpected conversion warning!")
        return False
    
    # Cleanup
    print("\n" + "=" * 70)
    print("[CLEANUP] Removing test files...")
    test_file.unlink(missing_ok=True)
    test_file_vel.unlink(missing_ok=True)
    print("✓ Cleanup complete")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = test_unit_conversion_warning()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
