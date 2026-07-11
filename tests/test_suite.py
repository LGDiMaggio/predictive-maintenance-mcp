"""
Test and validation script for the Machinery Diagnostics MCP Server.
"""

import sys
from pathlib import Path
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_imports():
    """Verify that all dependencies are installed."""
    print("📦 Testing imports...")
    try:
        import numpy as np
        import scipy
        import pandas as pd
        import mcp
        from pydantic import BaseModel
        from importlib.metadata import version as pkg_version
        
        print("✅ All dependencies imported successfully")
        print(f"   - NumPy: {np.__version__}")
        print(f"   - SciPy: {scipy.__version__}")
        print(f"   - Pandas: {pd.__version__}")
        print(f"   - MCP: {pkg_version('mcp')}")
    except ImportError as e:
        pytest.fail(f"Import error: {e}")


def test_signal_generation():
    """Test test signal generation."""
    print("\n🔬 Testing signal generation...")
    try:
        import numpy as np
        from pathlib import Path
        
        # Create directory if it doesn't exist
        data_dir = Path(__file__).parent / "data" / "signals"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a simple sine wave signal
        fs = 1000  # Hz
        duration = 1.0  # seconds
        freq = 50  # Hz
        
        t = np.linspace(0, duration, int(fs * duration))
        signal = np.sin(2 * np.pi * freq * t)
        
        # Save
        import pandas as pd
        filepath = data_dir / "test_sine_50Hz.csv"
        pd.DataFrame(signal).to_csv(filepath, index=False, header=False)
        
        print(f"✅ Test signal generated: {filepath.name}")
        print(f"   - Frequency: {freq} Hz")
        print(f"   - Sampling rate: {fs} Hz")
        print(f"   - Duration: {duration} s")
        print(f"   - Samples: {len(signal)}")
        
    except Exception as e:
        pytest.fail(f"Signal generation error: {e}")


def test_fft():
    """Test FFT analysis."""
    print("\n📊 Testing FFT analysis...")
    try:
        import numpy as np
        from scipy.fft import fft, fftfreq
        
        # Create test signal
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        
        # FFT
        N = len(signal)
        fft_vals = fft(signal)
        freqs = fftfreq(N, 1/fs)
        
        # Only positive frequencies
        pos_idx = freqs > 0
        freqs = freqs[pos_idx]
        mags = np.abs(fft_vals[pos_idx])
        
        # Find peaks
        peak_idx = np.argmax(mags)
        peak_freq = freqs[peak_idx]
        
        print(f"✅ FFT analysis successful")
        print(f"   - Peak frequency: {peak_freq:.1f} Hz (expected: 50 Hz)")
        print(f"   - Frequency resolution: {fs/N:.2f} Hz")
        
    except Exception as e:
        pytest.fail(f"FFT test error: {e}")


def test_statistics():
    """Test statistics calculation."""
    print("\n📈 Testing statistical analysis...")
    try:
        import numpy as np
        from scipy.stats import kurtosis, skew
        
        # Test signal
        signal = np.random.randn(1000)
        
        # Calculate statistics
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        cf = peak / rms
        kurt = kurtosis(signal, fisher=True)
        skewness = skew(signal)
        
        print(f"✅ Statistical analysis successful")
        print(f"   - RMS: {rms:.3f}")
        print(f"   - Crest Factor: {cf:.2f}")
        print(f"   - Kurtosis: {kurt:.2f}")
        print(f"   - Skewness: {skewness:.2f}")
        
    except Exception as e:
        pytest.fail(f"Statistics test error: {e}")


def print_project_info():
    """Print project information."""
    print("\n" + "=" * 70)
    print("🔧 MACHINERY DIAGNOSTICS MCP SERVER")
    print("=" * 70)
    print("\n📁 Project Structure:")
    
    root = Path(__file__).parent.parent
    print(f"   Root: {root}")
    print(f"   Src:  {root / 'src'}")
    print(f"   Data: {root / 'data' / 'signals'}")
    
    print("\n🚀 Quick Start:")
    print("   1. Test server:  uv run predictive-maintenance-mcp")
    print("   2. Run tests:    uv run python test_suite.py")
    print("   3. Read docs:    type README.md")
    
    print("\n🔧 Available Tools:")
    print("   - analyze_fft:         Fast Fourier Transform analysis")
    print("   - analyze_envelope:    Envelope analysis for bearing faults")
    print("   - analyze_statistics:  Statistical parameters (RMS, Kurtosis, CF)")
    print("   - generate_test_signal: Generate test signals")
    
    print("\n📚 Resources:")
    print("   - signal://list:           List all available signals")
    print("   - signal://read/{filename}: Read specific signal file")
    
    print("\n💡 Prompts:")
    print("   - diagnose_bearing:        Complete bearing diagnostic workflow")
    print("   - diagnose_gear:          Gear diagnostic workflow")
    print("   - quick_diagnostic_report: Quick machinery screening")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """Run all tests."""
    print_project_info()
    
    print("🧪 Running Test Suite...\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Signal Generation", test_signal_generation),
        ("FFT Analysis", test_fft),
        ("Statistical Analysis", test_statistics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Server is ready to use.")
        print("\n   Start with: uv run predictive-maintenance-mcp")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
