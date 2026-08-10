"""
Performance benchmarks for Phase 1 modules.

Measures against PRD acceptance criteria:
- Signal load time: <100ms for 1GB file
- FFT computation: <50ms for 10-second signal at 10kHz
- Bearing diagnosis: <500ms end-to-end
- Memory usage: <1GB for 100 loaded signals
- PSD, STFT, envelope: measured for reference

Usage:
    python -m benchmarks.bench_phase1
    python -m benchmarks.bench_phase1 --signals large
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predictive_maintenance_mcp.signal_acquisition.repository import SignalRepository
from predictive_maintenance_mcp.signal_processing.spectral import (
    compute_psd,
    compute_stft_spectrogram,
    compute_envelope_spectrum,
)
from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
    check_all_bearing_faults,
)
from predictive_maintenance_mcp.diagnostics.iso20816 import assess_vibration_severity
from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
    diagnose_vibration,
)


def _fmt(ms: float, target: float) -> str:
    status = "PASS" if ms <= target else "FAIL"
    return f"{ms:8.2f} ms  (target: <{target:.0f} ms)  [{status}]"


def bench_signal_repository(signal_sizes: dict[str, int]):
    """Benchmark signal load/get times."""
    print("\n=== Signal Repository ===")
    repo = SignalRepository()
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        for label, n_samples in signal_sizes.items():
            # Create temp file
            path = Path(tmpdir) / f"bench_{label}.npy"
            data = np.random.randn(n_samples).astype(np.float64)
            np.save(path, data)
            file_size_mb = data.nbytes / 1024**2

            # Benchmark load
            start = time.perf_counter()
            repo.load_signal(str(path), signal_id=f"bench_{label}", sampling_rate=10000)
            load_ms = (time.perf_counter() - start) * 1000

            # Benchmark get
            start = time.perf_counter()
            _ = repo.get_signal(f"bench_{label}")
            get_ms = (time.perf_counter() - start) * 1000

            print(f"  {label} ({n_samples:>10,} samples, {file_size_mb:.1f} MB):")
            print(f"    Load: {load_ms:8.2f} ms")
            print(f"    Get:  {get_ms:8.2f} ms")

    # Benchmark 100 signals memory usage
    print(f"\n  100-signal memory test:")
    repo2 = SignalRepository()
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(100):
            path = Path(tmpdir) / f"sig_{i}.npy"
            np.save(path, np.random.randn(10000))  # ~80 KB each
            repo2.load_signal(str(path), signal_id=f"s{i}")
        mem_mb = repo2.current_memory_bytes / 1024**2
        target = 1024  # <1 GB
        status = "PASS" if mem_mb < target else "FAIL"
        print(
            f"    Memory for 100 signals: {mem_mb:.1f} MB  (target: <{target} MB)  [{status}]"
        )
        print(f"    Signals loaded: {repo2.signal_count}")


def bench_fft(signal_sizes: dict[str, int], fs: float = 10000):
    """Benchmark FFT computation."""
    print("\n=== FFT (via diagnosis pipeline) ===")
    from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
        _compute_fft_summary,
    )

    for label, n_samples in signal_sizes.items():
        signal = np.random.randn(n_samples)
        start = time.perf_counter()
        _compute_fft_summary(signal, fs)
        ms = (time.perf_counter() - start) * 1000
        duration_s = n_samples / fs
        print(f"  {label} ({duration_s:.1f}s @ {fs/1000:.0f}kHz): {_fmt(ms, 50)}")


def bench_psd(signal_sizes: dict[str, int], fs: float = 10000):
    """Benchmark PSD (Welch)."""
    print("\n=== PSD (Welch) ===")
    for label, n_samples in signal_sizes.items():
        signal = np.random.randn(n_samples)
        start = time.perf_counter()
        compute_psd(signal, fs)
        ms = (time.perf_counter() - start) * 1000
        print(f"  {label}: {ms:8.2f} ms")


def bench_stft(signal_sizes: dict[str, int], fs: float = 10000):
    """Benchmark STFT."""
    print("\n=== STFT Spectrogram ===")
    for label, n_samples in signal_sizes.items():
        signal = np.random.randn(n_samples)
        start = time.perf_counter()
        compute_stft_spectrogram(signal, fs)
        ms = (time.perf_counter() - start) * 1000
        print(f"  {label}: {ms:8.2f} ms")


def bench_envelope(signal_sizes: dict[str, int], fs: float = 10000):
    """Benchmark envelope spectrum."""
    print("\n=== Envelope Spectrum ===")
    for label, n_samples in signal_sizes.items():
        signal = np.random.randn(n_samples)
        start = time.perf_counter()
        compute_envelope_spectrum(signal, fs)
        ms = (time.perf_counter() - start) * 1000
        print(f"  {label}: {ms:8.2f} ms")


def bench_bearing_diagnosis(fs: float = 10000):
    """Benchmark bearing fault detection end-to-end."""
    print("\n=== Bearing Diagnosis (end-to-end) ===")
    sizes = {
        "1s signal": int(1 * fs),
        "10s signal": int(10 * fs),
        "60s signal": int(60 * fs),
    }
    for label, n in sizes.items():
        signal = np.random.randn(n)
        start = time.perf_counter()
        check_all_bearing_faults(signal, fs, bearing_id="6205", rpm=1797)
        ms = (time.perf_counter() - start) * 1000
        print(f"  {label}: {_fmt(ms, 500)}")


def bench_iso_severity(fs: float = 10000):
    """Benchmark ISO severity assessment."""
    print("\n=== ISO 10816 Severity ===")
    signal = np.random.randn(int(10 * fs))
    start = time.perf_counter()
    assess_vibration_severity(signal, fs, signal_unit="g")
    ms = (time.perf_counter() - start) * 1000
    print(f"  10s signal: {ms:8.2f} ms")


def bench_full_diagnosis(fs: float = 10000):
    """Benchmark full diagnosis pipeline."""
    print("\n=== Full Diagnosis Pipeline ===")
    sizes = {
        "1s signal": int(1 * fs),
        "10s signal": int(10 * fs),
    }
    for label, n in sizes.items():
        signal = np.random.randn(n)
        start = time.perf_counter()
        diagnose_vibration(
            signal,
            fs,
            rpm=1797,
            bearing_id="6205",
            signal_unit="g",
            anomaly_model_name="nonexistent",
        )
        ms = (time.perf_counter() - start) * 1000
        print(f"  {label} (with bearing): {_fmt(ms, 5000)}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 performance benchmarks")
    parser.add_argument(
        "--signals",
        choices=["small", "medium", "large"],
        default="medium",
        help="Signal size preset: small (1s), medium (10s), large (60s)",
    )
    args = parser.parse_args()

    fs = 10000
    presets = {
        "small": {"1s": int(1 * fs)},
        "medium": {"1s": int(1 * fs), "10s": int(10 * fs)},
        "large": {"1s": int(1 * fs), "10s": int(10 * fs), "60s": int(60 * fs)},
    }
    sizes = presets[args.signals]

    print(f"Phase 1 Benchmarks (preset: {args.signals}, fs: {fs} Hz)")
    print("=" * 60)

    bench_signal_repository(sizes)
    bench_fft(sizes, fs)
    bench_psd(sizes, fs)
    bench_stft(sizes, fs)
    bench_envelope(sizes, fs)
    bench_bearing_diagnosis(fs)
    bench_iso_severity(fs)
    bench_full_diagnosis(fs)

    print("\n" + "=" * 60)
    print("Benchmarks complete.")


if __name__ == "__main__":
    main()
