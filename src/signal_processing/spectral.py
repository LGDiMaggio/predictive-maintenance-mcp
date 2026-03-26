"""
Pure spectral analysis functions: PSD, STFT, envelope spectrum.

No MCP dependency, no file I/O. Takes numpy arrays, returns dicts.
Suitable for direct testing and reuse across MCP tools and pipelines.
"""

import logging
from typing import Optional

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch, stft, hilbert, butter, sosfiltfilt, find_peaks

logger = logging.getLogger(__name__)


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
    num_peaks: int = 20,
) -> dict:
    """Compute Power Spectral Density using Welch's method.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        nperseg: Samples per FFT segment.
        noverlap: Overlap between segments.
        window: Window function name.
        num_peaks: Number of top peaks to return.

    Returns:
        Dict with keys: top_peaks, total_power, freq_range_hz,
        frequency_resolution, num_freq_bins.
    """
    if nperseg > len(signal):
        nperseg = len(signal)
    if noverlap >= nperseg:
        noverlap = nperseg // 2

    freqs, pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)

    # Total integrated power
    total_power = float(np.trapezoid(pxx, freqs))

    # Peak detection
    pxx_db = 10 * np.log10(np.maximum(pxx / np.max(pxx), 1e-12))
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else fs / nperseg
    min_dist = max(1, int(1.0 / freq_res))

    peak_idx, _ = find_peaks(pxx_db, distance=min_dist, prominence=3)

    if len(peak_idx) == 0:
        peak_idx = np.argsort(pxx)[::-1][:num_peaks]

    # Sort by power and take top N
    sorted_by_power = peak_idx[np.argsort(pxx[peak_idx])[::-1]]
    top_idx = sorted_by_power[:num_peaks]
    top_idx = np.sort(top_idx)  # re-sort by frequency

    max_pxx = float(np.max(pxx)) if np.max(pxx) > 0 else 1e-12
    top_peaks = []
    for i in top_idx:
        mag = float(pxx[i])
        mag_db = float(10 * np.log10(max(mag, 1e-12) / max_pxx))
        top_peaks.append({
            "frequency_hz": round(float(freqs[i]), 3),
            "magnitude": round(mag, 8),
            "magnitude_db": round(mag_db, 2),
            "note": "",
        })

    return {
        "top_peaks": top_peaks,
        "total_power": round(total_power, 6),
        "freq_range_hz": [round(float(freqs[0]), 3), round(float(freqs[-1]), 3)],
        "frequency_resolution": round(float(freq_res), 4),
        "num_freq_bins": len(freqs),
    }


def compute_stft_spectrogram(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> dict:
    """Compute STFT spectrogram and return summary statistics.

    Returns a compact summary (no full 2D array) suitable for LLM consumption.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        nperseg: Samples per STFT segment.
        noverlap: Overlap between segments.
        window: Window function name.

    Returns:
        Dict with summary: time/freq bin counts, max power location,
        energy per frequency band.
    """
    if nperseg > len(signal):
        nperseg = len(signal)
    if noverlap >= nperseg:
        noverlap = nperseg // 2

    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    power = np.abs(Zxx) ** 2

    # Find location of maximum power
    max_idx = np.unravel_index(np.argmax(power), power.shape)
    max_power_freq = float(f[max_idx[0]])
    max_power_time = float(t[max_idx[1]])

    # Energy per frequency band
    bands = [
        ("0-100 Hz", 0, 100),
        ("100-500 Hz", 100, 500),
        ("500-2000 Hz", 500, 2000),
        ("2000-5000 Hz", 2000, 5000),
        ("5000+ Hz", 5000, fs / 2),
    ]
    energy_per_band = []
    for label, lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        if np.any(mask):
            band_energy = float(np.sum(power[mask, :]))
            energy_per_band.append({"band": label, "energy": round(band_energy, 6)})

    return {
        "num_time_bins": len(t),
        "num_freq_bins": len(f),
        "freq_range_hz": [round(float(f[0]), 3), round(float(f[-1]), 3)],
        "time_range_s": [round(float(t[0]), 4), round(float(t[-1]), 4)],
        "max_power_freq_hz": round(max_power_freq, 3),
        "max_power_time_s": round(max_power_time, 4),
        "energy_per_band": energy_per_band,
    }


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    frequency_range: tuple[float, float] = (500, 5000),
    method: str = "hilbert",
    num_peaks: int = 20,
) -> dict:
    """Compute envelope spectrum via Hilbert transform.

    Steps: bandpass filter -> Hilbert demodulation -> FFT of envelope -> peaks.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        frequency_range: Bandpass filter range (low, high) in Hz.
        method: Envelope method (currently only 'hilbert').
        num_peaks: Number of top peaks to return.

    Returns:
        Dict with top_peaks, diagnosis text.
    """
    nyquist = fs / 2.0
    low = max(frequency_range[0], 1.0) / nyquist
    high = min(frequency_range[1], nyquist - 1.0) / nyquist

    # Clamp filter range
    low = max(low, 0.001)
    high = min(high, 0.999)
    if low >= high:
        low = 0.01
        high = 0.99

    # Bandpass filter (Butterworth, 4th order, SOS for numerical stability)
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)

    # Hilbert transform -> envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # Envelope spectrum
    N = len(envelope)
    env_fft = fft(envelope)
    env_freqs = fftfreq(N, 1 / fs)

    pos_mask = env_freqs > 0
    env_freqs = env_freqs[pos_mask]
    env_mags = np.abs(env_fft[pos_mask])

    # Peak detection
    max_mag = float(np.max(env_mags)) if len(env_mags) > 0 else 1e-12
    env_mags_db = 20 * np.log10(np.maximum(env_mags / max_mag, 1e-10))

    freq_res = fs / N
    min_dist = max(1, int(1.0 / freq_res))

    peak_idx, _ = find_peaks(env_mags_db, distance=min_dist, prominence=2)

    if len(peak_idx) == 0:
        peak_idx = np.argsort(env_mags)[::-1][:num_peaks]

    sorted_by_mag = peak_idx[np.argsort(env_mags[peak_idx])[::-1]]
    top_idx = sorted_by_mag[:num_peaks]
    top_idx = np.sort(top_idx)

    top_peaks = []
    for i in top_idx:
        mag = float(env_mags[i])
        mag_db = float(20 * np.log10(max(mag, 1e-12) / max_mag))
        top_peaks.append({
            "frequency_hz": round(float(env_freqs[i]), 3),
            "magnitude": round(mag, 6),
            "magnitude_db": round(mag_db, 2),
            "note": "",
        })

    # Diagnosis text
    lines = [
        "Envelope Spectrum Analysis:",
        f"  Bandpass filter: {frequency_range[0]}-{frequency_range[1]} Hz",
        f"  Method: {method}",
        f"  Top {len(top_peaks)} peaks:",
    ]
    for i, p in enumerate(top_peaks[:10], 1):
        lines.append(f"    {i}. {p['frequency_hz']:7.2f} Hz  ({p['magnitude']:.2e})")
    lines.append("")
    lines.append("Compare peaks with bearing fault frequencies for diagnosis.")

    return {
        "top_peaks": top_peaks,
        "diagnosis": "\n".join(lines),
        "num_envelope_samples": N,
    }
