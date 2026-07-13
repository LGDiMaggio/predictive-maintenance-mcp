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


def validate_bandpass_band(
    filter_low: float, filter_high: float, fs: float
) -> None:
    """Validate a bandpass band against the signal's Nyquist limit, or raise.

    Single band-validation path for every envelope/bandpass consumer.
    An invalid band is ALWAYS a ValueError — it is never silently clamped
    to a different band than the one requested (audit 2.8: the old clamp
    could quietly fall back to a quasi-full-band analysis).

    Args:
        filter_low: Requested lower band edge (Hz).
        filter_high: Requested upper band edge (Hz).
        fs: Sampling frequency (Hz).

    Raises:
        ValueError: If filter_low <= 0, filter_high <= filter_low, or
            filter_high exceeds the Nyquist frequency fs/2.
    """
    nyquist = fs / 2.0
    if filter_low <= 0:
        raise ValueError(
            f"Invalid bandpass band: filter_low={filter_low:g} Hz must be "
            f"positive — pass a lower edge above 0 Hz (e.g. 500 Hz for "
            f"bearing envelope analysis)."
        )
    if filter_high <= filter_low:
        raise ValueError(
            f"Invalid bandpass band: filter_high={filter_high:g} Hz must be "
            f"greater than filter_low={filter_low:g} Hz — re-specify the "
            f"band edges in (low, high) order."
        )
    if filter_high > nyquist:
        raise ValueError(
            f"Invalid bandpass band: filter_high={filter_high:g} Hz exceeds "
            f"the Nyquist frequency {nyquist:g} Hz of this signal "
            f"(fs={fs:g} Hz) — choose filter_high <= {nyquist:g} Hz or "
            f"re-acquire at a higher sampling rate. The band is never "
            f"clamped silently."
        )


def compute_envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    frequency_range: Optional[tuple[float, float]] = None,
    method: str = "hilbert",
    num_peaks: int = 20,
) -> dict:
    """Compute envelope spectrum via Hilbert transform.

    Steps: bandpass filter -> Hilbert demodulation -> mean subtraction +
    Hann window -> FFT of envelope -> peaks.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        frequency_range: Bandpass filter range (low, high) in Hz. ``None``
            (the default) resolves to an fs-AWARE band — 500 Hz up to
            ``min(5000, just-below-Nyquist)`` — so a legitimate low-fs
            signal (e.g. fs=8 kHz, analyzable over 500-3999 Hz) is analyzed
            instead of raising. An EXPLICIT band is honored verbatim and
            validated as given: invalid bands (low <= 0, low >= high,
            high > Nyquist) raise ValueError — never a silent clamp. At
            fs=10000 Hz the default resolves to the same 500-4999 Hz edge
            used historically, so byte-level outputs are unchanged.
        method: Envelope method (currently only 'hilbert').
        num_peaks: Number of top peaks to return.

    Returns:
        Dict with top_peaks, diagnosis text.

    Raises:
        ValueError: If an explicitly requested band is invalid for this
            sampling rate, or the fs-aware default is unusable (fs so low
            the 500 Hz lower edge meets the clamped upper edge).
    """
    nyquist = fs / 2.0

    if frequency_range is None:
        # fs-AWARE DEFAULT. Cap the upper edge just below Nyquist so the
        # band never exceeds a low-fs signal's Nyquist. This clamp is a
        # NO-OP at fs=10000 Hz (5000 -> nyquist-1 = 4999), the exact edge
        # the historical (500, 5000) default already resolved to.
        band_low, band_high = 500.0, min(5000.0, nyquist - 1.0)
    else:
        # An EXPLICIT band is honored verbatim (fail loud, never clamp).
        band_low, band_high = float(frequency_range[0]), float(frequency_range[1])

    # Single validation path. The explicit band is checked as requested (an
    # upper edge above Nyquist is a hard error); the default is pre-capped,
    # so it only trips when the band is genuinely unusable (low >= high at
    # a very low fs).
    validate_bandpass_band(band_low, band_high, fs)

    low = band_low / nyquist
    # A digital filter corner cannot sit exactly AT Nyquist: an upper edge
    # equal to Nyquist is realized 1 Hz below it (same realization the
    # pre-U9 code used). Edges ABOVE Nyquist were already rejected — this
    # is filter realizability, not a band clamp.
    high = min(band_high, nyquist - 1.0) / nyquist

    # Bandpass filter (Butterworth, 4th order, SOS for numerical stability)
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, signal)

    # Hilbert transform -> envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    # INTENTIONAL CHANGE (U9, audit 2.8): subtract the envelope mean and
    # apply a Hann window BEFORE the FFT. The envelope is strictly
    # positive, so its mean is a large DC component whose leakage skirt
    # (rectangular window) buried exactly the low-frequency FTF zone.
    N = len(envelope)
    envelope_ac = (envelope - np.mean(envelope)) * np.hanning(N)

    # Envelope spectrum
    env_fft = fft(envelope_ac)
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
        f"  Bandpass filter: {band_low:g}-{band_high:g} Hz",
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
