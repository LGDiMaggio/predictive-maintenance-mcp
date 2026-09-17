"""
Pure spectral analysis functions: amplitude spectrum, PSD, STFT, envelope.

No MCP dependency, no file I/O. Takes numpy arrays, returns arrays or dicts.
Suitable for direct testing and reuse across MCP tools, the diagnosis
pipeline and the asset ledger's health snapshot.
"""

import logging
from typing import Any, Optional

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import welch, stft, hilbert, butter, sosfiltfilt, find_peaks

logger = logging.getLogger(__name__)


def select_leading_segment(
    signal: np.ndarray, fs: float, duration_s: Optional[float]
) -> np.ndarray:
    """Return the LEADING *duration_s* seconds of *signal*, deterministically.

    ``None`` returns the whole signal; so does a duration that covers it
    (never padded, never an error). This is the deterministic segment rule
    ``analyze_fft`` applies by default and the asset ledger's health
    snapshot uses for the 1x amplitude: the same samples for the same
    input, always, so two analyses of one file are comparable.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        duration_s: Segment length in seconds, or ``None`` for the whole
            signal.

    Returns:
        A view of the first ``int(duration_s * fs)`` samples, or *signal*
        itself when it is not longer than that.

    Raises:
        ValueError: If *duration_s* is not positive.
    """
    if duration_s is None:
        return signal
    if duration_s <= 0:
        raise ValueError(
            f"duration_s must be positive, got {duration_s!r} — pass the segment "
            f"length in seconds, or None to analyze the whole signal."
        )
    n = int(duration_s * fs)
    if n >= len(signal):
        return signal
    return signal[:n]


def amplitude_spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum: Hamming window, FFT, ``2|X|/N``.

    THE amplitude-spectrum core of the codebase, extracted from the three
    places that used to inline it (``analyze_fft``, the diagnosis pipeline's
    ``fft_summary`` and ``generate_fft_report``) so that the asset ledger's
    1x amplitude, the tool's peaks and the report's spectrum coincide by
    construction. The operation order is exactly the historical one
    (Hamming window, ``scipy.fft.fft``, positive frequencies only,
    ``2|X|/N``), so the numbers did not change;
    ``tests/fixtures/analyze_fft_golden.json`` pins them.

    A sinusoid of amplitude ``A`` sitting exactly on a bin reads about
    ``0.54 x A`` (the Hamming coherent gain): no gain correction is applied,
    deliberately, so every value stays comparable with every FFT amplitude
    the server ever reported.

    Args:
        signal: 1D time-domain signal with at least 3 samples (so that at
            least one positive-frequency bin exists).
        fs: Sampling frequency (Hz), > 0.

    Returns:
        ``(frequencies, magnitudes)``: the positive frequencies in Hz (DC
        excluded) and the single-sided amplitudes in the signal's unit. Full
        arrays: callers returning data to an LLM must summarise first.

    Raises:
        ValueError: If *signal* is not 1-D with at least 3 samples, or *fs*
            is not positive.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size < 3:
        raise ValueError(
            f"amplitude_spectrum needs a 1-D signal of at least 3 samples, got "
            f"shape {signal.shape} — pass the raw waveform or a longer segment."
        )
    if not fs > 0:
        raise ValueError(f"amplitude_spectrum needs fs > 0 Hz, got {fs!r}.")

    N = len(signal)

    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(N)
    signal_windowed = signal * window

    fft_values = fft(signal_windowed)
    frequencies = fftfreq(N, 1 / fs)

    # Positive frequencies only (the DC bin, which must not be doubled, is
    # excluded by the strict inequality).
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]

    # Single-sided normalization: x2 for the energy of the negative
    # frequencies, /N for the FFT scaling.
    magnitudes = 2.0 * np.abs(fft_values[positive_freq_idx]) / N
    return frequencies, magnitudes


def amplitude_near_frequency(
    freqs: np.ndarray, mags: np.ndarray, target_hz: float, tolerance_pct: float
) -> dict[str, Any]:
    """Largest amplitude within ``target_hz`` +/- ``tolerance_pct`` percent.

    Pure helper shared by the asset ledger's health snapshot: the 1x
    amplitude on :func:`amplitude_spectrum` and the envelope amplitude at
    every expected bearing frequency on :func:`envelope_spectrum_arrays`. It
    searches BINS, not detected peaks, so the value exists in every
    spectrum, also when nothing stands out: the emergence of a line over a
    sequence of measurements is trendable from the first snapshot.

    Args:
        freqs: Frequency axis (Hz), any order.
        mags: Amplitudes aligned with *freqs*.
        target_hz: Centre of the search window (Hz), > 0.
        tolerance_pct: Half-width of the window in percent of *target_hz*,
            >= 0.

    Returns:
        Dict with ``target_hz``, ``amplitude`` (the maximum amplitude among
        the bins inside the window), ``frequency_hz`` (where that maximum
        sits), ``tolerance_pct`` and ``bins_searched``. When NO bin falls
        inside the window (frequency resolution coarser than the window, or
        a target beyond the axis) ``amplitude`` is ``0.0`` and
        ``frequency_hz`` is ``None`` with ``bins_searched == 0`` saying why:
        the absence of a bin is reported as such, never raised and never
        replaced by the nearest bin outside the window.

    Raises:
        ValueError: If *target_hz* is not positive, *tolerance_pct* is
            negative, or the two arrays are not 1-D of the same length.
    """
    freqs = np.asarray(freqs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    if freqs.ndim != 1 or freqs.shape != mags.shape:
        raise ValueError(
            f"freqs and mags must be 1-D arrays of the same length, got shapes "
            f"{freqs.shape} and {mags.shape}."
        )
    if not target_hz > 0:
        raise ValueError(f"target_hz must be positive, got {target_hz!r}.")
    if not tolerance_pct >= 0:
        raise ValueError(f"tolerance_pct must be >= 0, got {tolerance_pct!r}.")

    half_width = target_hz * tolerance_pct / 100.0
    inside = np.flatnonzero(np.abs(freqs - target_hz) <= half_width)
    if inside.size == 0:
        return {
            "target_hz": float(target_hz),
            "amplitude": 0.0,
            "frequency_hz": None,
            "tolerance_pct": float(tolerance_pct),
            "bins_searched": 0,
        }
    best = inside[int(np.argmax(mags[inside]))]
    return {
        "target_hz": float(target_hz),
        "amplitude": float(mags[best]),
        "frequency_hz": float(freqs[best]),
        "tolerance_pct": float(tolerance_pct),
        "bins_searched": int(inside.size),
    }


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
        top_peaks.append(
            {
                "frequency_hz": round(float(freqs[i]), 3),
                "magnitude": round(mag, 8),
                "magnitude_db": round(mag_db, 2),
                "note": "",
            }
        )

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


def validate_bandpass_band(filter_low: float, filter_high: float, fs: float) -> None:
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


def resolve_envelope_band(
    fs: float, frequency_range: Optional[tuple[float, float]] = None
) -> tuple[float, float]:
    """Resolve and validate the envelope bandpass band.

    Args:
        fs: Sampling frequency (Hz).
        frequency_range: Explicit band, or ``None`` for the fs-aware default.

    Returns:
        The (low, high) band edges in Hz.

    Raises:
        ValueError: If the band is invalid for this sampling rate.
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
    return band_low, band_high


def envelope_spectrum_arrays(
    signal: np.ndarray,
    fs: float,
    frequency_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the envelope spectrum and return its full arrays.

    Extracted so that anything drawing the envelope spectrum draws the SAME
    one the bearing fault matching consumed. A figure computed by a second,
    subtly different route can show a peak the verdict never saw — and a
    figure that disagrees with the verdict it illustrates is worse than no
    figure at all.

    Args:
        signal: 1D time-domain signal.
        fs: Sampling frequency (Hz).
        frequency_range: Bandpass band (low, high) in Hz. ``None`` resolves
            to the fs-aware default described in
            :func:`compute_envelope_spectrum`.

    Returns:
        Tuple of (positive frequencies, magnitudes) — full arrays, not a
        summary. Callers returning data to an LLM must summarise first.

    Raises:
        ValueError: If the requested or resolved band is invalid for this
            sampling rate.
    """
    nyquist = fs / 2.0
    band_low, band_high = resolve_envelope_band(fs, frequency_range)

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

    env_fft = fft(envelope_ac)
    env_freqs = fftfreq(N, 1 / fs)

    pos_mask = env_freqs > 0
    return env_freqs[pos_mask], np.abs(env_fft[pos_mask])


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
    band_low, band_high = resolve_envelope_band(fs, frequency_range)
    env_freqs, env_mags = envelope_spectrum_arrays(signal, fs, (band_low, band_high))
    N = len(signal)

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
        top_peaks.append(
            {
                "frequency_hz": round(float(env_freqs[i]), 3),
                "magnitude": round(mag, 6),
                "magnitude_db": round(mag_db, 2),
                "note": "",
            }
        )

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
