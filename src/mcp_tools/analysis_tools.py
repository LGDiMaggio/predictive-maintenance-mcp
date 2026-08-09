"""MCP tools for signal analysis and spectral processing (ISO 13374 Block 2).

Logging note
------------
Every tool here takes a ``ctx`` parameter it never uses. That is deliberate,
not leftover: ``tests/fixtures/tool_inventory.json`` pins ``context_kwarg``
per tool, so dropping the parameter would be a protocol-visible change to
the tool surface.

What changed in 0.12.0 is *how* progress is emitted, not whether tools
accept a context. SEP-2577 deprecated the MCP logging capability with no
in-protocol replacement, so narration goes to this module's logger, which
``server.configure_logging`` binds to stderr — stdout is the stdio
transport's JSON-RPC channel. Clients no longer receive progress
notifications; any fact a caller needs is carried by the return value.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from mcp.server.mcpserver import MCPServer, Context

from ..signal_acquisition.loaders import extract_segment
from ..models import (
    FFTResult,
    SpectralPeak,
    EnvelopeResult,
    StatisticalResult,
    FeatureExtractionResult,
    PSDResult,
    STFTResult,
)
from ..signal_processing.spectral import (
    compute_psd as _compute_psd,
    compute_stft_spectrogram as _compute_stft,
    compute_envelope_spectrum as _compute_envelope,
)
from ..signal_processing.features import extract_time_domain_features
from ._utils import resolve_signal

logger = logging.getLogger(__name__)


def _select_segment(
    signal_data: np.ndarray,
    segment_duration: Optional[float],
    sampling_rate: float,
    random_seed: Optional[int],
) -> np.ndarray:
    """Select the analysis segment DETERMINISTICALLY by default.

    None -> full signal. Otherwise the LEADING segment_duration seconds,
    so two identical calls analyze identical samples (reproducibility
    invariant). Pass random_seed to sample a seeded random segment
    position instead — still reproducible for the same seed.
    """
    if segment_duration is None:
        return signal_data
    n = int(segment_duration * sampling_rate)
    if n >= len(signal_data):
        return signal_data
    if random_seed is None:
        return signal_data[:n]
    return extract_segment(
        signal_data, segment_duration, sampling_rate, seed=random_seed
    )


# ================================================================
# TOOLS - FFT ANALYSIS
# ================================================================


async def analyze_fft(
    ctx: Context,
    signal_id: str,
    max_frequency: Optional[float] = None,
    segment_duration: Optional[float] = 1.0,
    random_seed: Optional[int] = None,
) -> FFTResult:
    """
    Perform FFT (Fast Fourier Transform) analysis on a stored signal.

    FFT analysis converts the signal from time domain to frequency domain,
    allowing identification of harmonic components and faults that manifest
    at specific frequencies. Requires the signal loaded via load_signal()
    first; the sampling rate comes from the stored signal metadata.

    By default analyzes the LEADING 1.0-second segment (deterministic:
    two identical calls return identical results). Set
    segment_duration=None to analyze the entire signal, or pass
    random_seed to sample a seeded random segment position instead.

    Args:
        ctx: MCP context. Unused — see this module's docstring on logging.
        signal_id: ID of the stored signal (from load_signal).
        max_frequency: Maximum frequency to analyze (default: Nyquist frequency)
        segment_duration: Duration in seconds to analyze (default: leading
            1.0 s). Set to None to analyze the full signal.
        random_seed: Seed for random segment position (default: None =
            deterministic leading segment).

    Returns:
        FFTResult with top peaks, dominant peak, and spectrum stats.

    Raises:
        ValueError: If the signal_id is not loaded, or the stored signal
            has no sampling rate.
    """
    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate

    full_signal_length = len(signal_data)
    signal_duration_sec = full_signal_length / sampling_rate

    signal_data = _select_segment(
        signal_data, segment_duration, sampling_rate, random_seed
    )
    if len(signal_data) < full_signal_length:
        logger.info(
            f"Analyzing {segment_duration}s segment from "
            f"{signal_duration_sec:.1f}s signal"
        )
    else:
        logger.info(
            f"Analyzing full signal ({signal_duration_sec:.1f}s, "
            f"{full_signal_length} samples)"
        )

    # Number of samples
    N = len(signal_data)

    # Apply Hamming window to reduce spectral leakage
    window = np.hamming(N)
    signal_windowed = signal_data * window

    # Calculate FFT
    fft_values = fft(signal_windowed)
    frequencies = fftfreq(N, 1 / sampling_rate)

    # Take only positive frequencies (excluding DC component at index 0)
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]

    # Correct normalization for single-sided spectrum:
    # - Multiply by 2 (energy from negative frequencies)
    # - Divide by N (FFT normalization)
    # Note: DC component (freq=0) should not be multiplied by 2, but we exclude it with frequencies > 0
    magnitudes = 2.0 * np.abs(fft_values[positive_freq_idx]) / N

    # Apply maximum frequency limit if specified
    if max_frequency is not None:
        freq_limit_idx = frequencies <= max_frequency
        frequencies = frequencies[freq_limit_idx]
        magnitudes = magnitudes[freq_limit_idx]

    # Find dominant peak
    peak_idx = np.argmax(magnitudes)
    peak_frequency = float(frequencies[peak_idx])
    peak_magnitude = float(magnitudes[peak_idx])

    # Calculate frequency resolution
    frequency_resolution = sampling_rate / N

    # Build compact summary: top N peaks + stats (no full arrays)
    top_n = 20
    max_mag = float(np.max(magnitudes)) if len(magnitudes) > 0 else 1e-12
    order = np.argsort(magnitudes)[::-1]
    n_peaks = min(top_n, len(order))
    top_idx = np.sort(order[:n_peaks])  # sort back by frequency

    top_peaks = []
    for i in top_idx:
        mag_val = float(magnitudes[i])
        mag_db = float(20 * np.log10(max(mag_val, 1e-12) / max(max_mag, 1e-12)))
        top_peaks.append(
            SpectralPeak(
                frequency_hz=round(float(frequencies[i]), 3),
                magnitude=round(mag_val, 6),
                magnitude_db=round(mag_db, 2),
            )
        )

    rms_spectral = float(np.sqrt(np.mean(magnitudes**2)))

    return FFTResult(
        top_peaks=top_peaks,
        peak_frequency=peak_frequency,
        peak_magnitude=peak_magnitude,
        rms_spectral=round(rms_spectral, 6),
        total_bins=len(frequencies),
        freq_range_hz=(
            [round(float(frequencies[0]), 3), round(float(frequencies[-1]), 3)]
            if len(frequencies) > 0
            else [0, 0]
        ),
        sampling_rate=sampling_rate,
        num_samples=N,
        frequency_resolution=frequency_resolution,
    )


# ================================================================
# TOOLS - ENVELOPE ANALYSIS
# ================================================================


async def analyze_envelope(
    ctx: Context,
    signal_id: str,
    filter_low: float = 500.0,
    filter_high: float = 5000.0,
    num_peaks: int = 5,
    segment_duration: Optional[float] = 1.0,
    random_seed: Optional[int] = None,
) -> EnvelopeResult:
    """
    Envelope-spectrum analysis of a stored signal (bearing fault screening).

    THE unified envelope tool: bandpass filter -> Hilbert
    envelope -> mean subtraction + Hann window -> FFT -> top peaks.
    The mean subtraction/window step is an intentional U9 fix: the
    envelope's DC leakage used to bury the low-frequency FTF zone.
    Requires the signal loaded via load_signal() first; the sampling
    rate comes from the stored signal metadata.

    The requested band must fit the signal: an invalid band (low <= 0,
    low >= high, high > Nyquist) raises a ValueError — it is NEVER
    silently clamped. The band used is echoed in the result.

    By default analyzes the LEADING 1.0-second segment (deterministic:
    two identical calls return identical results). Set
    segment_duration=None to analyze the entire signal, or pass
    random_seed to sample a seeded random segment position instead.

    No reference bearing frequencies are assumed: compare the returned
    peaks against frequencies computed for the actual bearing and
    shaft speed (check_bearing_faults or
    calculate_bearing_characteristic_frequencies).

    Args:
        ctx: MCP context. Unused — see this module's docstring on logging.
        signal_id: ID of the stored signal (from load_signal).
        filter_low: Bandpass low edge in Hz (default: 500).
        filter_high: Bandpass high edge in Hz (default: 5000). Must
            not exceed the signal's Nyquist frequency.
        num_peaks: Number of top peaks to return (default: 5).
        segment_duration: Duration in seconds to analyze (default:
            leading 1.0 s). None analyzes the full signal.
        random_seed: Seed for random segment position (default: None =
            deterministic leading segment).

    Returns:
        EnvelopeResult with the band actually used, top peaks, and
        comparison guidance.

    Raises:
        ValueError: If the signal_id is not loaded, the stored signal
            has no sampling rate, or the band is invalid vs Nyquist.
    """
    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate

    full_signal_length = len(signal_data)
    signal_duration_sec = full_signal_length / sampling_rate

    signal_data = _select_segment(
        signal_data, segment_duration, sampling_rate, random_seed
    )
    if len(signal_data) < full_signal_length:
        logger.info(
            f"Analyzing {segment_duration}s segment from "
            f"{signal_duration_sec:.1f}s signal"
        )
    else:
        logger.info(
            f"Analyzing full signal ({signal_duration_sec:.1f}s, "
            f"{full_signal_length} samples)"
        )

    # Single envelope engine: band validation (raise, never clamp),
    # Hilbert demodulation, detrend + Hann window, FFT, peak picking.
    result = _compute_envelope(
        signal_data,
        sampling_rate,
        frequency_range=(filter_low, filter_high),
        num_peaks=num_peaks,
    )
    top_peaks = [SpectralPeak(**p) for p in result["top_peaks"]]

    # Diagnosis text: peaks + comparison guidance (no invented references)
    diagnosis_lines = [
        "Envelope Analysis Results:",
        f"Filter band: {filter_low:g}-{filter_high:g} Hz",
        "",
        f"Top {len(top_peaks)} peaks in envelope spectrum:",
    ]
    for i, p in enumerate(top_peaks, 1):
        diagnosis_lines.append(
            f"  {i}. {p.frequency_hz:7.2f} Hz  (magnitude: {p.magnitude:.2e})"
        )
    diagnosis_lines.extend(
        [
            "",
            "No reference bearing frequencies are assumed for this machine.",
            "Compare the peaks above against BPFO/BPFI/BSF/FTF computed for the",
            "actual bearing and shaft speed: use check_bearing_faults(...) with a",
            "catalog bearing_id, explicit frequencies, or the bearing geometry",
            "from the machine manual.",
            "Use generate_envelope_report(...) for visual analysis and harmonic "
            "identification.",
        ]
    )

    return EnvelopeResult(
        signal_id=signal_id,
        num_samples=result["num_envelope_samples"],
        sampling_rate=sampling_rate,
        filter_band=(filter_low, filter_high),
        top_peaks=top_peaks,
        diagnosis="\n".join(diagnosis_lines),
    )


# ================================================================
# TOOLS - STATISTICAL ANALYSIS
# ================================================================


def analyze_statistics(signal_id: str) -> StatisticalResult:
    """
    Calculate statistical parameters of a stored signal for diagnostics.

    Statistical parameters are key indicators for diagnostics:
    - RMS: Effective value, correlated to signal energy
    - Crest Factor: Indicates presence of impulses (high = possible faults)
    - Kurtosis: Measures impulsiveness (excess kurtosis; >0 = non-Gaussian, >3 = strong impulses)
    - Peak-to-Peak: Signal range

    Requires the signal loaded via load_signal() first. Statistical
    parameters are screening indicators, not definitive diagnostics —
    combine with frequency-domain evidence.

    **Signal units:** all values are in the signal's native unit. The unit
    is reported only when DECLARED — load_signal(signal_unit=...) or the
    companion _metadata.json — and never guessed from signal amplitude.
    ISO 20816-3 severity tools refuse to produce a verdict until the unit
    is declared.

    Args:
        signal_id: ID of the stored signal (from load_signal).

    Returns:
        StatisticalResult with all statistical parameters

    Raises:
        ValueError: If the signal_id is not loaded.
    """
    signal_data, info = resolve_signal(signal_id, require_sampling_rate=False)

    # Calculate statistical parameters
    rms = float(np.sqrt(np.mean(signal_data**2)))
    peak = float(np.max(np.abs(signal_data)))
    peak_to_peak = float(np.max(signal_data) - np.min(signal_data))
    mean_val = float(np.mean(signal_data))
    std_dev = float(np.std(signal_data))

    # Crest Factor
    crest_factor = peak / rms if rms > 0 else 0.0

    # Kurtosis (using scipy)
    kurtosis_val = float(
        kurtosis(signal_data, fisher=True)
    )  # Fisher=True for excess kurtosis
    skewness_val = float(skew(signal_data))

    # Signal unit: DECLARED only (load_signal parameter or companion
    # metadata, already normalized by the repository) — never guessed
    # from amplitude.
    declared_unit = info.signal_unit

    if declared_unit is not None:
        unit_note = (
            f"Signal unit declared as '{declared_unit}' for "
            f"'{signal_id}'. All values above are in this unit."
        )
    else:
        unit_note = (
            "Signal unit NOT declared — values are in the signal's native "
            "(unknown) unit; the unit is never guessed from amplitude. For "
            "ISO 20816-3 severity assessment, declare it via "
            "load_signal(filepath=..., signal_unit='g'|'m/s2'|'mm/s'|'m/s') "
            "or add a 'signal_unit' field to the companion _metadata.json."
        )

    return StatisticalResult(
        rms=rms,
        peak_to_peak=peak_to_peak,
        peak=peak,
        crest_factor=crest_factor,
        kurtosis=kurtosis_val,
        skewness=skewness_val,
        mean=mean_val,
        std_dev=std_dev,
        signal_unit=declared_unit,
        unit_note=unit_note,
    )


# ================================================================
# TOOLS - FEATURE EXTRACTION
# ================================================================


async def extract_features_from_signal(
    signal_id: str,
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    ctx: Context = None,
) -> FeatureExtractionResult:
    """
    Extract time-domain features from a stored signal using sliding windows.

    Segments the signal into overlapping windows and extracts 17 statistical features
    from each segment. Features include: mean, std, RMS, kurtosis, crest factor, entropy, etc.
    Requires the signal loaded via load_signal() first; the sampling rate
    comes from the stored signal metadata. Returns an in-memory summary
    only — no CSV is written to data/signals/.

    Args:
        signal_id: ID of the stored signal (from load_signal).
        segment_duration: Duration of each segment in seconds (default: 0.1)
        overlap_ratio: Overlap between segments, 0-1 (default: 0.5 = 50%)
        ctx: MCP context. Unused — see this module's docstring on logging.

    Returns:
        FeatureExtractionResult with features matrix and metadata

    Raises:
        ValueError: If the signal_id is not loaded, or the stored signal
            has no sampling rate.

    Example:
        extract_features_from_signal(
            "healthy_motor",
            segment_duration=0.2,
            overlap_ratio=0.5
        )
    """
    logger.info(f"Extracting features from '{signal_id}'...")

    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate
    logger.info(f"Using sampling rate: {sampling_rate} Hz")

    # Calculate segment parameters
    segment_length_samples = int(segment_duration * sampling_rate)
    hop_length = int(segment_length_samples * (1 - overlap_ratio))

    # Extract segments
    segments = []
    num_samples = len(signal_data)

    for start in range(0, num_samples - segment_length_samples + 1, hop_length):
        end = start + segment_length_samples
        segment = signal_data[start:end]
        segments.append(segment)

    logger.info(f"Created {len(segments)} segments from signal")

    # Extract features from each segment
    features_list = []
    for segment in segments:
        features = extract_time_domain_features(segment)
        features_list.append(features)

    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features_list)
    feature_names = list(features_df.columns)

    logger.info(f"Feature matrix shape: {features_df.shape}")

    return FeatureExtractionResult(
        num_segments=len(segments),
        segment_length_samples=segment_length_samples,
        segment_duration_s=segment_duration,
        overlap_ratio=overlap_ratio,
        features_shape=list(features_df.shape),
        feature_names=feature_names,
        features_preview=[features_list[i] for i in range(min(5, len(features_list)))],
    )


# ================================================================
# TOOLS - SPECTRAL ANALYSIS (Phase 1 — signal_id based)
# ================================================================


async def compute_power_spectral_density(
    ctx: Context,
    signal_id: str,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> PSDResult:
    """Compute Power Spectral Density (Welch method) for a stored signal.

    Requires signal loaded via load_signal() first.

    Args:
        signal_id: ID of the stored signal.
        nperseg: Samples per FFT segment (default 256).
        noverlap: Overlap between segments (default 128).
        window: Window function (default 'hann').
    """
    signal_data, info = resolve_signal(signal_id)
    fs = info.sampling_rate

    logger.info(
        f"Computing PSD for '{signal_id}' ({info.num_samples} samples, {fs} Hz)"
    )

    result = _compute_psd(
        signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window
    )

    return PSDResult(
        signal_id=signal_id,
        num_samples=info.num_samples,
        sampling_rate=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        top_peaks=[SpectralPeak(**p) for p in result["top_peaks"]],
        total_power=result["total_power"],
        freq_range_hz=result["freq_range_hz"],
        frequency_resolution=result["frequency_resolution"],
    )


async def compute_spectrogram_stft(
    ctx: Context,
    signal_id: str,
    nperseg: int = 256,
    noverlap: int = 128,
    window: str = "hann",
) -> STFTResult:
    """Compute STFT spectrogram for a stored signal.

    Returns time-frequency summary (no full 2D array). Use for detecting
    time-varying frequency content (transient faults, speed changes).

    Args:
        signal_id: ID of the stored signal.
        nperseg: Samples per STFT segment (default 256).
        noverlap: Overlap between segments (default 128).
        window: Window function (default 'hann').
    """
    signal_data, info = resolve_signal(signal_id)
    fs = info.sampling_rate

    logger.info(f"Computing STFT for '{signal_id}'")

    result = _compute_stft(
        signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window
    )

    return STFTResult(
        signal_id=signal_id,
        num_samples=info.num_samples,
        sampling_rate=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
        num_time_bins=result["num_time_bins"],
        num_freq_bins=result["num_freq_bins"],
        freq_range_hz=result["freq_range_hz"],
        time_range_s=result["time_range_s"],
        max_power_freq_hz=result["max_power_freq_hz"],
        max_power_time_s=result["max_power_time_s"],
        energy_per_band=result["energy_per_band"],
    )


def register(mcp: MCPServer) -> None:
    """Register signal-analysis MCP tools on *mcp*."""
    mcp.tool()(analyze_fft)
    mcp.tool()(analyze_envelope)
    mcp.tool()(analyze_statistics)
    mcp.tool()(extract_features_from_signal)
    mcp.tool()(compute_power_spectral_density)
    mcp.tool()(compute_spectrogram_stft)
