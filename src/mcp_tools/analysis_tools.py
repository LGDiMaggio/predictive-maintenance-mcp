"""MCP tools for signal analysis and spectral processing (ISO 13374 Block 2)."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, find_peaks, sosfiltfilt
from scipy.stats import kurtosis, skew
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR
from ..signal_acquisition.loaders import extract_segment
from ..models import (
    FFTResult, SpectralPeak, EnvelopeResult, StatisticalResult,
    FeatureExtractionResult, PSDResult, STFTResult, EnvelopeSpectrumResult,
)
from ..signal_processing.spectral import (
    compute_psd as _compute_psd,
    compute_stft_spectrogram as _compute_stft,
    compute_envelope_spectrum as _compute_envelope,
)
from ..signal_processing.features import extract_time_domain_features
from ._utils import resolve_signal, sanitize_filename

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
    random_seed: Optional[int] = None
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

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from a signal_id or filename** (e.g., an id
          containing "OuterRaceFault" does NOT mean an outer race fault exists)
        - Treat ALL signal_ids as opaque identifiers
        - Base analysis ONLY on frequency spectrum data returned by this tool

        Args:
            ctx: MCP context for user communication
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
    frequencies = fftfreq(N, 1/sampling_rate)

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
        top_peaks.append(SpectralPeak(
            frequency_hz=round(float(frequencies[i]), 3),
            magnitude=round(mag_val, 6),
            magnitude_db=round(mag_db, 2)
        ))

    rms_spectral = float(np.sqrt(np.mean(magnitudes**2)))

    return FFTResult(
        top_peaks=top_peaks,
        peak_frequency=peak_frequency,
        peak_magnitude=peak_magnitude,
        rms_spectral=round(rms_spectral, 6),
        total_bins=len(frequencies),
        freq_range_hz=[round(float(frequencies[0]), 3), round(float(frequencies[-1]), 3)] if len(frequencies) > 0 else [0, 0],
        sampling_rate=sampling_rate,
        num_samples=N,
        frequency_resolution=frequency_resolution
    )

# ================================================================
# TOOLS - ENVELOPE ANALYSIS
# ================================================================

async def analyze_envelope(
    ctx: Context,
    signal_id: str,
    filter_low: float = 500.0,
    filter_high: float = 2000.0,
    num_peaks: int = 5,
    segment_duration: Optional[float] = 1.0,
    random_seed: Optional[int] = None
) -> EnvelopeResult:
    """
        Perform Envelope Analysis on a stored signal to detect bearing faults.

        Envelope analysis is particularly effective for detecting faults in ball/roller bearings.
        The signal is bandpass filtered, the envelope is calculated via Hilbert
        transform, and the envelope spectrum is analyzed. Requires the signal
        loaded via load_signal() first; the sampling rate comes from the
        stored signal metadata.

        By default analyzes the LEADING 1.0-second segment (deterministic:
        two identical calls return identical results). Set
        segment_duration=None to analyze the entire signal, or pass
        random_seed to sample a seeded random segment position instead.

        Returns ONLY peak information and diagnosis text (no full arrays) to avoid context overflow.

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from a signal_id or filename**
        - Treat ALL signal_ids as opaque identifiers
        - Base diagnosis ONLY on frequency-domain evidence (peaks matching BPFO/BPFI/BSF/FTF)

        Args:
            ctx: MCP context for user communication
            signal_id: ID of the stored signal (from load_signal).
            filter_low: Low frequency of bandpass filter in Hz (default: 500 Hz)
            filter_high: High frequency of bandpass filter in Hz (default: 2000 Hz)
            num_peaks: Number of main peaks to identify (default: 5)
            segment_duration: Duration in seconds to analyze (default: leading
                1.0 s). Set to None to analyze the full signal.
            random_seed: Seed for random segment position (default: None =
                deterministic leading segment).

        Returns:
            EnvelopeResult with peak information and diagnosis (optimized for chat display)

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

    # Design Butterworth bandpass filter using SOS (numerically stable)
    nyquist = sampling_rate / 2
    low = filter_low / nyquist
    high = filter_high / nyquist

    # Clamp to valid range (0, 1) and ensure low < high
    low = max(low, 0.01)
    high = min(high, 0.99)
    if low >= high:
        low = 0.01
        high = 0.99

    sos = butter(4, [low, high], btype='band', output='sos')

    # Apply filter
    filtered_signal = sosfiltfilt(sos, signal_data)

    # Calculate envelope using Hilbert transform
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)

    # Calculate envelope spectrum
    N = len(envelope)
    envelope_fft = fft(envelope)
    envelope_frequencies = fftfreq(N, 1/sampling_rate)

    # Take only positive frequencies
    positive_idx = envelope_frequencies > 0
    envelope_frequencies = envelope_frequencies[positive_idx]
    envelope_magnitudes = np.abs(envelope_fft[positive_idx])

    # Find main peaks using scipy.signal.find_peaks (same method as HTML reports)
    # Convert to dB for prominence calculation
    max_magnitude = np.max(envelope_magnitudes)
    envelope_magnitudes_db = 20 * np.log10(np.maximum(envelope_magnitudes / max_magnitude, 1e-10))

    # Find peaks with minimum prominence (at least 2 dB above surroundings)
    # and minimum distance (avoid adjacent FFT bins of same peak)
    freq_resolution = sampling_rate / N
    min_distance_samples = max(1, int(1.0 / freq_resolution))  # At least 1 Hz spacing

    peak_indices, _ = find_peaks(
        envelope_magnitudes_db,
        distance=min_distance_samples,
        prominence=2  # At least 2 dB prominence
    )

    # Sort by magnitude and keep top num_peaks
    if len(peak_indices) > num_peaks:
        sorted_indices = np.argsort(envelope_magnitudes[peak_indices])[::-1]
        peak_indices = peak_indices[sorted_indices[:num_peaks]]
    elif len(peak_indices) == 0:
        # Fallback: if no peaks found with find_peaks, use simple sorting
        peak_indices = np.argsort(envelope_magnitudes)[-num_peaks:][::-1]

    peak_frequencies = envelope_frequencies[peak_indices].tolist()
    peak_magnitudes = envelope_magnitudes[peak_indices].tolist()

    # Create diagnosis text
    diagnosis_lines = [
        f"Envelope Analysis Results:",
        f"Filter band: {filter_low}-{filter_high} Hz",
        f"",
        f"Top {num_peaks} peaks in envelope spectrum:"
    ]

    for i, (freq, mag) in enumerate(zip(peak_frequencies, peak_magnitudes), 1):
        diagnosis_lines.append(f"  {i}. {freq:7.2f} Hz  (magnitude: {mag:.2e})")

    diagnosis_lines.extend([
        "",
        "No reference bearing frequencies are assumed for this machine.",
        "Compare the peaks above against BPFO/BPFI/BSF/FTF computed for the",
        "actual bearing and shaft speed: use search_bearing_catalog(...) for a",
        "verified catalog entry, or calculate_bearing_characteristic_frequencies(...)",
        "with the bearing geometry from the machine manual.",
        "💡 Use plot_envelope(...) for visual analysis and harmonic identification."
    ])

    diagnosis = "\n".join(diagnosis_lines)

    # Small preview (first 100 points for hint/context)
    preview_size = min(100, len(envelope_frequencies))

    return EnvelopeResult(
        num_samples=len(envelope),
        sampling_rate=sampling_rate,
        filter_band=(filter_low, filter_high),
        peak_frequencies=peak_frequencies,
        peak_magnitudes=peak_magnitudes,
        diagnosis=diagnosis,
        spectrum_preview_freq=envelope_frequencies[:preview_size].tolist(),
        spectrum_preview_mag=envelope_magnitudes[:preview_size].tolist()
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

        Requires the signal loaded via load_signal() first.

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from a signal_id or filename**
        - Treat ALL signal_ids as opaque identifiers
        - Statistical parameters (RMS/CF/Kurtosis) are indicators ONLY - NOT definitive diagnostics
        - High kurtosis indicates "possible fault" - NOT "confirmed fault"
        - Must be combined with frequency-domain evidence for diagnosis

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
    kurtosis_val = float(kurtosis(signal_data, fisher=True))  # Fisher=True for excess kurtosis
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
        unit_note=unit_note
    )

# ================================================================
# TOOLS - FEATURE EXTRACTION
# ================================================================

async def extract_features_from_signal(
    signal_id: str,
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    ctx: Context = None
) -> FeatureExtractionResult:
    """
        Extract time-domain features from a stored signal using sliding windows.

        Segments the signal into overlapping windows and extracts 17 statistical features
        from each segment. Features include: mean, std, RMS, kurtosis, crest factor, entropy, etc.
        Requires the signal loaded via load_signal() first; the sampling rate
        comes from the stored signal metadata.

        Args:
            signal_id: ID of the stored signal (from load_signal).
            segment_duration: Duration of each segment in seconds (default: 0.1)
            overlap_ratio: Overlap between segments, 0-1 (default: 0.5 = 50%)
            ctx: MCP context for progress/logging

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
    if ctx:
        await ctx.info(f"Extracting features from '{signal_id}'...")

    signal_data, info = resolve_signal(signal_id)
    sampling_rate = info.sampling_rate
    if ctx:
        await ctx.info(f"Using sampling rate: {sampling_rate} Hz")

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

    if ctx:
        await ctx.info(f"Created {len(segments)} segments from signal")

    # Extract features from each segment
    features_list = []
    for segment in segments:
        features = extract_time_domain_features(segment)
        features_list.append(features)

    # Convert to DataFrame for easier handling
    features_df = pd.DataFrame(features_list)
    feature_names = list(features_df.columns)

    # Save features to file (sanitize to prevent path traversal)
    features_file = DATA_DIR / f"features_{sanitize_filename(signal_id)}.csv"
    features_df.to_csv(features_file, index=False)

    if ctx:
        await ctx.info(f"Features saved to {features_file.name}")
        await ctx.info(f"Feature matrix shape: {features_df.shape}")

    return FeatureExtractionResult(
        num_segments=len(segments),
        segment_length_samples=segment_length_samples,
        segment_duration_s=segment_duration,
        overlap_ratio=overlap_ratio,
        features_shape=list(features_df.shape),
        feature_names=feature_names,
        features_preview=[features_list[i] for i in range(min(5, len(features_list)))]
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

    if ctx:
        await ctx.info(f"Computing PSD for '{signal_id}' ({info.num_samples} samples, {fs} Hz)")

    result = _compute_psd(signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window)

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

    if ctx:
        await ctx.info(f"Computing STFT for '{signal_id}'")

    result = _compute_stft(signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window)

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

async def compute_envelope_spectrum_tool(
    ctx: Context,
    signal_id: str,
    filter_low: float = 500.0,
    filter_high: float = 5000.0,
    method: str = "hilbert",
) -> EnvelopeSpectrumResult:
    """Compute envelope spectrum for a stored signal (signal_id pattern).

        Use for bearing fault detection. The envelope spectrum reveals
        modulation patterns caused by bearing defects.

        Args:
            signal_id: ID of the stored signal.
            filter_low: Bandpass filter low frequency (Hz).
            filter_high: Bandpass filter high frequency (Hz).
            method: Envelope method (default 'hilbert').
        """
    signal_data, info = resolve_signal(signal_id)
    fs = info.sampling_rate

    if ctx:
        await ctx.info(f"Computing envelope spectrum for '{signal_id}' ({filter_low}-{filter_high} Hz)")

    result = _compute_envelope(
        signal_data, fs,
        frequency_range=(filter_low, filter_high),
        method=method,
    )

    return EnvelopeSpectrumResult(
        signal_id=signal_id,
        num_samples=result["num_envelope_samples"],
        sampling_rate=fs,
        method=method,
        frequency_range=(filter_low, filter_high),
        top_peaks=[SpectralPeak(**p) for p in result["top_peaks"]],
        diagnosis=result["diagnosis"],
    )


def register(mcp: FastMCP) -> None:
    """Register signal-analysis MCP tools on *mcp*."""
    mcp.tool()(analyze_fft)
    mcp.tool()(analyze_envelope)
    mcp.tool()(analyze_statistics)
    mcp.tool()(extract_features_from_signal)
    mcp.tool()(compute_power_spectral_density)
    mcp.tool()(compute_spectrogram_stft)
    mcp.tool()(compute_envelope_spectrum_tool)
