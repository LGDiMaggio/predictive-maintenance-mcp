"""MCP tools for signal analysis and spectral processing (ISO 13374 Block 2)."""

import logging
import json
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, butter, filtfilt, sosfiltfilt
from scipy.stats import entropy
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR, MODELS_DIR
from ..signal_loader import load_signal_data, get_metadata_path, SUPPORTED_EXTENSIONS
from ..signal_repository import get_repository, normalize_signal_unit
from ..models import (
    FFTResult, SpectralPeak, EnvelopeResult, StatisticalResult, SignalInfo,
    FeatureExtractionResult, PSDResult, STFTResult, EnvelopeSpectrumResult,
    StoredSignalInfo
)
from ..spectral import (
    compute_psd as _compute_psd,
    compute_stft_spectrogram as _compute_stft,
    compute_envelope_spectrum as _compute_envelope,
)
from ..signal_processing.features import (
    extract_time_domain_features,
    segment_and_extract_features as _segment_and_extract_features,
    resolve_sampling_rate as _resolve_sampling_rate,
)
from ._utils import sanitize_filename, load_and_validate_metadata as _load_and_validate_metadata

logger = logging.getLogger(__name__)


# ============================================================================
# Tool registration
# ============================================================================


def register(mcp: FastMCP) -> None:
    """Register signal-analysis MCP tools on *mcp*."""

    # -- lazy import; only needed inside tool bodies that call extract_segment
    from ..signal_loader import extract_segment
    from mcp.server.session import ServerSession

    # ================================================================
    # TOOLS - FFT ANALYSIS
    # ================================================================

    @mcp.tool()
    async def analyze_fft(
        ctx: Context,
        filename: str,
        sampling_rate: Optional[float] = None,
        max_frequency: Optional[float] = None,
        segment_duration: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> FFTResult:
        """
        Perform FFT (Fast Fourier Transform) analysis on a signal.

        FFT analysis converts the signal from time domain to frequency domain,
        allowing identification of harmonic components and faults that manifest
        at specific frequencies.

        By default, analyzes a RANDOM 1.0-second segment from the signal for efficiency.
        Set segment_duration=None to analyze the entire signal.

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from filename** (e.g., "OuterRaceFault_1.csv" does NOT mean outer race fault exists)
        - **NEVER assume signal characteristics from filename** (e.g., "baseline" does NOT mean healthy)
        - Treat ALL filenames as opaque identifiers
        - Base analysis ONLY on frequency spectrum data returned by this tool
        - If filename suggests a characteristic but data shows otherwise, report the data findings

        **CRITICAL - Parameter Validation:**
        - Sampling rate resolution: explicit parameter > companion metadata file
          > structured error. There is NO silent default sampling rate.
        - Segment duration defaults to 1.0s but can be customized

        Args:
            ctx: MCP context for user communication
            filename: Name of the file containing the signal
            sampling_rate: Sampling frequency in Hz. If None, it is read from
                the companion _metadata.json; if neither is available the tool
                raises ValueError (analysis never proceeds on a guessed rate).
            max_frequency: Maximum frequency to analyze (default: Nyquist frequency)
            segment_duration: Duration in seconds to analyze (default: 1.0s random segment).
                             Set to None to analyze full signal.
            random_seed: Random seed for reproducible segment selection (default: None = random)

        Returns:
            FFTResult with frequencies, magnitudes and dominant peak

        Raises:
            ValueError: If no sampling rate is provided and none is found in
                the companion metadata file.
        """
        # Resolve sampling rate (explicit > metadata > error) and segment duration
        sampling_rate, segment_duration = await _load_and_validate_metadata(
            ctx=ctx,
            filename=filename,
            data_dir=DATA_DIR,
            load_signal_data_fn=load_signal_data,
            provided_sampling_rate=sampling_rate,
            provided_segment_duration=segment_duration,
            default_segment_duration=1.0,
        )

        # Load data
        signal_data = load_signal_data(filename)

        if signal_data is None:
            raise ValueError(f"Unable to load signal from {filename}")

        # Extract segment if requested
        full_signal_length = len(signal_data)
        signal_duration_sec = full_signal_length / sampling_rate

        if segment_duration is not None and segment_duration < signal_duration_sec:
            signal_data = extract_segment(signal_data, segment_duration, sampling_rate, seed=random_seed)
            logger.info(f"Analyzing {segment_duration}s random segment from {signal_duration_sec:.1f}s signal")
        else:
            logger.info(f"Analyzing full signal ({signal_duration_sec:.1f}s, {full_signal_length} samples)")

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

    @mcp.tool()
    async def analyze_envelope(
        ctx: Context,
        filename: str,
        sampling_rate: Optional[float] = None,
        filter_low: float = 500.0,
        filter_high: float = 2000.0,
        num_peaks: int = 5,
        segment_duration: Optional[float] = None,
        random_seed: Optional[int] = None
    ) -> EnvelopeResult:
        """
        Perform Envelope Analysis to detect bearing faults.

        Envelope analysis is particularly effective for detecting faults in ball/roller bearings.
        The signal is high-pass filtered, then the envelope is calculated via Hilbert transform,
        and finally the envelope spectrum is analyzed.

        By default, analyzes a RANDOM 1.0-second segment from the signal for efficiency.
        Set segment_duration=None to analyze the entire signal.

        Returns ONLY peak information and diagnosis text (no full arrays) to avoid context overflow.

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from filename** (e.g., "OuterRaceFault_1.csv" does NOT mean outer race fault exists)
        - **NEVER assume signal characteristics from filename** (e.g., "baseline" does NOT mean healthy)
        - Treat ALL filenames as opaque identifiers
        - Base diagnosis ONLY on frequency-domain evidence (peaks matching BPFO/BPFI/BSF/FTF)
        - If filename suggests a fault but analysis shows no evidence, report "No fault detected despite filename"

        **CRITICAL - Parameter Validation:**
        - Sampling rate resolution: explicit parameter > companion metadata file
          > structured error. There is NO silent default sampling rate.
        - Segment duration defaults to 1.0s but can be customized

        Args:
            ctx: MCP context for user communication
            filename: Name of the file containing the signal
            sampling_rate: Sampling frequency in Hz. If None, it is read from
                the companion _metadata.json; if neither is available the tool
                raises ValueError (analysis never proceeds on a guessed rate).
            filter_low: Low frequency of bandpass filter in Hz (default: 500 Hz)
            filter_high: High frequency of bandpass filter in Hz (default: 2000 Hz)
            num_peaks: Number of main peaks to identify (default: 5)
            segment_duration: Duration in seconds to analyze (default: 1.0s random segment).
                             Set to None to analyze full signal.
            random_seed: Random seed for reproducible segment selection (default: None = random)

        Returns:
            EnvelopeResult with peak information and diagnosis (optimized for chat display)

        Raises:
            ValueError: If no sampling rate is provided and none is found in
                the companion metadata file.
        """
        # Resolve sampling rate (explicit > metadata > error) and segment duration
        sampling_rate, segment_duration = await _load_and_validate_metadata(
            ctx=ctx,
            filename=filename,
            data_dir=DATA_DIR,
            load_signal_data_fn=load_signal_data,
            provided_sampling_rate=sampling_rate,
            provided_segment_duration=segment_duration,
            default_segment_duration=1.0,
        )

        # Load data
        signal_data = load_signal_data(filename)

        if signal_data is None:
            raise ValueError(f"Unable to load signal from {filename}")

        # Extract segment if requested
        full_signal_length = len(signal_data)
        signal_duration_sec = full_signal_length / sampling_rate

        if segment_duration is not None and segment_duration < signal_duration_sec:
            signal_data = extract_segment(signal_data, segment_duration, sampling_rate, seed=random_seed)
            logger.info(f"Analyzing {segment_duration}s random segment from {signal_duration_sec:.1f}s signal")
        else:
            logger.info(f"Analyzing full signal ({signal_duration_sec:.1f}s, {full_signal_length} samples)")

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
        from scipy.signal import find_peaks

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

    @mcp.tool()
    def analyze_statistics(filename: str) -> StatisticalResult:
        """
        Calculate statistical parameters of the signal for diagnostics.

        Statistical parameters are key indicators for diagnostics:
        - RMS: Effective value, correlated to signal energy
        - Crest Factor: Indicates presence of impulses (high = possible faults)
        - Kurtosis: Measures impulsiveness (excess kurtosis; >0 = non-Gaussian, >3 = strong impulses)
        - Peak-to-Peak: Signal range

        **CRITICAL - LLM Inference Policy:**
        - **NEVER infer fault type from filename** (e.g., "OuterRaceFault_1.csv" does NOT mean outer race fault exists)
        - **NEVER assume signal characteristics from filename** (e.g., "baseline" does NOT mean healthy)
        - Treat ALL filenames as opaque identifiers
        - Statistical parameters (RMS/CF/Kurtosis) are indicators ONLY - NOT definitive diagnostics
        - High kurtosis indicates "possible fault" - NOT "confirmed fault"
        - Must be combined with frequency-domain evidence for diagnosis

        **Signal units:** all values are in the signal's native unit. The unit
        is reported only when DECLARED in the companion _metadata.json — it is
        never guessed from signal amplitude. ISO 20816-3 severity tools refuse
        to produce a verdict until the unit is declared.

        Args:
            filename: Name of the file containing the signal

        Returns:
            StatisticalResult with all statistical parameters
        """
        # Load data
        signal_data = load_signal_data(filename)

        if signal_data is None:
            raise ValueError(f"Unable to load signal from {filename}")

        # Calculate statistical parameters
        rms = float(np.sqrt(np.mean(signal_data**2)))
        peak = float(np.max(np.abs(signal_data)))
        peak_to_peak = float(np.max(signal_data) - np.min(signal_data))
        mean_val = float(np.mean(signal_data))
        std_dev = float(np.std(signal_data))

        # Crest Factor
        crest_factor = peak / rms if rms > 0 else 0.0

        # Kurtosis (using scipy)
        from scipy.stats import kurtosis, skew
        kurtosis_val = float(kurtosis(signal_data, fisher=True))  # Fisher=True for excess kurtosis
        skewness_val = float(skew(signal_data))

        # Signal unit: DECLARED only (companion metadata) — never guessed
        # from amplitude. An unrecognized declaration counts as undeclared.
        declared_unit = None
        metadata_path = get_metadata_path(filename)
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                declared_unit = normalize_signal_unit(metadata.get("signal_unit"))
            except Exception as e:
                logger.warning(f"Error reading metadata {metadata_path}: {e}")

        if declared_unit is not None:
            unit_note = (
                f"Signal unit declared as '{declared_unit}' in "
                f"{metadata_path.name}. All values above are in this unit."
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

    @mcp.tool()
    async def extract_features_from_signal(
        signal_file: str,
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
        ctx: Context = None
    ) -> FeatureExtractionResult:
        """
        Extract time-domain features from signal using sliding window segmentation.

        Segments the signal into overlapping windows and extracts 17 statistical features
        from each segment. Features include: mean, std, RMS, kurtosis, crest factor, entropy, etc.

        Args:
            signal_file: Name of the CSV file in data/signals/
            sampling_rate: Sampling frequency in Hz. If None, it is read from
                the companion _metadata.json; if neither is available the tool
                raises ValueError (never a silent default rate).
            segment_duration: Duration of each segment in seconds (default: 0.1)
            overlap_ratio: Overlap between segments, 0-1 (default: 0.5 = 50%)
            ctx: MCP context for progress/logging

        Returns:
            FeatureExtractionResult with features matrix and metadata

        Raises:
            ValueError: If no sampling rate is provided and none is found in
                the companion metadata file.

        Example:
            extract_features_from_signal(
                "healthy_motor.csv",
                sampling_rate=10000,
                segment_duration=0.2,
                overlap_ratio=0.5
            )
        """
        if ctx:
            await ctx.info(f"Extracting features from {signal_file}...")

        # Resolve sampling rate: explicit > metadata > structured error
        # (never a silent default rate).
        sampling_rate = _resolve_sampling_rate(signal_file, sampling_rate, strict=True)
        if ctx:
            await ctx.info(f"Using sampling rate: {sampling_rate} Hz")

        # Load signal
        filepath = DATA_DIR / signal_file
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {signal_file}")

        signal_data = load_signal_data(signal_file)
        if signal_data is None:
            raise ValueError(f"Could not load signal data from: {signal_file}")

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
        features_file = DATA_DIR / f"features_{sanitize_filename(signal_file)}"
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

    @mcp.tool()
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
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'. Re-load with sampling_rate parameter.")

        if ctx:
            await ctx.info(f"Computing PSD for '{signal_id}' ({info['num_samples']} samples, {fs} Hz)")

        result = _compute_psd(signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window)

        return PSDResult(
            signal_id=signal_id,
            num_samples=info["num_samples"],
            sampling_rate=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            top_peaks=[SpectralPeak(**p) for p in result["top_peaks"]],
            total_power=result["total_power"],
            freq_range_hz=result["freq_range_hz"],
            frequency_resolution=result["frequency_resolution"],
        )

    @mcp.tool()
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
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'.")

        if ctx:
            await ctx.info(f"Computing STFT for '{signal_id}'")

        result = _compute_stft(signal_data, fs, nperseg=nperseg, noverlap=noverlap, window=window)

        return STFTResult(
            signal_id=signal_id,
            num_samples=info["num_samples"],
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

    @mcp.tool()
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
        repo = get_repository()
        signal_data = repo.get_signal(signal_id)
        info = repo.get_signal_info(signal_id)
        fs = info.get("sampling_rate")
        if fs is None:
            raise ValueError(f"No sampling_rate for signal '{signal_id}'.")

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
