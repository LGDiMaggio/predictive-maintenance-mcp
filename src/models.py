"""
Pydantic models for structured MCP tool outputs.

These models define the data contracts for all MCP tool responses,
ensuring consistent and well-documented return types.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class SpectralPeak(BaseModel):
    """A single peak in the frequency spectrum."""
    frequency_hz: float = Field(description="Peak frequency in Hz")
    magnitude: float = Field(description="Peak magnitude (linear)")
    magnitude_db: float = Field(description="Peak magnitude in dB (relative to max)")
    note: str = Field(default="", description="Optional annotation (e.g. harmonic label)")


class FFTResult(BaseModel):
    """FFT analysis result — compact summary (top peaks + stats, no full arrays).
    
    Full-length arrays are never returned to the LLM to avoid context overflow.
    Use generate_fft_report() or plot_spectrum() for visual inspection."""
    top_peaks: list[SpectralPeak] = Field(description="Top spectral peaks sorted by magnitude")
    peak_frequency: float = Field(description="Dominant peak frequency (Hz)")
    peak_magnitude: float = Field(description="Dominant peak magnitude")
    rms_spectral: float = Field(description="RMS of the magnitude spectrum")
    total_bins: int = Field(description="Total number of FFT bins computed")
    freq_range_hz: list[float] = Field(description="[min_freq, max_freq] of the spectrum")
    sampling_rate: float = Field(description="Sampling frequency (Hz)")
    num_samples: int = Field(description="Number of analyzed samples")
    frequency_resolution: float = Field(description="Frequency resolution (Hz)")


class EnvelopeResult(BaseModel):
    """Envelope analysis result - optimized for chat display."""
    # Summary statistics instead of full arrays
    num_samples: int = Field(description="Number of samples in envelope signal")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    filter_band: tuple[float, float] = Field(description="Bandpass filter band (Hz)")

    # Only top peaks (not full spectrum)
    peak_frequencies: list[float] = Field(description="Top peak frequencies (Hz)")
    peak_magnitudes: list[float] = Field(description="Top peak magnitudes")

    # Human-readable diagnosis
    diagnosis: str = Field(description="Interpretive diagnosis text with bearing frequency analysis")

    # Optional: small preview of spectrum (first 100 points for visualization hint)
    spectrum_preview_freq: list[float] = Field(default=[], description="First 100 freq points (Hz)")
    spectrum_preview_mag: list[float] = Field(default=[], description="First 100 magnitude points")


class StatisticalResult(BaseModel):
    """Statistical analysis result of the signal.

    Values are in the signal's native unit. The unit is reported only when
    DECLARED (companion ``_metadata.json`` or ``load_signal(signal_unit=...)``)
    — it is never guessed from signal amplitude.
    """
    rms: float = Field(description="Root Mean Square (effective value)")
    peak_to_peak: float = Field(description="Peak-to-peak value")
    peak: float = Field(description="Peak value")
    crest_factor: float = Field(description="Crest Factor (Peak/RMS)")
    kurtosis: float = Field(description="Kurtosis (measure of impulsiveness)")
    skewness: float = Field(description="Skewness (asymmetry)")
    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")
    signal_unit: Optional[str] = Field(
        None,
        description=(
            "Declared signal unit ('g', 'm/s2', 'mm/s', 'm/s') from companion "
            "metadata — never guessed from amplitude. None when not declared."
        ),
    )
    unit_note: str = Field(description="Unit declaration status and how to declare the unit for ISO severity assessment")


class SignalInfo(BaseModel):
    """Information about an available signal."""
    filename: str = Field(description="File name")
    path: str = Field(description="Full path")
    size_bytes: int = Field(description="File size in bytes")
    num_samples: Optional[int] = Field(None, description="Number of samples (if available)")


class ISO20816Result(BaseModel):
    """ISO 20816-3 vibration severity evaluation result."""
    rms_velocity: float = Field(description="RMS velocity in mm/s (broadband)")
    machine_group: int = Field(description="Machine group (1 or 2)")
    support_type: str = Field(description="Support type: 'rigid' or 'flexible'")
    zone: str = Field(description="Evaluation zone: 'A', 'B', 'C', or 'D'")
    zone_description: str = Field(description="Zone description and recommendation")
    severity_level: str = Field(description="Severity level: 'Good', 'Acceptable', 'Unsatisfactory', 'Unacceptable'")
    color_code: str = Field(description="Color code: 'green', 'yellow', 'orange', 'red'")
    boundary_ab: float = Field(description="Zone A/B boundary (mm/s)")
    boundary_bc: float = Field(description="Zone B/C boundary (mm/s)")
    boundary_cd: float = Field(description="Zone C/D boundary (mm/s)")
    frequency_range: str = Field(description="Frequency range used for measurement")
    operating_speed_rpm: Optional[float] = Field(None, description="Operating speed in RPM")
    threshold_provenance: Optional[str] = Field(None, description="Provenance of the zone boundary values (ISO edition note)")


class FeatureExtractionResult(BaseModel):
    """Result of time-domain feature extraction from signal segments."""
    num_segments: int = Field(description="Number of segments extracted")
    segment_length_samples: int = Field(description="Samples per segment")
    segment_duration_s: float = Field(description="Duration of each segment in seconds")
    overlap_ratio: float = Field(description="Overlap ratio between segments")
    features_shape: list[int] = Field(description="Shape of feature matrix [num_segments, num_features]")
    feature_names: list[str] = Field(description="Names of extracted features")
    features_preview: list[dict[str, float]] = Field(description="First 5 segments features (preview)")


class AnomalyModelResult(BaseModel):
    """Result of anomaly detection model training."""
    model_type: str = Field(description="Type of model: 'OneClassSVM' or 'LocalOutlierFactor'")
    num_training_samples: int = Field(description="Number of healthy samples used for training")
    num_features_original: int = Field(description="Number of original features")
    num_features_pca: int = Field(description="Number of PCA components (features after dimensionality reduction)")
    variance_explained: float = Field(description="Cumulative variance explained by PCA components")
    model_params: dict[str, Any] = Field(description="Best model hyperparameters")
    model_path: str = Field(description="Path to saved model file (.pkl)")
    scaler_path: str = Field(description="Path to saved scaler file (.pkl)")
    pca_path: str = Field(description="Path to saved PCA file (.pkl)")
    validation_accuracy: Optional[float] = Field(None, description="Overall balanced accuracy on healthy + fault validation data")
    validation_details: Optional[str] = Field(None, description="Validation details with healthy and fault metrics")
    validation_metrics: Optional[dict[str, Any]] = Field(None, description="Detailed validation metrics (healthy/fault accuracy breakdown)")


class AnomalyPredictionResult(BaseModel):
    """Result of anomaly detection prediction on new data."""
    num_segments: int = Field(description="Number of segments analyzed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    anomaly_ratio: float = Field(description="Ratio of anomalies (0-1)")
    predictions: list[int] = Field(description="Predictions per segment: 1=normal, -1=anomaly")
    anomaly_scores: Optional[list[float]] = Field(None, description="Anomaly scores if available")
    overall_health: str = Field(description="Overall health status: 'Healthy', 'Suspicious', 'Faulty' (thresholded on anomaly_ratio: <0.1, <0.3, >=0.3)")


# ============================================================================
# Phase 1 Models — Signal Repository, Spectral, Bearing, ISO, Diagnosis
# ============================================================================


class StoredSignalInfo(BaseModel):
    """Metadata for a signal stored in the SignalRepository."""
    signal_id: str = Field(description="Unique identifier for the stored signal")
    filepath: str = Field(description="Original file path")
    load_timestamp: str = Field(description="ISO 8601 timestamp when signal was loaded")
    shape: list[int] = Field(description="Shape of the signal array")
    num_samples: int = Field(description="Number of samples")
    sampling_rate: Optional[float] = Field(None, description="Sampling rate in Hz")
    duration_s: Optional[float] = Field(None, description="Duration in seconds")
    size_bytes: int = Field(description="Approximate memory size in bytes")
    signal_unit: Optional[Literal["g", "m/s2", "mm/s", "m/s"]] = Field(
        None,
        description=(
            "DECLARED signal unit — from load_signal(signal_unit=...) or the "
            "companion _metadata.json ('signal_unit' field). Never guessed. "
            "None means undeclared: ISO severity verdicts will be refused "
            "until the unit is declared."
        ),
    )


class PSDResult(BaseModel):
    """Power Spectral Density (Welch method) result."""
    signal_id: str = Field(description="Signal identifier used")
    num_samples: int = Field(description="Number of samples analyzed")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    nperseg: int = Field(description="Samples per segment")
    noverlap: int = Field(description="Overlap between segments")
    window: str = Field(description="Window function used")
    top_peaks: list[SpectralPeak] = Field(description="Top spectral peaks by power")
    total_power: float = Field(description="Total integrated power")
    freq_range_hz: list[float] = Field(description="[min_freq, max_freq]")
    frequency_resolution: float = Field(description="Frequency resolution (Hz)")


class STFTResult(BaseModel):
    """STFT Spectrogram result — summary only, no full 2D arrays."""
    signal_id: str = Field(description="Signal identifier used")
    num_samples: int = Field(description="Number of samples analyzed")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    nperseg: int = Field(description="Samples per segment")
    noverlap: int = Field(description="Overlap between segments")
    window: str = Field(description="Window function used")
    num_time_bins: int = Field(description="Number of time bins")
    num_freq_bins: int = Field(description="Number of frequency bins")
    freq_range_hz: list[float] = Field(description="[min_freq, max_freq]")
    time_range_s: list[float] = Field(description="[start_time, end_time]")
    max_power_freq_hz: float = Field(description="Frequency with maximum power")
    max_power_time_s: float = Field(description="Time of maximum power")
    energy_per_band: list[dict[str, float]] = Field(description="Energy in predefined frequency bands")


class EnvelopeSpectrumResult(BaseModel):
    """Envelope spectrum result using signal_id pattern."""
    signal_id: str = Field(description="Signal identifier used")
    num_samples: int = Field(description="Number of samples analyzed")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    method: str = Field(description="Envelope extraction method")
    frequency_range: tuple[float, float] = Field(description="Analysis frequency range (Hz)")
    top_peaks: list[SpectralPeak] = Field(description="Top peaks in envelope spectrum")
    diagnosis: str = Field(description="Interpretive diagnosis text")


class BearingFaultCheckResult(BaseModel):
    """Result of checking for a specific bearing fault frequency."""
    signal_id: str = Field(description="Signal identifier used")
    bearing_id: str = Field(description="Bearing designation")
    fault_type: str = Field(description="Fault type: BPFO, BPFI, BSF, or FTF")
    expected_frequency_hz: float = Field(description="Expected fault frequency")
    detected: bool = Field(description="Whether a peak was detected within tolerance")
    detected_frequency_hz: Optional[float] = Field(None, description="Actual peak frequency")
    magnitude: Optional[float] = Field(None, description="Magnitude at detected frequency")
    deviation_pct: Optional[float] = Field(None, description="Deviation from expected (%)")
    harmonics_detected: list[dict[str, float]] = Field(default=[], description="Harmonics found")
    evidence_strength: str = Field(
        description=(
            "Strength of the spectral evidence for this fault: 'high' "
            "(fundamental + >=2 harmonics), 'moderate' (fundamental, weaker "
            "harmonics), 'low' (harmonics only), or 'none'. Derived from "
            "detected peaks — not a probability."
        )
    )


class BearingFaultsSummary(BaseModel):
    """Summary of all fault checks for one bearing."""
    signal_id: str = Field(description="Signal identifier used")
    bearing_id: str = Field(description="Bearing designation")
    rpm: float = Field(description="Shaft speed (RPM)")
    shaft_frequency_hz: float = Field(description="Shaft frequency (Hz)")
    bearing_frequencies: dict[str, float] = Field(description="Calculated BPFO/BPFI/BSF/FTF (Hz)")
    fault_checks: list[BearingFaultCheckResult] = Field(description="Results for each fault type")
    overall_assessment: str = Field(description="Summary assessment text")
    most_likely_fault: Optional[str] = Field(None, description="Most likely fault type if any")


class BearingCatalogMiss(BaseModel):
    """Typed 'not in catalog' result for a bearing catalog lookup.

    A missing catalog entry is a legitimate negative outcome, not a tool
    failure — the catalog is intentionally small (verified geometry only),
    so the miss is expressed in the SCHEMA (status + suggestion) instead of
    an ad-hoc dict with an 'error' key. No geometry is ever invented.
    """
    status: Literal["not_found"] = Field(
        "not_found",
        description="Always 'not_found' — discriminates from a catalog hit",
    )
    bearing_designation: str = Field(
        description="The designation that was searched for"
    )
    suggestion: str = Field(
        description="Concrete next step to obtain the bearing geometry"
    )
    catalog_contains: list[str] = Field(
        description="Designations actually present in the verified catalog"
    )


class ISOSeverityRefusal(BaseModel):
    """Structured refusal of an ISO severity verdict.

    Returned in place of a severity assessment when the verdict cannot be
    produced honestly (undeclared signal unit, sampling rate too low for the
    ISO evaluation band, machine out of scope). The refusal is part of the
    SCHEMA — not prose in a log message — so LLM clients cannot lose it.
    """
    status: Literal["refused"] = Field(
        "refused", description="Always 'refused' — discriminates from an assessed result"
    )
    signal_id: str = Field(default="", description="Signal identifier used")
    reason: str = Field(description="Why the ISO severity verdict was refused")
    remedy: str = Field(
        description=(
            "Concrete action to obtain a verdict, e.g. re-load with "
            "load_signal(signal_unit=...) or re-acquire at a higher sampling rate"
        )
    )


class VibrationSeverityResult(BaseModel):
    """ISO 20816-3 severity result (zone boundaries from ISO 10816-3:2009) using signal_id pattern."""
    status: Literal["assessed"] = Field(
        "assessed", description="Always 'assessed' — discriminates from a refused result"
    )
    signal_id: str = Field(description="Signal identifier used")
    rms_velocity_mm_s: float = Field(description="RMS velocity in mm/s")
    machine_group: int = Field(description="ISO 20816-3 machine group: 1 (large, >300 kW) or 2 (medium, 15-300 kW)")
    support_type: str = Field(description="Support type: 'rigid' or 'flexible'")
    axis: str = Field(description="Measurement axis")
    zone: str = Field(description="ISO zone: A, B, C, or D")
    zone_description: str = Field(description="Zone description")
    severity_level: str = Field(description="Good, Acceptable, Unsatisfactory, or Unacceptable")
    color_code: str = Field(description="green, yellow, orange, or red")
    boundaries: dict[str, float] = Field(description="Zone boundaries {AB, BC, CD} in mm/s")
    frequency_range: str = Field(description="Actual evaluation band used (may be narrower than the ISO nominal 10-1000 Hz when fs limits it)")
    unit_conversion_performed: bool = Field(description="Whether acceleration-to-velocity conversion was done")
    original_unit: Optional[str] = Field(None, description="Original signal unit before conversion")
    threshold_provenance: str = Field(description="Provenance of the zone boundary values (ISO edition note)")


class DiagnosisResult(BaseModel):
    """Full integrated diagnosis pipeline result."""
    signal_id: str = Field(description="Signal identifier used")
    rpm: float = Field(description="Machine speed (RPM)")
    bearing_id: Optional[str] = Field(None, description="Bearing used (if any)")
    machine_group: int = Field(description="ISO 20816-3 machine group used for severity: 1 (large) or 2 (medium)")
    support_type: str = Field(description="Support type used for severity: 'rigid' or 'flexible'")
    fft_summary: dict[str, Any] = Field(description="FFT key findings")
    psd_summary: dict[str, Any] = Field(description="PSD key findings")
    stft_summary: dict[str, Any] = Field(description="STFT key findings")
    bearing_faults: Optional[BearingFaultsSummary] = Field(None, description="Bearing fault results")
    iso_severity: VibrationSeverityResult | ISOSeverityRefusal = Field(
        description=(
            "ISO severity assessment, or a structured refusal "
            "(status='refused' with reason + remedy) when the verdict cannot "
            "be produced honestly — e.g. undeclared signal unit or Nyquist "
            "below the ISO evaluation band. The other diagnosis blocks "
            "(spectral, bearing, anomaly) still run."
        )
    )
    anomaly_detection: Optional[dict[str, Any]] = Field(None, description="Anomaly detection results (health, ratio, score)")
    overall_diagnosis: str = Field(description="Combined diagnostic text")
    evidence_strength: str = Field(
        description=(
            "Strength of corroborating fault evidence: 'none', 'weak', "
            "'moderate', or 'strong'. Derived from the number and quality of "
            "independent findings (bearing fault frequency matches, shaft "
            "signatures, anomaly detection, ISO severity) — NOT from severity "
            "alone and NOT a probability. 'none' means no fault evidence was "
            "found (machine appears healthy)."
        )
    )
    recommendations: list[str] = Field(description="Recommended actions")


# ============================================================================
# Phase 2 Models — Prognostics & Decision Support
# ============================================================================


class RULEstimationResult(BaseModel):
    """Remaining Useful Life estimate from repeated measurements over time.

    RUL is only physically meaningful when fitted on a degradation trend
    across multiple measurements of the same machine (days/weeks/months).
    ``fit_r_squared`` describes how well the degradation curve fits the
    observed series; it is NOT a probability that the estimate is correct.
    Extrapolation beyond the observation horizon is inherently uncertain.
    """
    status: Literal["estimated", "no_degradation_trend", "threshold_already_exceeded"] = Field(
        description=(
            "'estimated' (RUL computed), 'no_degradation_trend' (no "
            "statistically significant trend toward the threshold — healthy "
            "outcome, no RUL number), or 'threshold_already_exceeded' (last "
            "measurement is at/above the failure threshold)."
        )
    )
    method: str = Field(description="Estimation method used: linear, exponential, or kalman")
    feature_name: str = Field(description="Degradation indicator tracked (e.g. 'rms')")
    num_measurements: int = Field(description="Number of measurements in the series")
    observation_horizon: float = Field(
        description=(
            "Time span covered by the measurement series (last minus first "
            "timestamp), in time_unit. RUL estimates far beyond this horizon "
            "are extrapolations with low reliability."
        )
    )
    time_unit: str = Field(description="Unit of timestamps, observation_horizon, and rul")
    failure_threshold: float = Field(description="Indicator value considered as failure")
    current_value: float = Field(description="Most recent measured indicator value")
    trend_p_value: Optional[float] = Field(
        None,
        description=(
            "Two-sided p-value of the series' linear slope (None when not "
            "computable). The trend gate requires p < 0.05."
        ),
    )
    rul: Optional[float] = Field(
        None,
        description="Estimated remaining useful life in time_unit (only when status='estimated')",
    )
    fit_r_squared: Optional[float] = Field(
        None,
        description=(
            "R-squared of the fitted degradation curve on the observed data. "
            "Goodness of fit only — NOT a confidence or probability. None for "
            "the kalman method."
        ),
    )
    estimated_rate: Optional[float] = Field(
        None,
        description="Estimated degradation rate in feature units per time_unit (linear/kalman)",
    )
    rul_interval_95: Optional[list[float]] = Field(
        None,
        description=(
            "[lower, upper] approximate 95% interval from the delta-method "
            "variance (kalman only). Coverage not validated — treat as an "
            "order-of-magnitude band."
        ),
    )
    precision_heuristic: Optional[float] = Field(
        None,
        description=(
            "Heuristic in [0,1]: 1 - rul_std/rul, clipped (kalman only). "
            "This is a heuristic, NOT a statistical confidence — do not "
            "present it as a probability of correctness."
        ),
    )
    message: str = Field(description="Human-readable explanation of the outcome and its caveats")


class TrendAnalysisResult(BaseModel):
    """Within-recording feature trend — screening only, NOT a prognosis.

    Segments a single recording (seconds of data) and fits a trend on the
    per-segment feature values. Use it to screen whether a recording is
    stationary. It cannot estimate Remaining Useful Life: for RUL, collect
    repeated measurements over days/weeks and pass them to estimate_rul.
    """
    feature_name: str = Field(description="Feature analyzed")
    slope: float = Field(description="Trend slope in feature units per second (within the recording)")
    intercept: float = Field(description="Trend intercept")
    r_squared: float = Field(description="R-squared goodness of fit of the linear trend")
    trend_direction: str = Field(
        description=(
            "increasing, decreasing, or stable — based on the slope "
            "significance test (p < 0.05), not on an R-squared cutoff"
        )
    )
    p_value: Optional[float] = Field(
        None,
        description="Two-sided p-value of the slope (None when not computable)",
    )
    num_segments: int = Field(description="Number of segments analyzed")
    analysis_scope: str = Field(
        description=(
            "Always 'within_recording_screening': this trend spans seconds "
            "of one recording, not the machine's life"
        )
    )
    feature_series: list[float] = Field(
        description=(
            "Per-segment feature values (evenly subsampled to at most 50 "
            "points). One recording yields ONE point for estimate_rul (e.g. "
            "the recording's overall feature value) — accumulate recordings "
            "over time to build its input series."
        )
    )
    segment_times_s: list[float] = Field(
        description="Segment center times in seconds for feature_series (same subsampling)"
    )
    series_truncated: bool = Field(
        description="True when feature_series was subsampled to the 50-point cap"
    )


class DegradationOnsetResult(BaseModel):
    """Degradation onset detection result (within-recording screening)."""
    feature_name: str = Field(description="Feature analyzed")
    onset_detected: bool = Field(description="Whether degradation onset was detected")
    onset_segment_index: int | None = Field(default=None, description="Segment index where degradation starts")
    threshold_sigma: float = Field(description="Sigma threshold used for detection")
    num_segments: int = Field(description="Total number of segments")
    baseline_segments: int = Field(
        description=(
            "Number of leading segments used as the baseline window. Onset "
            "is only searched AFTER this window; degradation starting inside "
            "the baseline cannot be detected by this method."
        )
    )


class AlertResult(BaseModel):
    """Vibration alert assessment result."""
    alert_level: str = Field(description="none, warning, alarm, or danger")
    zone: str = Field(description="ISO zone classification (A/B/C/D)")
    rms_velocity: float = Field(description="Input RMS velocity value")
    exceeded_threshold: float | None = Field(default=None, description="Threshold that was exceeded")
    message: str = Field(description="Human-readable alert message")
