"""
Pydantic models for structured MCP tool outputs.

These models define the data contracts for all MCP tool responses,
ensuring consistent and well-documented return types.
"""

from typing import Any, Optional
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
    """Statistical analysis result of the signal."""
    rms: float = Field(description="Root Mean Square (effective value)")
    peak_to_peak: float = Field(description="Peak-to-peak value")
    peak: float = Field(description="Peak value")
    crest_factor: float = Field(description="Crest Factor (Peak/RMS)")
    kurtosis: float = Field(description="Kurtosis (measure of impulsiveness)")
    skewness: float = Field(description="Skewness (asymmetry)")
    mean: float = Field(description="Mean value")
    std_dev: float = Field(description="Standard deviation")
    detected_unit: str = Field(description="Auto-detected signal unit (g acceleration or mm/s velocity)")
    unit_note: str = Field(description="Important note about signal units and conversion requirements")


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
    overall_health: str = Field(description="Overall health status: 'Healthy', 'Suspicious', 'Faulty'")
    confidence: str = Field(description="Confidence level: 'High', 'Medium', 'Low'")


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
    signal_unit: Optional[str] = Field(None, description="Signal unit (g, mm/s, m/s², etc.)")


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
    confidence: str = Field(description="Confidence: high, moderate, low, or none")


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


class VibrationSeverityResult(BaseModel):
    """ISO 10816/20816 severity result using signal_id pattern."""
    signal_id: str = Field(description="Signal identifier used")
    rms_velocity_mm_s: float = Field(description="RMS velocity in mm/s")
    machine_class: str = Field(description="Machine class (I, II, III, or IV)")
    axis: str = Field(description="Measurement axis")
    zone: str = Field(description="ISO zone: A, B, C, or D")
    zone_description: str = Field(description="Zone description")
    severity_level: str = Field(description="Good, Acceptable, Unsatisfactory, or Unacceptable")
    color_code: str = Field(description="green, yellow, orange, or red")
    boundaries: dict[str, float] = Field(description="Zone boundaries {AB, BC, CD} in mm/s")
    unit_conversion_performed: bool = Field(description="Whether acceleration-to-velocity conversion was done")
    original_unit: Optional[str] = Field(None, description="Original signal unit before conversion")


class DiagnosisResult(BaseModel):
    """Full integrated diagnosis pipeline result."""
    signal_id: str = Field(description="Signal identifier used")
    rpm: float = Field(description="Machine speed (RPM)")
    bearing_id: Optional[str] = Field(None, description="Bearing used (if any)")
    machine_class: str = Field(description="Machine class for ISO severity")
    fft_summary: dict[str, Any] = Field(description="FFT key findings")
    psd_summary: dict[str, Any] = Field(description="PSD key findings")
    stft_summary: dict[str, Any] = Field(description="STFT key findings")
    bearing_faults: Optional[BearingFaultsSummary] = Field(None, description="Bearing fault results")
    iso_severity: VibrationSeverityResult = Field(description="ISO severity assessment")
    anomaly_detection: Optional[dict[str, Any]] = Field(None, description="Anomaly detection results (health, ratio, score)")
    overall_diagnosis: str = Field(description="Combined diagnostic text")
    confidence: str = Field(description="Overall confidence: high, moderate, or low")
    recommendations: list[str] = Field(description="Recommended actions")


# ============================================================================
# Phase 2 Models — Prognostics & Decision Support
# ============================================================================


class RULEstimationResult(BaseModel):
    """Remaining Useful Life estimation result."""
    rul: float = Field(description="Estimated remaining useful life in time units")
    confidence: float = Field(description="R-squared or confidence metric (0-1)")
    method: str = Field(description="Estimation method used")
    confidence_interval: list[float] | None = Field(default=None, description="[lower, upper] RUL bounds (Kalman only)")
    shape: float | None = Field(default=None, description="Weibull shape parameter (Weibull only)")
    scale: float | None = Field(default=None, description="Weibull scale parameter (Weibull only)")
    estimated_rate: float | None = Field(default=None, description="Estimated degradation rate (Kalman only)")


class TrendAnalysisResult(BaseModel):
    """Signal trend analysis result."""
    feature_name: str = Field(description="Feature analyzed")
    slope: float = Field(description="Trend slope per segment")
    intercept: float = Field(description="Trend intercept")
    r_squared: float = Field(description="R-squared goodness of fit")
    trend_direction: str = Field(description="increasing, decreasing, or stable")
    p_value: float = Field(description="Statistical significance")
    num_segments: int = Field(description="Number of segments analyzed")


class DegradationOnsetResult(BaseModel):
    """Degradation onset detection result."""
    feature_name: str = Field(description="Feature analyzed")
    onset_detected: bool = Field(description="Whether degradation onset was detected")
    onset_segment_index: int | None = Field(default=None, description="Segment index where degradation starts")
    threshold_sigma: float = Field(description="Sigma threshold used for detection")
    num_segments: int = Field(description="Total number of segments")


class AlertResult(BaseModel):
    """Vibration alert assessment result."""
    alert_level: str = Field(description="none, warning, alarm, or danger")
    zone: str = Field(description="ISO zone classification (A/B/C/D)")
    rms_velocity: float = Field(description="Input RMS velocity value")
    exceeded_threshold: float | None = Field(default=None, description="Threshold that was exceeded")
    message: str = Field(description="Human-readable alert message")
