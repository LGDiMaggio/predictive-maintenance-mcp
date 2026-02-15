"""
Pydantic models for structured MCP tool outputs.

These models define the data contracts for all MCP tool responses,
ensuring consistent and well-documented return types.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class FFTResult(BaseModel):
    """FFT analysis result with structured output."""
    frequencies: list[float] = Field(description="Frequency array (Hz)")
    magnitudes: list[float] = Field(description="Magnitude array")
    peak_frequency: float = Field(description="Dominant peak frequency (Hz)")
    peak_magnitude: float = Field(description="Dominant peak magnitude")
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
