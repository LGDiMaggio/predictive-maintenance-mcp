"""
Pydantic models for structured MCP tool outputs.

These models define the data contracts for all MCP tool responses,
ensuring consistent and well-documented return types.
"""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class SpectralPeak(BaseModel):
    """A single peak in the frequency spectrum."""

    frequency_hz: float = Field(description="Peak frequency in Hz")
    magnitude: float = Field(description="Peak magnitude (linear)")
    magnitude_db: float = Field(description="Peak magnitude in dB (relative to max)")
    note: str = Field(
        default="", description="Optional annotation (e.g. harmonic label)"
    )


class FFTResult(BaseModel):
    """FFT analysis result — compact summary (top peaks + stats, no full arrays).

    Full-length arrays are never returned to the LLM to avoid context overflow.
    Use generate_fft_report() for visual inspection."""

    top_peaks: list[SpectralPeak] = Field(
        description="Top spectral peaks sorted by magnitude"
    )
    peak_frequency: float = Field(description="Dominant peak frequency (Hz)")
    peak_magnitude: float = Field(description="Dominant peak magnitude")
    rms_spectral: float = Field(description="RMS of the magnitude spectrum")
    total_bins: int = Field(description="Total number of FFT bins computed")
    freq_range_hz: list[float] = Field(
        description="[min_freq, max_freq] of the spectrum"
    )
    sampling_rate: float = Field(description="Sampling frequency (Hz)")
    num_samples: int = Field(description="Number of analyzed samples")
    frequency_resolution: float = Field(description="Frequency resolution (Hz)")


class EnvelopeResult(BaseModel):
    """Unified envelope-spectrum analysis result (U9 merge).

    Compact summary: top peaks + the band ACTUALLY used — no full arrays.
    The envelope FFT is computed after mean subtraction + Hann window
    (audit 2.8 fix), so low-frequency (FTF-zone) peaks are not buried by
    DC leakage.
    """

    signal_id: str = Field(description="Signal identifier used")
    num_samples: int = Field(description="Number of samples analyzed (envelope length)")
    sampling_rate: float = Field(description="Sampling rate (Hz)")
    filter_band: tuple[float, float] = Field(
        description="Bandpass filter band (Hz) actually used — echoed from the request"
    )
    top_peaks: list[SpectralPeak] = Field(
        description="Top peaks in the envelope spectrum, sorted by frequency"
    )
    diagnosis: str = Field(
        description=(
            "Peak listing and comparison guidance. No reference bearing "
            "frequencies are assumed — compare against frequencies computed "
            "for the actual bearing and shaft speed."
        )
    )


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
    unit_note: str = Field(
        description="Unit declaration status and how to declare the unit for ISO severity assessment"
    )


class FeatureExtractionResult(BaseModel):
    """Result of time-domain feature extraction from signal segments."""

    num_segments: int = Field(description="Number of segments extracted")
    segment_length_samples: int = Field(description="Samples per segment")
    segment_duration_s: float = Field(description="Duration of each segment in seconds")
    overlap_ratio: float = Field(description="Overlap ratio between segments")
    features_shape: list[int] = Field(
        description="Shape of feature matrix [num_segments, num_features]"
    )
    feature_names: list[str] = Field(description="Names of extracted features")
    features_preview: list[dict[str, float]] = Field(
        description="First 5 segments features (preview)"
    )


class AnomalyModelResult(BaseModel):
    """Result of anomaly detection model training."""

    model_name: str = Field(
        description=(
            "Name under which the model was saved — pass this to "
            "predict_anomalies(model_name=...)"
        )
    )
    model_type: str = Field(
        description="Type of model: 'OneClassSVM' or 'LocalOutlierFactor'"
    )
    num_training_samples: int = Field(
        description="Number of healthy samples used for training"
    )
    num_features_original: int = Field(description="Number of original features")
    num_features_pca: int = Field(
        description="Number of PCA components (features after dimensionality reduction)"
    )
    variance_explained: float = Field(
        description="Cumulative variance explained by PCA components"
    )
    model_params: dict[str, Any] = Field(description="Best model hyperparameters")
    model_path: str = Field(description="Path to saved model file (.pkl)")
    scaler_path: str = Field(description="Path to saved scaler file (.pkl)")
    pca_path: str = Field(description="Path to saved PCA file (.pkl)")
    validation_accuracy: Optional[float] = Field(
        None, description="Overall balanced accuracy on healthy + fault validation data"
    )
    validation_details: Optional[str] = Field(
        None, description="Validation details with healthy and fault metrics"
    )
    validation_metrics: Optional[dict[str, Any]] = Field(
        None,
        description="Detailed validation metrics (healthy/fault accuracy breakdown)",
    )


class AnomalyPredictionResult(BaseModel):
    """Result of anomaly detection prediction on new data.

    Bounded output by design: counts, score percentiles, and the worst
    segments only — never per-segment arrays (a 6M-sample signal would
    dump tens of thousands of entries into the chat context).
    """

    model_name: str = Field(description="Name of the trained model used")
    num_segments: int = Field(description="Number of segments analyzed")
    anomaly_count: int = Field(description="Number of anomalies detected")
    anomaly_ratio: float = Field(description="Ratio of anomalies (0-1)")
    segment_duration_s: float = Field(
        description="Segment length in seconds (from the model's training metadata)"
    )
    score_percentiles: Optional[dict[str, float]] = Field(
        None,
        description=(
            "Percentiles (p5/p25/p50/p75/p95) of the model decision scores; "
            "negative = anomalous side. None when the model exposes no "
            "decision_function."
        ),
    )
    worst_segments: list[dict[str, float]] = Field(
        default=[],
        description=(
            "Up to 10 most anomalous segments, each with segment_index, "
            "start_time_s, and score (when available) — enough to locate the "
            "worst regions without dumping per-segment arrays."
        ),
    )
    overall_health: str = Field(
        description="Overall health status: 'Healthy', 'Suspicious', 'Faulty' (thresholded on anomaly_ratio: <0.1, <0.3, >=0.3)"
    )


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
    sampling_rate: Optional[float] = Field(
        None, gt=0, description="Sampling rate in Hz (must be positive when set)"
    )
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
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Complete companion _metadata.json of the source file (rpm/"
            "shaft_speed, reference frequencies, ...). Empty when the file "
            "has no companion metadata."
        ),
    )
    raw_format: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "EFFECTIVE raw-binary decode parameters (sample_format, "
            "byte_order, n_channels, channel_index, header_offset, "
            "scale_factor) after the explicit > companion > default merge — "
            "recorded as provenance so get_signal_info can answer 'how was "
            "this file decoded'. None for self-describing formats."
        ),
    )
    measurement: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Normalized measurement identity declared in the companion's "
            '"measurement" object: asset_id, measurement_point_id, '
            "acquired_at (ISO 8601 normalized to UTC), timezone_declared, "
            "timestamp_suspect, rpm, load, operating_state, sensor_id, "
            "direction, declared_by, measurement_id (first 16 hex of the "
            "SHA-256 of the file bytes plus the channel index), "
            "channel_index, content_sha256 (full digest of the file bytes) "
            "and size_bytes (file size). This block is AUTHORITATIVE over "
            "the verbatim object kept in source_metadata. None when the "
            "companion declares no measurement object (the file behaves "
            "exactly as before). The value RETURNED BY load_signal carries, "
            "in addition, the outcome of the asset-ledger registration: "
            "ledger_status ('recorded' | 'already_recorded' | 'superseded' "
            "| 'not_recorded'), reason (None, or why the status or the "
            "snapshot is not nominal), changed (keys that differ from the "
            "previous declaration of the same measurement, 'location' when "
            "the file moved), reattributed_from (asset the measurement was "
            "recorded under by mistake, or None), declaration_version, "
            "snapshot_status ('complete' | 'partial' | 'failed' | "
            "'skipped'), snapshot_id, processing_id (the snapshot lineage), "
            "comparability ({grade, qualifications} against the current "
            "declaration of the measurement point, informational, never "
            "stored) and missing ({block: {reason, remedy}} of the snapshot "
            "blocks the declared context could not support). "
            "get_signal_info and list_signals show the identity only."
        ),
    )
    companion_warning: Optional[str] = Field(
        None,
        description=(
            "Set when a companion _metadata.json exists but could not be used "
            "(not valid JSON, or not a JSON object): names the file and the "
            "case, and the signal was loaded exactly as if it had no "
            "companion. None when the companion was read fine or is absent."
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
    energy_per_band: list[dict[str, float]] = Field(
        description="Energy in predefined frequency bands"
    )


class BearingFaultCheckResult(BaseModel):
    """Result of checking for a specific expected fault frequency."""

    signal_id: str = Field(description="Signal identifier used")
    bearing_id: Optional[str] = Field(
        None, description="Bearing designation (None for arbitrary-frequency checks)"
    )
    fault_type: str = Field(
        description=(
            "Checked label: BPFO, BPFI, BSF, FTF, or an arbitrary "
            "user-provided label (e.g. 'GMF')"
        )
    )
    fault_type_canonical: Optional[
        Literal["outer_race", "inner_race", "ball", "cage"]
    ] = Field(
        None,
        description=(
            "Canonical fault vocabulary for the standard acronyms "
            "(BPFO=outer_race, BPFI=inner_race, BSF=ball, FTF=cage); "
            "None for arbitrary labels"
        ),
    )
    expected_frequency_hz: float = Field(description="Expected fault frequency")
    detected: bool = Field(description="Whether a peak was detected within tolerance")
    detected_frequency_hz: Optional[float] = Field(
        None, description="Actual peak frequency"
    )
    magnitude: Optional[float] = Field(
        None, description="Magnitude at detected frequency"
    )
    deviation_pct: Optional[float] = Field(
        None, description="Deviation from expected (%)"
    )
    harmonics_detected: list[dict[str, float]] = Field(
        default=[], description="Harmonics found"
    )
    evidence_strength: str = Field(
        description=(
            "Strength of the spectral evidence for this fault: 'high' "
            "(fundamental + >=2 harmonics), 'moderate' (fundamental, weaker "
            "harmonics), 'low' (harmonics only), or 'none'. Derived from "
            "detected peaks — not a probability."
        )
    )


class BearingFaultsSummary(BaseModel):
    """Summary of all expected-frequency checks for one bearing/machine."""

    signal_id: str = Field(description="Signal identifier used")
    bearing_id: Optional[str] = Field(
        None,
        description=(
            "Bearing designation (catalog route); None for the "
            "explicit-frequencies and explicit-geometry routes"
        ),
    )
    rpm: float = Field(description="Shaft speed (RPM)")
    shaft_frequency_hz: float = Field(description="Shaft frequency (Hz)")
    bearing_frequencies: dict[str, float] = Field(
        description="Expected frequencies checked (Hz), plus shaft_freq_hz"
    )
    fault_checks: list[BearingFaultCheckResult] = Field(
        description="Results for each checked frequency"
    )
    overall_assessment: str = Field(description="Summary assessment text")
    most_likely_fault: Optional[str] = Field(
        None, description="Most likely fault label if any"
    )
    most_likely_fault_canonical: Optional[
        Literal["outer_race", "inner_race", "ball", "cage"]
    ] = Field(
        None,
        description="Canonical form of most_likely_fault (None for arbitrary labels)",
    )
    source: Optional[str] = Field(
        None,
        description=(
            "Provenance of the expected frequencies: the catalog entry's "
            "source citation (bearing_id route), or a note for "
            "user-provided geometry/frequencies"
        ),
    )


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
    bearing_id: str = Field(description="The bearing designation that was searched for")
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
        "refused",
        description="Always 'refused' — discriminates from an assessed result",
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
    """Unified severity assessment result (U9 merge — ``assess_severity``).

    Zone boundaries from ISO 10816-3:2009 unless user-defined custom
    thresholds were supplied (see ``threshold_provenance``). Covers both
    input routes: a stored signal (``signal_id`` set) or a direct broadband
    RMS velocity reading (``signal_id`` None). Alert fields
    (``alert_level``, ``exceeded_threshold``) are derived from the zone.
    """

    status: Literal["assessed"] = Field(
        "assessed",
        description="Always 'assessed' — discriminates from a refused result",
    )
    signal_id: Optional[str] = Field(
        None,
        description="Signal identifier used (None for a direct rms_velocity_mm_s reading)",
    )
    rms_velocity_mm_s: float = Field(description="RMS velocity in mm/s")
    machine_group: int = Field(
        description="ISO 20816-3 machine group: 1 (large, >300 kW) or 2 (medium, 15-300 kW)"
    )
    support_type: str = Field(description="Support type: 'rigid' or 'flexible'")
    axis: str = Field("vertical", description="Measurement axis (informational)")
    zone: str = Field(description="ISO zone: A, B, C, or D")
    zone_description: str = Field(description="Zone description")
    severity_level: str = Field(
        description="Good, Acceptable, Unsatisfactory, or Unacceptable"
    )
    color_code: str = Field(description="green, yellow, orange, or red")
    boundaries: dict[str, float] = Field(
        description="Zone boundaries {AB, BC, CD} in mm/s (ISO or custom)"
    )
    frequency_range: str = Field(
        description="Actual evaluation band used (may be narrower than the ISO nominal 10-1000 Hz when fs limits it); 'not applicable' for direct RMS readings"
    )
    unit_conversion_performed: bool = Field(
        description="Whether acceleration-to-velocity conversion was done"
    )
    original_unit: Optional[str] = Field(
        None, description="Original signal unit before conversion"
    )
    operating_speed_rpm: Optional[float] = Field(
        None,
        description="Operating speed in RPM, when provided (selects the band's lower edge)",
    )
    machine_power_kw: Optional[float] = Field(
        None,
        description=(
            "Declared rated machine power in kW, when provided. Values "
            "below 15 kW are refused as out of ISO 20816-3 scope."
        ),
    )
    alert_level: Optional[Literal["none", "warning", "alarm", "danger"]] = Field(
        None,
        description=(
            "Alert level derived from the zone: A=none, B=warning, "
            "C=alarm, D=danger (filled automatically)"
        ),
    )
    exceeded_threshold: Optional[float] = Field(
        None,
        description=(
            "The boundary (mm/s) exceeded by the reading (None in zone A; "
            "filled automatically from zone + boundaries)"
        ),
    )
    threshold_provenance: str = Field(
        description="Provenance of the zone boundary values (ISO edition note, or custom-threshold note)"
    )

    @model_validator(mode="after")
    def _derive_alert_fields(self):
        """Derive alert_level/exceeded_threshold from zone + boundaries."""
        if self.alert_level is None:
            self.alert_level = {
                "A": "none",
                "B": "warning",
                "C": "alarm",
                "D": "danger",
            }.get(self.zone)
        if self.exceeded_threshold is None and self.zone in ("B", "C", "D"):
            self.exceeded_threshold = {
                "B": self.boundaries.get("AB"),
                "C": self.boundaries.get("BC"),
                "D": self.boundaries.get("CD"),
            }[self.zone]
        return self


#: Where one diagnostic parameter of ``diagnose_vibration`` came from: the
#: closed vocabulary of the values of ``DiagnosisResult.parameter_sources``.
DiagnosisParameterSource = Literal[
    "explicit",
    "measurement",
    "point",
    "default",
    "not_supported_fault_orders",
    "none",
]


class DiagnosisResult(BaseModel):
    """Full integrated diagnosis pipeline result."""

    signal_id: str = Field(description="Signal identifier used")
    rpm: float = Field(description="Machine speed (RPM)")
    bearing_id: Optional[str] = Field(None, description="Bearing used (if any)")
    machine_group: int = Field(
        description="ISO 20816-3 machine group used for severity: 1 (large) or 2 (medium)"
    )
    support_type: str = Field(
        description="Support type used for severity: 'rigid' or 'flexible'"
    )
    fft_summary: dict[str, Any] = Field(description="FFT key findings")
    psd_summary: dict[str, Any] = Field(description="PSD key findings")
    stft_summary: dict[str, Any] = Field(description="STFT key findings")
    bearing_faults: Optional[BearingFaultsSummary] = Field(
        None, description="Bearing fault results"
    )
    iso_severity: VibrationSeverityResult | ISOSeverityRefusal = Field(
        description=(
            "ISO severity assessment, or a structured refusal "
            "(status='refused' with reason + remedy) when the verdict cannot "
            "be produced honestly — e.g. undeclared signal unit or Nyquist "
            "below the ISO evaluation band. The other diagnosis blocks "
            "(spectral, bearing, anomaly) still run."
        )
    )
    anomaly_detection: Optional[dict[str, Any]] = Field(
        None, description="Anomaly detection results (health, ratio, score)"
    )
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
    parameter_sources: Optional[dict[str, DiagnosisParameterSource]] = Field(
        None,
        description=(
            "Origin of each diagnostic parameter, keyed rpm, bearing_id, "
            "machine_group and support_type. Precedence: 'explicit' (passed "
            "to the call) > 'measurement' (the rpm declared in the "
            "companion's \"measurement\" object) > 'point' (the current "
            "declaration of the measurement point in the asset ledger: "
            "nominal_rpm, bearing_id, machine_group, support_type) > "
            "'default' (the historical machine_group=2 / "
            "support_type='rigid'; rpm has no default and is refused "
            "instead). bearing_id only: 'none' (no bearing from any source, "
            "bearing block skipped) or 'not_supported_fault_orders' (the "
            "point declares fault_orders without a bearing_id: the bearing "
            "block was not computed because frequency sets are not "
            "supported by diagnose_vibration in this stage; use "
            "check_bearing_faults(frequencies=...))."
        ),
    )


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

    status: Literal[
        "estimated", "no_degradation_trend", "threshold_already_exceeded"
    ] = Field(
        description=(
            "'estimated' (RUL computed), 'no_degradation_trend' (no "
            "statistically significant trend toward the threshold — healthy "
            "outcome, no RUL number), or 'threshold_already_exceeded' (last "
            "measurement is at/above the failure threshold)."
        )
    )
    method: str = Field(
        description="Estimation method used: linear, exponential, or kalman"
    )
    feature_name: str = Field(description="Degradation indicator tracked (e.g. 'rms')")
    num_measurements: int = Field(description="Number of measurements in the series")
    observation_horizon: float = Field(
        description=(
            "Time span covered by the measurement series (last minus first "
            "timestamp), in time_unit. RUL estimates far beyond this horizon "
            "are extrapolations with low reliability."
        )
    )
    time_unit: str = Field(
        description="Unit of timestamps, observation_horizon, and rul"
    )
    failure_threshold: float = Field(
        description="Indicator value considered as failure"
    )
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
    message: str = Field(
        description="Human-readable explanation of the outcome and its caveats"
    )


class TrendAnalysisResult(BaseModel):
    """Within-recording feature trend — screening only, NOT a prognosis.

    Segments a single recording (seconds of data) and fits a trend on the
    per-segment feature values. Use it to screen whether a recording is
    stationary. It cannot estimate Remaining Useful Life: for RUL, collect
    repeated measurements over days/weeks and pass them to estimate_rul.
    """

    feature_name: str = Field(description="Feature analyzed")
    slope: float = Field(
        description="Trend slope in feature units per second (within the recording)"
    )
    intercept: float = Field(description="Trend intercept")
    r_squared: float = Field(
        description="R-squared goodness of fit of the linear trend"
    )
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
    # Onset detection (U9 merge: absorbed detect_signal_degradation_onset)
    onset_detected: bool = Field(
        description=(
            "Whether a degradation onset was detected after the baseline "
            "window (first value exceeding baseline mean + "
            "onset_threshold_sigma * std)"
        )
    )
    onset_segment_index: Optional[int] = Field(
        None,
        description=(
            "Segment index where degradation starts (always >= "
            "baseline_segments); None when no onset detected"
        ),
    )
    onset_time_s: Optional[float] = Field(
        None,
        description="Center time (s) of the onset segment within the recording",
    )
    onset_threshold_sigma: float = Field(
        description="Baseline standard deviations used as the onset trigger"
    )
    baseline_segments: int = Field(
        description=(
            "Number of leading segments used as the baseline window. Onset "
            "is only searched AFTER this window; degradation starting inside "
            "the baseline cannot be detected by this method."
        )
    )


# ============================================================================
# Phase 3 Models: Asset Health Ledger
# ============================================================================


class MeasurementPointDeclarationResult(BaseModel):
    """Outcome of ``declare_measurement_point``.

    The declaration is versioned per point in the local append-only asset
    ledger. A re-declaration identical to the current version appends
    nothing and reports the current version; a different one appends the
    next version and names the keys that changed. The count of recorded
    measurements whose snapshot no longer matches the declared context is
    reported with the exact re-processing call, never applied silently.
    """

    asset_id: str = Field(description="Asset the point belongs to (ledger id)")
    measurement_point_id: str = Field(description="The declared point (ledger id)")
    declaration_version: int = Field(
        description=(
            "Version of the point's declaration after this call (1-based, "
            "per point); unchanged when nothing was appended"
        )
    )
    appended: bool = Field(
        description=(
            "True when a new declaration version was appended to the ledger; "
            "False when the declaration equals the current version"
        )
    )
    changed: list[str] = Field(
        description=(
            "Declared keys whose value differs from the previous version "
            "(every key declared with a value for version 1; empty when "
            "nothing was appended)"
        )
    )
    previous_version: Optional[int] = Field(
        None,
        description="Version this declaration supersedes; None for a first declaration",
    )
    measurements_with_stale_context: int = Field(
        description=(
            "Recorded measurements of the point that lack a health snapshot "
            "computed with the current declared context and the current "
            "processing lineage; their existing snapshots are kept"
        )
    )
    remedy: Optional[str] = Field(
        None,
        description=(
            "The exact assess_asset_change(..., reprocess=True) call that "
            "recomputes the stale snapshots (bounded per call); None when "
            "nothing is stale"
        ),
    )
    declaration: dict[str, Any] = Field(
        description=(
            "The point declaration as recorded in the ledger: "
            "measurement_point_id, declaration_version, bearing_id, "
            "fault_orders, machine_group, support_type, machine_power_kw, "
            "expected_signal_unit, expected_sensor_id, expected_direction, "
            "nominal_rpm (the design speed of the point, distinct from the "
            "observed rpm of a measurement), declared_by, note, changed"
        )
    )
    event_id: Optional[str] = Field(
        None,
        description="Id of the appended ledger event; None when nothing was appended",
    )
    bearing_in_catalog: Optional[bool] = Field(
        None,
        description=(
            "Whether the declared bearing_id is in the verified bearing catalog "
            "(a bearing outside it leaves the bearing block of every snapshot "
            "missing); None when no bearing_id is declared"
        ),
    )
    message: str = Field(description="One-paragraph summary of the outcome")


class BaselineDeclarationResult(BaseModel):
    """Outcome of ``declare_healthy_baseline``.

    A declared baseline is the only reference the assessment reports as
    health_declared; it is attributed to the declarer named in
    ``declared_by`` (a user-supplied string), never to the server. An empty
    measurement list withdraws the active baseline, and later assessments
    fall back to the automatic window while naming the withdrawal.
    """

    asset_id: str = Field(description="Asset the point belongs to (ledger id)")
    measurement_point_id: str = Field(description="The point (ledger id)")
    baseline_id: str = Field(
        description=(
            "Deterministic id of this baseline declaration (hash of the point, "
            "the sorted measurement ids and the declaration instant)"
        )
    )
    measurement_ids: list[str] = Field(
        description=(
            "Members of the baseline in acquisition order; empty for a withdrawal"
        )
    )
    members: list[dict[str, Any]] = Field(
        description=(
            "One entry per member: measurement_id, declaration_version of the "
            "measurement and point_declaration_version it was validated against "
            "(a later re-declaration excludes the member at query time with a "
            "qualification, never silently)"
        )
    )
    declared_by: str = Field(
        description=(
            "Who declared the baseline, as given by the caller; quoted verbatim "
            "in every assessment that uses it"
        )
    )
    note: Optional[str] = Field(
        None, description="Free-text note of the declarer, as given; None when absent"
    )
    declared_at: str = Field(description="Declaration instant (ISO 8601, UTC)")
    superseded_baseline_id: Optional[str] = Field(
        None,
        description=(
            "Id of the baseline that was active before this call; None when "
            "the point had none"
        ),
    )
    withdrawn: bool = Field(
        description="True when this call withdrew the active baseline (empty list)"
    )
    event_id: str = Field(description="Id of the appended ledger event")
    message: str = Field(description="One-paragraph summary of the outcome")


class AssetHistoryResult(BaseModel):
    """Outcome of ``get_asset_history``: the index of the assets, the history
    of one asset, or a typed miss.

    ``status`` discriminates the three shapes: 'index' (no asset_id given:
    ``assets`` lists at most the configured number of assets, ``truncated``
    says whether more exist), 'found' (``asset`` holds the history read from
    that ledger alone) and 'not_found' (an unknown asset or point, with the
    known ids and a suggestion; never an exception).
    """

    status: Literal["index", "found", "not_found"] = Field(
        description="'index', 'found' or 'not_found' (see the class description)"
    )
    assets: list[dict[str, Any]] = Field(
        description=(
            "Index entries (status 'index' only, else empty): asset_id, points "
            "(measurement_point_id, measurement_count, first_acquired_at, "
            "last_acquired_at, latest_lineage, baseline_declared, "
            "declaration_version), point_count, measurement_count, "
            "first_acquired_at, last_acquired_at, reattributed_count, "
            "event_count, ledger_bytes, integrity (counters)"
        )
    )
    asset: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "History of the asset (status 'found' only): summary, measurements "
            "newest first (measurement_id, measurement_point_id, acquired_at, "
            "signal_id, location, declaration_version, rpm, direction, "
            "sensor_id, lineages, snapshot_count, indicators preview from the "
            "latest snapshot, comparability grade and codes against the point), "
            "measurement_count, truncated, point_declarations and baselines "
            "(current plus history per point), reattributed, integrity, "
            "event_count, ledger_bytes"
        ),
    )
    known_assets: list[str] = Field(
        description=(
            "Asset ids the ledger directory lists (the listed ones for the "
            "index, every id for a miss; empty for a found asset, whose ledger "
            "is the only one read)"
        )
    )
    known_points: list[str] = Field(
        description=(
            "Points of the asset (declared or named by its measurements) when "
            "the asset is known; empty otherwise"
        )
    )
    suggestion: Optional[str] = Field(
        None, description="Concrete next step on a miss; None otherwise"
    )
    truncated: bool = Field(
        description=(
            "True when the index holds fewer assets than exist, or the history "
            "fewer measurements than max_measurements would have to cover"
        )
    )
    message: str = Field(description="One-paragraph summary of the outcome")


class AssetChangeAssessment(BaseModel):
    """Outcome of ``assess_asset_change``: the change of one measurement point
    against its reference, or a typed reason why it cannot be assessed.

    ``status`` discriminates: 'assessed' fills reference, lineage, observed,
    derived, assessed, comparability and suggested_verification;
    'not_found' names the known assets and points with a suggestion;
    'insufficient_history' reports available versus required with a remedy;
    'processing_not_homogeneous' reports the snapshot lineages and the
    re-processing remedy. The reference is a declared baseline only when
    one was declared (health_declared True); otherwise it is the automatic
    window of the first comparable acquisitions, a relative comparison
    whose health is not declared.
    """

    status: Literal[
        "assessed",
        "not_found",
        "insufficient_history",
        "processing_not_homogeneous",
    ] = Field(description="Outcome discriminator (see the class description)")
    asset_id: str = Field(description="The assessed asset (ledger id)")
    measurement_point_id: str = Field(description="The assessed point (ledger id)")
    reference: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "The reference used: kind ('automatic_window' or "
            "'declared_baseline'), health_declared (True only for a declared "
            "baseline), message (cites declared_by, declared_at and note of a "
            "baseline verbatim), measurement_ids, count, acquired_from, "
            "acquired_to, provisional, statistics_quality ('relative_only', "
            "'provisional', 'full'), qualification_codes of the reference "
            "slots, outside_window, baseline, withdrawn_baseline, "
            "excluded_inside_span; None for a miss"
        ),
    )
    lineage: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Processing lineage the assessment used: processing_id, "
            "algorithm_version, covered slots, candidates per lineage, "
            "current_processing_id, is_current, missing_for_current, "
            "stale_context; None unless assessed"
        ),
    )
    observed: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Observed values: reference_statistics per indicator (mean, std, "
            "n, band, unit), latest values, latest_measurement_id, "
            "latest_acquired_at, evidence_presence per bearing label over the "
            "last K acquisitions, indicators_unavailable with reasons, "
            "acquisitions_assessed, slots_assessed, measurement_ids_assessed; "
            "None unless assessed"
        ),
    )
    derived: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Derived comparisons: deltas per indicator against the reference "
            "mean, exceedance_runs, drift regressions, iso_change (change "
            "against 25 percent of the ISO 20816-3 B/C boundary when group and "
            "support are known) and per_indicator classifications with their "
            "criterion; None unless assessed"
        ),
    )
    assessed: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "The verdict: classification ('no_change', 'isolated_episode', "
            "'unconfirmed_single_acquisition', 'persistent_change'), direction "
            "('increase' or 'decrease'), sudden, criterion, "
            "indicators_driving, onset_measurement_id, onset_acquired_at, "
            "onset_coincides_with (qualifications of the onset acquisition) "
            "and evidence per bearing label; None unless assessed"
        ),
    )
    comparability: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Comparability of the point's acquisitions: counts per grade, "
            "qualifications (code, count, detail), excluded measurements with "
            "reasons and collapsed duplicates; None for a miss"
        ),
    )
    suggested_verification: Optional[str] = Field(
        None,
        description=(
            "At most ONE verification sentence the evidence calls for; None "
            "when no verification is needed or the point was not assessed"
        ),
    )
    remedy: Optional[str] = Field(
        None,
        description=(
            "Concrete action for 'insufficient_history' or "
            "'processing_not_homogeneous' (the exact re-processing call); "
            "None otherwise"
        ),
    )
    suggestion: Optional[str] = Field(
        None, description="Concrete next step on 'not_found'; None otherwise"
    )
    known_assets: list[str] = Field(
        description="Asset ids the ledger directory lists (for a miss); else empty"
    )
    known_points: list[str] = Field(
        description="Points of a known asset when the point is unknown; else empty"
    )
    available: Optional[int] = Field(
        None,
        description="Usable acquisition slots available ('insufficient_history')",
    )
    required: Optional[int] = Field(
        None, description="Slots required ('insufficient_history')"
    )
    lineages: Optional[dict[str, int]] = Field(
        None,
        description=(
            "Covered evaluated slots per processing lineage "
            "('processing_not_homogeneous')"
        ),
    )
    evaluated_slots: Optional[int] = Field(
        None,
        description="Evaluated slots, reference plus post-reference ('processing_not_homogeneous')",
    )
    missing_for_current: Optional[int] = Field(
        None,
        description=(
            "Evaluated slots without a snapshot on the current lineage "
            "('processing_not_homogeneous')"
        ),
    )
    current_processing_id: Optional[str] = Field(
        None,
        description="The current processing lineage ('processing_not_homogeneous')",
    )
    reprocess: Optional[dict[str, Any]] = Field(
        None,
        description=(
            "Outcome of the re-processing run before the assessment when "
            "reprocess=True: processing_id, stale, reprocessed, "
            "not_reprocessable, up_to_date, remaining, results per attempted "
            "measurement, next_call (the exact call to continue, or None) and "
            "message; None when reprocess was False"
        ),
    )
    message: str = Field(description="One-paragraph summary of the outcome")
