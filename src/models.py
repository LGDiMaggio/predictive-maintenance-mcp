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


#: Closed vocabularies of the bearing evidence layer.
BearingEvidenceState = Literal[
    "not_testable", "absent", "unconfirmed", "shared", "supported"
]
BearingEvidenceStateReason = Literal[
    "search_window_too_narrow",
    "noise_floor_not_estimable",
    "outside_usable_envelope_band",
    "above_frequency_range",
    "no_significant_line",
    "line_at_shaft_order",
    "line_shared",
    "no_corroboration",
    "corroborated",
]
BearingCheckOutcomeName = Literal[
    "no_indication", "inconclusive", "indicated", "several_types"
]


class BearingLineTest(BaseModel):
    """How the line of one hypothesis was tested against the noise floor."""

    search_bins: int = Field(description="Frequency bins searched for the line (M)")
    reference_bins: int = Field(
        description="Reference bins on both sides of the search window (N)"
    )
    reference_rank: int = Field(
        description="Rank k of the reference power taken as the noise floor (the median rank)"
    )
    false_alarm_probability: float = Field(
        description=(
            "Nominal false-alarm budget allocated to this line's test over all its "
            "search bins together: the check's budget split evenly over the "
            "hypotheses it could test. It is not a measured probability for this "
            "record. The per-bin value is this divided by search_bins."
        )
    )


class BearingCorroboratingLine(BaseModel):
    """One further line the kinematics of a detected line predict."""

    kind: Literal["harmonic", "sideband", "subharmonic"] = Field(
        description=(
            "harmonic of the detected line; sideband at the shaft rate (inner "
            "race) or the cage rate (rolling element); subharmonic, the 1 x BSF "
            "line of the rolling element"
        )
    )
    harmonic: Optional[int] = Field(
        None, description="Harmonic number for a harmonic; None otherwise"
    )
    expected_hz: float = Field(
        description="Centre of the search window, anchored at the detected line (Hz)"
    )
    peak_hz: Optional[float] = Field(
        None,
        description="Strongest bin in the window (Hz); None when the window could not be tested",
    )
    level_db: Optional[float] = Field(
        None, description="Power of that bin over the local noise floor (dB)"
    )
    threshold_db: Optional[float] = Field(
        None, description="Level the line had to exceed (dB over the noise floor)"
    )
    significant: bool = Field(
        description="Whether the line is a local maximum above its threshold"
    )
    counted: bool = Field(
        description=(
            "Whether the line counts as corroboration: significant, not at a "
            "shaft order and not shared with another hypothesis"
        )
    )
    excluded_by: Optional[Literal["shaft_order", "shared"]] = Field(
        None,
        description="Why a significant line is not counted, if it is not",
    )
    untestable_reason: Optional[
        Literal[
            "search_window_too_narrow",
            "noise_floor_not_estimable",
            "outside_usable_envelope_band",
            "above_frequency_range",
        ]
    ] = Field(
        None, description="Why the line could not be tested; None when it was tested"
    )


class BearingCandidate(BaseModel):
    """A hypothesis with a significant line of its own and at least one counted
    corroborating line.

    A significant corroborating line that is shared with another hypothesis or
    lies at a shaft order is not counted and does not make a candidate.
    Evidence in one envelope spectrum, not a confirmed fault.
    """

    fault_type: str = Field(description="Checked label")
    fault_type_canonical: Optional[
        Literal["outer_race", "inner_race", "ball", "cage"]
    ] = Field(None, description="Canonical fault type (None for arbitrary labels)")
    line_frequency_hz: float = Field(description="Frequency of the line tested (Hz)")
    line_peak_hz: float = Field(description="Frequency of the detected line (Hz)")
    level_db: float = Field(
        description="Power of the detected line over the local noise floor (dB)"
    )


class BearingEvidencePolicy(BaseModel):
    """The statistical policy behind the evidence fields, with its assumptions."""

    method: str = Field(description="Name of the line test")
    version: int = Field(description="Version of the evidence policy")
    false_alarm_probability_per_check: float = Field(
        description=(
            "Nominal family-wise false-alarm budget of one complete check, "
            "allocated over its tested hypotheses and their search bins by "
            "Bonferroni. Under the assumptions listed, the probability that the "
            "check finds a significant line at a tested frequency in a record with "
            "no periodic content there is at most this budget; the realized "
            "probability is not asserted to equal it. It is not the false-alarm "
            "rate or the specificity of the system, not the probability of an "
            "indicated outcome, and it applies to one check, not to repeated "
            "checks of one machine."
        )
    )
    false_alarm_probability_meaning: str = Field(
        description="What the per-check budget means, and what it does not"
    )
    false_alarm_probability_per_corroborating_line: float = Field(
        description=(
            "Nominal false-alarm budget of each corroborating line, allocated over "
            "the bins of its window by Bonferroni"
        )
    )
    search_window_pct: float = Field(
        description="Half-width of each fundamental search window (% of its frequency)"
    )
    harmonics_tested: int = Field(
        description=(
            "Number of harmonics tested above each detected line, starting at 2x "
            "(3 means 2x, 3x and 4x)"
        )
    )
    min_search_bins: int = Field(
        description="Fewest bins a search window needs to be testable"
    )
    min_reference_bins: int = Field(
        description="Fewest reference bins the noise floor needs"
    )
    reference_width_fraction: float = Field(
        description="Reference width on each side, as a fraction of the line frequency"
    )
    min_reference_bins_per_side: int = Field(
        description="Least reference width on each side (bins), where the spectrum allows"
    )
    guard_bins: int = Field(
        description="Bins left out between a search window and its reference"
    )
    usable_band_fraction: float = Field(
        description=(
            "Lines and reference bins are used only below this fraction of the "
            "envelope band width (band high minus band low), which bounds the line "
            "frequencies the envelope can carry"
        )
    )
    lowest_reference_bin: int = Field(
        description=(
            "Reference bins start this many bins above 0 Hz, above the leakage of "
            "the removed envelope mean"
        )
    )
    narrow_window_fraction: float = Field(
        description=(
            "Half-width of corroborating and shaft-order windows as a fraction "
            "of their frequency"
        )
    )
    min_narrow_window_bins: int = Field(
        description="Least half-width of corroborating and shaft-order windows (bins)"
    )
    assumptions: list[str] = Field(
        description="Statistical assumptions the nominal budgets rest on"
    )
    exclusions: list[str] = Field(description="What the nominal budgets do not cover")


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
    state: Optional[BearingEvidenceState] = Field(
        None,
        description=(
            "Significance-tested evidence for this hypothesis on the check's "
            "envelope spectrum: not_testable (the record cannot test its line), "
            "absent (no eligible peak, a local maximum strictly inside the search "
            "window, above threshold_db), "
            "unconfirmed (a line without a counted corroborating line, or a line "
            "at a shaft order), shared (another tested hypothesis also predicts a "
            "line where this line lies), supported (a significant line of its own "
            "with at least one counted corroborating line its kinematics predict). "
            "detected and "
            "evidence_strength come from an older tolerance match without "
            "false-alarm control and can disagree with state."
        ),
    )
    state_reason: Optional[BearingEvidenceStateReason] = Field(
        None, description="Why the hypothesis is in its state"
    )
    line_frequency_hz: Optional[float] = Field(
        None,
        description=(
            "Frequency of the line tested (Hz): the expected frequency, or twice "
            "the ball spin frequency for the rolling element (detected_frequency_hz "
            "is still matched near the ball spin frequency itself)"
        ),
    )
    line_peak_hz: Optional[float] = Field(
        None,
        description=(
            "Strongest bin in the search window of the line tested (Hz), whether "
            "or not it is significant; None when not testable"
        ),
    )
    line_amplitude: Optional[float] = Field(
        None, description="Envelope amplitude at line_peak_hz; None when not testable"
    )
    level_db: Optional[float] = Field(
        None,
        description=(
            "Power at line_peak_hz over the local noise floor (dB), reported "
            "whether or not the line is significant: compare it with threshold_db. "
            "It can exceed threshold_db while state is absent when the strongest "
            "bin lies on the edge of the search window, where a line outside the "
            "window leaks in. None when not testable."
        ),
    )
    threshold_db: Optional[float] = Field(
        None,
        description=(
            "Level the line had to exceed (dB over the local noise floor) for its "
            "share of the check's false-alarm budget; None when not testable"
        ),
    )
    line_test: Optional[BearingLineTest] = Field(
        None, description="How the line was tested; None when not testable"
    )
    corroboration: list[BearingCorroboratingLine] = Field(
        default_factory=list,
        description=(
            "Further lines tested for a significant line, anchored at it; empty "
            "when no significant line was found. Lines of one spectrum are "
            "correlated, not independent evidence."
        ),
    )
    shared_with: list[str] = Field(
        default_factory=list,
        description=(
            "Labels of the other detected hypotheses whose search window, or the "
            "window of a harmonic, sideband or 1 x BSF line they predict, holds "
            "this hypothesis's detected line"
        ),
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
    outcome: Optional[BearingCheckOutcomeName] = Field(
        None,
        description=(
            "Significance-tested outcome of the check on one envelope spectrum: "
            "indicated (exactly one hypothesis supported: the evidence is "
            "consistent with that defect, not a confirmation of it), "
            "several_types (two or more supported, none singled out), "
            "inconclusive (a line without a counted corroborating line, at a "
            "shaft order or shared, or nothing testable), no_indication (no "
            "eligible peak at a tested frequency satisfies the detection criteria, "
            "which is not proof that no defect exists). most_likely_fault, "
            "overall_assessment "
            "and each check's detected and evidence_strength come from an older "
            "tolerance match without false-alarm control and can disagree with "
            "outcome."
        ),
    )
    outcome_reason: Optional[str] = Field(
        None,
        description=(
            "Why the check reached its outcome, naming every hypothesis the record "
            "could not test"
        ),
    )
    next_step: Optional[str] = Field(
        None,
        description=(
            "What to do next for this outcome; None when the check suggests no "
            "step (every hypothesis was tested and none stands out), which does "
            "not mean the bearing is free of defects"
        ),
    )
    candidates: list[BearingCandidate] = Field(
        default_factory=list,
        description=(
            "Supported hypotheses, listed by level over the noise floor; the order "
            "does not rank which type is more likely or more severe. One entry for "
            "indicated, two or more for several_types, empty otherwise."
        ),
    )
    evidence_policy: Optional[BearingEvidencePolicy] = Field(
        None,
        description="Statistical policy and assumptions behind the evidence fields",
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


class BearingCheckOutcome(BaseModel):
    """Whether the bearing check of a diagnosis ran, and why not when it did not.

    ``bearing_faults`` is None both when no bearing check was requested and
    when one was requested but could not run; this block tells the two
    apart, so a check that never ran cannot read as a clean one.
    """

    status: Literal["assessed", "not_requested", "not_computed"] = Field(
        description=(
            "'assessed' (the check ran: see bearing_faults), 'not_requested' "
            "(no bearing information from any source: no bearing_id and no "
            "fault_orders declared on the measurement point) or "
            "'not_computed' (a bearing check was requested, by a bearing_id "
            "or by fault_orders declared on the measurement point, but could "
            "not run: see reason_code, reason and remedy). Only 'assessed' "
            "says anything about the bearing."
        )
    )
    reason_code: Optional[
        Literal[
            "not_in_catalog", "envelope_band_unusable", "fault_orders_not_supported"
        ]
    ] = Field(
        None,
        description=(
            "Why the check was not computed (status 'not_computed' only): "
            "'not_in_catalog' (the designation is not in the bearing "
            "catalog), 'envelope_band_unusable' (the sampling rate leaves the "
            "envelope band empty) or 'fault_orders_not_supported' (the "
            "measurement point declares fault_orders, which this tool does "
            "not check)"
        ),
    )
    reason: Optional[str] = Field(
        None,
        description="Why the check was not computed (status 'not_computed' only)",
    )
    remedy: Optional[str] = Field(
        None,
        description="How to check the bearing instead (status 'not_computed' only)",
    )


class DiagnosisResult(BaseModel):
    """Full integrated diagnosis pipeline result."""

    signal_id: str = Field(description="Signal identifier used")
    rpm: float = Field(description="Machine speed (RPM)")
    bearing_id: Optional[str] = Field(
        None,
        description=(
            "Bearing designation given (explicit or from the point), whether "
            "or not its check ran; None without one, also for a point that "
            "declares fault_orders only (see bearing_check)"
        ),
    )
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
        None,
        description=(
            "Bearing fault results; None when the check did not run (see "
            "bearing_check)"
        ),
    )
    bearing_check: BearingCheckOutcome = Field(
        description=(
            "Whether the bearing check ran. A bearing_faults of None alone "
            "does not say whether a check was requested: status "
            "'not_computed' means one was (a bearing_id, or fault_orders "
            "declared on the measurement point) and it did not run "
            "(reason_code, reason, remedy)."
        )
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
            "alone and NOT a probability. 'none' means the analyses that ran "
            "found no fault evidence, which does not establish that the "
            "machine is healthy; a bearing check that was not computed "
            "(bearing_check.status 'not_computed') adds nothing and says "
            "nothing about the bearing."
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

#: How a caller follows re-processing, stated by every output that names it.
_REPROCESS_STOP_RULE = (
    "A call without reprocess_cursor starts a new re-processing sequence, "
    "including the call a message, a remedy or a declaration names: start one "
    "only when no sequence of the point has ended in the conversation, after "
    "the user took an action a sequence listed, after a load or a declaration "
    "changed the point, once after an environment fix, or when the user asks "
    "to list the blocked measurements again; a named call is an attempt, not a "
    "promise that it succeeds. While the reprocess block gives a next_call, "
    "make exactly that call. When it gives none, read sequence_state: "
    "'complete' (every stale measurement that represents an acquisition slot "
    "has a current snapshot) or 'exhausted' (the measurements under "
    "needs_action stay stale until their action is taken; ask the user before "
    "any action that is theirs). A new sequence attempts again the failures "
    "only an attempt finds (a file rewritten with the same size, a file that "
    "no longer decodes), at most 10 attempts, 500 file checks and 50 failures "
    "per call."
)


class MeasurementPointDeclarationResult(BaseModel):
    """Outcome of ``declare_measurement_point``.

    The declaration is versioned per point in the local append-only asset
    ledger. A re-declaration identical to the current version appends
    nothing and reports the current version; a different one appends the
    next version and names the keys that changed. The count of recorded
    measurements whose snapshot no longer matches the declared context is
    reported, never applied silently, with the exact re-processing call when
    a call can act on one of them. A call without reprocess_cursor starts a
    new re-processing sequence, and a named call is an attempt, not a
    promise that it succeeds; while the reprocess block gives a next_call,
    make exactly that call; when it gives none, read sequence_state and
    report the measurements under needs_action; a new sequence attempts
    again the failures only an attempt finds.
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
            "The exact assess_asset_change(..., reprocess=True) call to attempt "
            "the stale snapshots, bounded per call, when a stale measurement "
            "passes the checks re-processing reads from the ledger and is not a "
            "recording collapsed into another acquisition slot; None otherwise, "
            "and the message counts the stale measurements by reason. "
            + _REPROCESS_STOP_RULE
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
            "'processing_not_homogeneous': for the latter, the exact "
            "re-processing call while a call can act on an evaluated "
            "acquisition missing the current lineage, else the actions that "
            "close the gap with counts and measurement ids; None otherwise. "
            + _REPROCESS_STOP_RULE
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
            "Outcome of the re-processing call run before the assessment when "
            "reprocess=True, None when reprocess was False: processing_id; the "
            "counts stale, reprocessed, not_reprocessable, up_to_date, "
            "remaining, unresolved, superseded and earlier_in_sequence; "
            "sequence_state ('continue', 'complete' or 'exhausted'); groups "
            "(the point's measurements in six groups, each with its count, at "
            "most 50 measurement_ids and truncated); results (each measurement "
            "attempted or found not reprocessable in this call: outcome, "
            "reason_code, reason, action_id, location, location_used, "
            "attempted, snapshot_id); needs_action (one entry per reason_code "
            "with its exact count, at most 50 ids in all, and actions: a list "
            "of {action_id, action} that gives once the text of each action_id "
            "a failure of this response references, or the plain action of the "
            "code when none of its failures is detailed here); earlier_failures "
            "(each with reason_code, action_id and "
            "location), earlier_by_reason and earlier_not_detailed (what earlier "
            "calls of the sequence found; the reasons of failures not detailed "
            "are in the responses that found them, or one new sequence lists "
            "them again); cursor (the token received), restarted (why a token "
            "was not used), next_cursor, next_call (the exact call to make "
            "next, or None once the sequence is complete or exhausted) and "
            "message. " + _REPROCESS_STOP_RULE
        ),
    )
    message: str = Field(description="One-paragraph summary of the outcome")
