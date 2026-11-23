"""
Pure feature extraction functions for vibration signals.

No MCP dependency, no file I/O. Takes numpy arrays, returns dicts.
Suitable for direct testing and reuse across MCP tools and pipelines.

ISO 13374 Block 2 — Data Manipulation (feature extraction).
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import kurtosis, skew, entropy

from ..signal_validation import InvalidInputError

logger = logging.getLogger(__name__)


def extract_time_domain_features(segment: np.ndarray) -> dict[str, float]:
    """
    Extract comprehensive time-domain features from a signal segment.

    Computes 17 statistical and shape-based features commonly used
    in vibration-based condition monitoring and anomaly detection.

    Args:
        segment: 1D numpy array with signal segment.

    Returns:
        Dictionary with 17 time-domain features:
            mean, std, var, mean_abs, rms, max, min, range,
            skewness, kurtosis, shape_factor, crest_factor,
            impulse_factor, clearance_factor, power, entropy,
            zero_crossing_rate.
    """
    # Basic statistics
    mean_val = float(np.mean(segment))
    std_val = float(np.std(segment))
    var_val = float(np.var(segment))
    mean_abs_val = float(np.mean(np.abs(segment)))

    # RMS (Root Mean Square)
    rms_val = float(np.sqrt(np.mean(segment**2)))

    # Peak values
    max_val = float(np.max(segment))
    min_val = float(np.min(segment))
    range_val = max_val - min_val

    # Shape indicators
    skewness_val = float(skew(segment))
    kurtosis_val = float(kurtosis(segment, fisher=True))

    # Avoid division by zero
    if rms_val == 0:
        rms_val = 1e-10
    if mean_abs_val == 0:
        mean_abs_val = 1e-10

    # Dimensionless indicators
    shape_factor_val = rms_val / mean_abs_val if mean_abs_val > 0 else 0.0
    crest_factor_val = float(np.max(np.abs(segment))) / rms_val if rms_val > 0 else 0.0
    impulse_factor_val = (
        float(np.max(np.abs(segment))) / mean_abs_val if mean_abs_val > 0 else 0.0
    )
    sqrt_mean = float(np.mean(np.sqrt(np.abs(segment))))
    clearance_factor_val = (
        float(np.max(np.abs(segment))) / (sqrt_mean**2) if sqrt_mean > 0 else 0.0
    )

    # Energy and entropy
    power_val = float(np.mean(segment**2))

    # Entropy (probability distribution of signal amplitudes)
    hist, _ = np.histogram(segment, bins=50, density=True)
    hist = hist + 1e-10  # Avoid log(0)
    entropy_val = float(entropy(hist))

    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(segment)))[0]
    zero_crossing_rate_val = float(len(zero_crossings) / len(segment))

    return {
        "mean": mean_val,
        "std": std_val,
        "var": var_val,
        "mean_abs": mean_abs_val,
        "rms": rms_val,
        "max": max_val,
        "min": min_val,
        "range": range_val,
        "skewness": skewness_val,
        "kurtosis": kurtosis_val,
        "shape_factor": shape_factor_val,
        "crest_factor": crest_factor_val,
        "impulse_factor": impulse_factor_val,
        "clearance_factor": clearance_factor_val,
        "power": power_val,
        "entropy": entropy_val,
        "zero_crossing_rate": zero_crossing_rate_val,
    }


class SegmentPlan(NamedTuple):
    """How a signal is cut into the windows an anomaly model scores."""

    #: Samples per segment, at least one.
    length: int
    #: Samples between successive segment starts, at least one.
    hop: int
    #: Seconds between the start times reported for successive segments.
    hop_s: float


#: Remedy for a segment of no samples when the caller chose its duration.
SEGMENT_DURATION_REMEDY = (
    "Use a segment_duration of at least one sample period, {period:g} s at this rate"
)
#: Remedy for a segment of no samples when its duration comes from a model.
MODEL_SEGMENT_REMEDY = (
    "Retrain the model with a segment_duration of at least one sample period, "
    "{period:g} s at this rate"
)


def plan_segments(
    sampling_rate: float,
    segment_duration: float,
    overlap_ratio: float,
    *,
    remedy: str = SEGMENT_DURATION_REMEDY,
) -> SegmentPlan:
    """The one segmentation rule of the anomaly tools.

    Training, ``predict_anomalies``, the diagnosis pipeline and the PCA report
    all cut their windows with it, so one model, one signal and one rate give
    the same windows everywhere. A segment holds
    ``int(segment_duration * sampling_rate)`` samples and successive segments
    start ``int(length * (1 - overlap_ratio))`` samples apart. When that leaves
    less than one sample between starts (a ratio close to 1, or a ratio of 1 or
    above in a model saved before the ratio was checked), segments advance one
    sample and their start times are reported one sample period apart;
    otherwise start times advance ``segment_duration * (1 - overlap_ratio)``
    seconds per segment.

    Args:
        sampling_rate: Sampling frequency in Hz.
        segment_duration: Duration of each segment in seconds.
        overlap_ratio: Overlap between successive segments, as a fraction of
            the segment length.
        remedy: What the caller should do about a segment of no samples, a
            sentence without its final period; ``{period}`` is replaced by
            one sample period in seconds.

    Returns:
        The segment length, the hop in samples and the hop in seconds.

    Raises:
        InvalidInputError: If a segment would hold no samples at this rate.
    """
    length = int(segment_duration * sampling_rate)
    if length < 1:
        raise InvalidInputError(
            f"A segment of {segment_duration:g} s holds no samples at "
            f"{sampling_rate:g} Hz, so no features can be computed from it. "
            f"{remedy.format(period=1.0 / sampling_rate)}."
        )
    hop = int(length * (1 - overlap_ratio))
    hop_s = segment_duration * (1 - overlap_ratio)
    if hop < 1:
        hop = 1
        hop_s = 1.0 / sampling_rate
    return SegmentPlan(length=length, hop=hop, hop_s=hop_s)


def segment_features(signal_data: np.ndarray, plan: SegmentPlan) -> list[dict]:
    """Time-domain features of every segment *plan* cuts from *signal_data*.

    Args:
        signal_data: Raw signal array (1D).
        plan: The segmentation from :func:`plan_segments`.

    Returns:
        List of feature dictionaries, one per segment; empty when the signal
        is shorter than one segment.
    """
    return [
        extract_time_domain_features(signal_data[start : start + plan.length])
        for start in range(0, len(signal_data) - plan.length + 1, plan.hop)
    ]


def segment_and_extract_features(
    signal_data: np.ndarray,
    sampling_rate: float,
    segment_duration: float,
    overlap_ratio: float,
) -> list[dict]:
    """
    Segment a signal and extract time-domain features from each segment.

    Args:
        signal_data: Raw signal array (1D).
        sampling_rate: Sampling frequency in Hz.
        segment_duration: Duration of each segment in seconds.
        overlap_ratio: Overlap ratio between segments, at least 0 and below 1.

    Returns:
        List of feature dictionaries, one per segment.

    Raises:
        InvalidInputError: If a segment would hold no samples at this rate.
    """
    return segment_features(
        signal_data, plan_segments(sampling_rate, segment_duration, overlap_ratio)
    )


def resolve_sampling_rate(
    signal_file: str,
    sampling_rate: Optional[float],
    strict: bool = True,
    metadata_resolver=None,
) -> Optional[float]:
    """
    Resolve sampling rate for a signal file.

    Priority: provided sampling_rate > metadata > error/None.

    Args:
        signal_file: Signal filename (used to locate metadata).
        sampling_rate: User-provided sampling rate (None = auto-detect).
        strict: If True, raise on missing rate; if False, return None.
        metadata_resolver: Callable(signal_file) -> Path returning the
            metadata file path. If None, uses the default from
            signal_acquisition.loaders.

    Returns:
        Resolved sampling rate, or None if strict=False and not found.

    Raises:
        ValueError: If strict=True and no sampling rate can be found.
    """
    if sampling_rate is not None:
        return sampling_rate

    if metadata_resolver is None:
        from ..signal_acquisition.loaders import get_metadata_path

        metadata_resolver = get_metadata_path

    metadata_path = metadata_resolver(signal_file)
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            rate = metadata.get("sampling_rate")
            if rate is not None:
                return rate

    if strict:
        raise ValueError(
            f"No sampling rate found for {signal_file}. "
            f"Please provide sampling_rate parameter or create "
            f"{Path(signal_file).stem}_metadata.json with 'sampling_rate' field."
        )
    return None
