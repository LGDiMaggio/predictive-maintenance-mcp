"""MCP tools for prognostic assessment (ISO 13374 Block 5).

Exposes RUL estimation, trend analysis, and degradation onset detection
as MCP tools that operate on signal files in the data directory.
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context

from ..config import DATA_DIR
from ..signal_loader import load_signal_data, get_metadata_path
from ..signal_processing.features import extract_time_domain_features
from ..prognostics import (
    estimate_rul_linear,
    estimate_rul_exponential,
    analyze_trend,
    detect_degradation_onset,
)
from ..models import (
    RULEstimationResult,
    TrendAnalysisResult,
    DegradationOnsetResult,
)
from ._utils import sanitize_filename

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_feature_series(
    signal_file: str,
    feature_name: str,
    sampling_rate: float,
    segment_duration: float,
    overlap_ratio: float,
) -> list[float]:
    """Load a signal, segment it, extract features, and return one feature series."""
    filepath = DATA_DIR / signal_file
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {signal_file}")

    signal_data = load_signal_data(signal_file)
    if signal_data is None:
        raise ValueError(f"Could not load signal data from: {signal_file}")

    segment_length = int(segment_duration * sampling_rate)
    hop_length = int(segment_length * (1 - overlap_ratio))

    if segment_length < 1:
        raise ValueError("segment_duration * sampling_rate must be >= 1")
    if hop_length < 1:
        hop_length = 1

    segments: list[np.ndarray] = []
    for start in range(0, len(signal_data) - segment_length + 1, hop_length):
        segments.append(signal_data[start : start + segment_length])

    if not segments:
        raise ValueError(
            f"Signal too short ({len(signal_data)} samples) for "
            f"segment_duration={segment_duration}s at {sampling_rate} Hz"
        )

    feature_values: list[float] = []
    for seg in segments:
        feats = extract_time_domain_features(seg)
        if feature_name not in feats:
            available = ", ".join(sorted(feats.keys()))
            raise ValueError(
                f"Unknown feature '{feature_name}'. Available: {available}"
            )
        feature_values.append(feats[feature_name])

    return feature_values


def _resolve_sampling_rate(signal_file: str, provided: Optional[float]) -> float:
    """Return provided rate, or auto-detect from metadata, or fall back to 10 kHz."""
    if provided is not None:
        return provided
    metadata_path = get_metadata_path(signal_file)
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            return metadata.get("sampling_rate", 10000.0)
    return 10000.0


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(mcp: FastMCP) -> None:
    """Register prognostics MCP tools on *mcp*."""

    @mcp.tool()
    async def estimate_rul(
        ctx: Context,
        signal_file: str,
        failure_threshold: float,
        method: str = "linear",
        feature_name: str = "rms",
        sampling_interval: float = 1.0,
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
    ) -> RULEstimationResult:
        """Estimate Remaining Useful Life from a degradation signal.

        Segments the signal, extracts a feature series (e.g. RMS over time),
        and fits a degradation curve to estimate when *failure_threshold*
        will be reached.

        Args:
            ctx: MCP context for user communication.
            signal_file: CSV signal file in the data directory.
            failure_threshold: Feature value at which the component is
                considered failed.
            method: Estimation method — "linear", "exponential", "weibull",
                or "kalman" (default: "linear").
            feature_name: Time-domain feature to track (default: "rms").
            sampling_interval: Time between successive segments in the
                units you want the RUL expressed in (default: 1.0).
            sampling_rate: Signal sampling rate in Hz (auto-detect if None).
            segment_duration: Duration of each segment in seconds.
            overlap_ratio: Overlap between segments (0-1).

        Returns:
            RULEstimationResult with estimated RUL and confidence.
        """
        sr = _resolve_sampling_rate(signal_file, sampling_rate)
        await ctx.info(f"Extracting '{feature_name}' series from {signal_file} at {sr} Hz ...")

        feature_series = _extract_feature_series(
            signal_file, feature_name, sr, segment_duration, overlap_ratio,
        )
        await ctx.info(f"Extracted {len(feature_series)} segments, estimating RUL ({method}) ...")

        result = None
        extra: dict = {}

        if method == "linear":
            result = estimate_rul_linear(feature_series, failure_threshold, sampling_interval)
        elif method == "exponential":
            result = estimate_rul_exponential(feature_series, failure_threshold, sampling_interval)
        elif method == "weibull":
            try:
                from ..prognostics.rul_estimator import estimate_rul_weibull  # type: ignore[attr-defined]
                result = estimate_rul_weibull(feature_series, failure_threshold, sampling_interval)
                if result:
                    extra["shape"] = result.get("shape")
                    extra["scale"] = result.get("scale")
            except (ImportError, AttributeError):
                raise ValueError(
                    "Weibull RUL estimation is not available. "
                    "Use 'linear' or 'exponential' instead."
                )
        elif method == "kalman":
            try:
                from ..prognostics.kalman_rul import estimate_rul_kalman
                result = estimate_rul_kalman(feature_series, failure_threshold, sampling_interval)
                if result:
                    extra["confidence_interval"] = result.get("confidence_interval")
                    extra["estimated_rate"] = result.get("estimated_rate")
            except (ImportError, AttributeError):
                raise ValueError(
                    "Kalman RUL estimation is not available. "
                    "Use 'linear' or 'exponential' instead."
                )
        else:
            raise ValueError(f"Unknown method '{method}'. Use: linear, exponential, weibull, kalman.")

        if result is None:
            return RULEstimationResult(
                rul=float("inf"),
                confidence=0.0,
                method=method,
                **{k: v for k, v in extra.items() if v is not None},
            )

        return RULEstimationResult(
            rul=result["rul"],
            confidence=result["confidence"],
            method=result["method"],
            **{k: v for k, v in extra.items() if v is not None},
        )

    @mcp.tool()
    async def analyze_signal_trend(
        ctx: Context,
        signal_file: str,
        feature_name: str = "rms",
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
    ) -> TrendAnalysisResult:
        """Analyze the trend of a feature extracted from a signal over time.

        Segments the signal, extracts the requested feature per segment,
        and fits a linear trend to detect increasing/decreasing behavior.

        Args:
            ctx: MCP context for user communication.
            signal_file: CSV signal file in the data directory.
            feature_name: Time-domain feature to analyze (default: "rms").
            sampling_rate: Signal sampling rate in Hz (auto-detect if None).
            segment_duration: Duration of each segment in seconds.
            overlap_ratio: Overlap between segments (0-1).

        Returns:
            TrendAnalysisResult with slope, direction, and fit quality.
        """
        sr = _resolve_sampling_rate(signal_file, sampling_rate)
        await ctx.info(f"Analyzing trend of '{feature_name}' in {signal_file} ...")

        feature_series = _extract_feature_series(
            signal_file, feature_name, sr, segment_duration, overlap_ratio,
        )
        await ctx.info(f"Extracted {len(feature_series)} segments, fitting trend ...")

        trend = analyze_trend(feature_series)

        return TrendAnalysisResult(
            feature_name=feature_name,
            slope=trend["slope"],
            intercept=trend["intercept"],
            r_squared=trend["r_squared"],
            trend_direction=trend["trend_direction"],
            p_value=trend["p_value"],
            num_segments=len(feature_series),
        )

    @mcp.tool()
    async def detect_signal_degradation_onset(
        ctx: Context,
        signal_file: str,
        feature_name: str = "rms",
        threshold_sigma: float = 3.0,
        sampling_rate: Optional[float] = None,
        segment_duration: float = 0.1,
        overlap_ratio: float = 0.5,
    ) -> DegradationOnsetResult:
        """Detect whether and where a signal begins to degrade.

        Extracts a feature series from the signal and applies change detection
        to identify the first segment where the feature exceeds baseline
        statistics by *threshold_sigma* standard deviations.

        Args:
            ctx: MCP context for user communication.
            signal_file: CSV signal file in the data directory.
            feature_name: Time-domain feature to monitor (default: "rms").
            threshold_sigma: Number of baseline standard deviations to trigger
                degradation onset (default: 3.0).
            sampling_rate: Signal sampling rate in Hz (auto-detect if None).
            segment_duration: Duration of each segment in seconds.
            overlap_ratio: Overlap between segments (0-1).

        Returns:
            DegradationOnsetResult with onset detection outcome.
        """
        sr = _resolve_sampling_rate(signal_file, sampling_rate)
        await ctx.info(f"Detecting degradation onset for '{feature_name}' in {signal_file} ...")

        feature_series = _extract_feature_series(
            signal_file, feature_name, sr, segment_duration, overlap_ratio,
        )
        await ctx.info(f"Extracted {len(feature_series)} segments, checking onset ...")

        onset_index = detect_degradation_onset(feature_series, threshold_sigma)

        return DegradationOnsetResult(
            feature_name=feature_name,
            onset_detected=onset_index is not None,
            onset_segment_index=onset_index,
            threshold_sigma=threshold_sigma,
            num_segments=len(feature_series),
        )
