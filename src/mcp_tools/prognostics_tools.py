"""MCP tools for prognostic assessment (ISO 13374 Block 5).

Exposes RUL estimation on multi-measurement series, plus a within-recording
trend + onset screening tool.

Honest-prognosis contract:
- ``estimate_rul`` requires a series of measurements taken over time
  (explicit values or multiple stored signals, each with a timestamp).
  It refuses a single recording/point: segmenting seconds of stationary
  signal yields noise, not a degradation trend.
- ``analyze_signal_trend`` (trend + onset, unified in U9) is a
  within-recording SCREENING tool. It looks at seconds of data inside
  one recording and cannot produce a prognosis.
"""

import logging
from typing import Literal, Optional

import numpy as np
from mcp.server.fastmcp import FastMCP, Context

from ..signal_processing.features import extract_time_domain_features
from ..signal_acquisition.repository import get_repository
from ._utils import resolve_signal
from ..prognostics import (
    estimate_rul_linear,
    estimate_rul_exponential,
    analyze_trend,
    detect_degradation_onset,
)
from ..prognostics.kalman_rul import estimate_rul_kalman
from ..models import (
    RULEstimationResult,
    TrendAnalysisResult,
)

logger = logging.getLogger(__name__)

# Maximum points echoed back in TrendAnalysisResult.feature_series.
MAX_SERIES_POINTS = 50

# Significance level for the degradation-trend gate in estimate_rul.
TREND_ALPHA = 0.05

# Maximum relative spread of measurement intervals tolerated by the
# Kalman method (which assumes uniform sampling).
KALMAN_MAX_SPACING_SPREAD = 0.10

_MULTI_MEASURE_ERROR = (
    "RUL estimation requires repeated measurements over time — got {n} "
    "measurement(s). A single recording (or single point) cannot show a "
    "degradation trend: provide at least 3 measurements of the same machine "
    "taken at different times (feature_values + timestamps, or one signal_id "
    "per measurement session + timestamps). For screening WITHIN a single "
    "recording use analyze_signal_trend instead."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_feature_series(
    signal_data: np.ndarray,
    feature_name: str,
    sampling_rate: float,
    segment_duration: float,
    overlap_ratio: float,
) -> tuple[list[float], list[float]]:
    """Segment a signal array and return (feature values, segment center times in s)."""
    segment_length = int(segment_duration * sampling_rate)
    hop_length = int(segment_length * (1 - overlap_ratio))

    if segment_length < 1:
        raise ValueError("segment_duration * sampling_rate must be >= 1")
    if hop_length < 1:
        hop_length = 1

    starts = list(range(0, len(signal_data) - segment_length + 1, hop_length))
    if not starts:
        raise ValueError(
            f"Signal too short ({len(signal_data)} samples) for "
            f"segment_duration={segment_duration}s at {sampling_rate} Hz"
        )

    feature_values: list[float] = []
    segment_times: list[float] = []
    for start in starts:
        seg = signal_data[start : start + segment_length]
        feats = extract_time_domain_features(seg)
        if feature_name not in feats:
            available = ", ".join(sorted(feats.keys()))
            raise ValueError(
                f"Unknown feature '{feature_name}'. Available: {available}"
            )
        feature_values.append(float(feats[feature_name]))
        segment_times.append((start + segment_length / 2.0) / sampling_rate)

    return feature_values, segment_times


def _truncate_series(
    values: list[float], times: list[float], max_points: int = MAX_SERIES_POINTS
) -> tuple[list[float], list[float], bool]:
    """Evenly subsample a series down to *max_points* for output echoing."""
    if len(values) <= max_points:
        return values, times, False
    idx = np.unique(
        np.linspace(0, len(values) - 1, max_points).round().astype(int)
    )
    return (
        [round(values[i], 6) for i in idx],
        [round(times[i], 4) for i in idx],
        True,
    )


def _measurement_series_from_signals(
    signal_ids: list[str], feature_name: str
) -> list[float]:
    """Reduce each stored signal to one scalar feature value (one per measurement)."""
    repo = get_repository()
    values: list[float] = []
    for sid in signal_ids:
        try:
            arr = repo.get_signal(sid)
        except KeyError as exc:
            # Standard repository not-found message (available ids,
            # eviction explanation, load_signal/list_signals remedy).
            raise ValueError(str(exc.args[0])) from None
        feats = extract_time_domain_features(np.asarray(arr, dtype=float))
        if feature_name not in feats:
            available_feats = ", ".join(sorted(feats.keys()))
            raise ValueError(
                f"Unknown feature '{feature_name}'. Available: {available_feats}"
            )
        values.append(float(feats[feature_name]))
    return values


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


async def estimate_rul(
    ctx: Context,
    failure_threshold: float,
    timestamps: list[float],
    feature_values: Optional[list[float]] = None,
    signal_ids: Optional[list[str]] = None,
    feature_name: str = "rms",
    method: Literal["linear", "exponential", "kalman"] = "linear",
    time_unit: str = "hours",
) -> RULEstimationResult:
    """Estimate Remaining Useful Life from repeated measurements over time.

        RUL is only physically meaningful when fitted on a degradation trend
        across MULTIPLE measurements of the same machine taken at different
        times (days/weeks/months apart). This tool refuses a single
        recording or single point — for within-recording screening use
        analyze_signal_trend instead.

        Two mutually exclusive input routes (both need `timestamps`, one
        entry per measurement, strictly increasing, in `time_unit`):
        1. `feature_values`: the degradation indicator already measured
           externally (e.g. RMS velocity trended by a data collector).
        2. `signal_ids`: one stored signal per measurement session (loaded
           via load_signal); each recording is reduced to a single
           `feature_name` value.

        The degradation indicator is assumed to RISE toward
        `failure_threshold`. A statistically significant increasing trend
        (slope p-value < 0.05) is required before any RUL is computed; a
        flat/insignificant series returns status 'no_degradation_trend'
        with no RUL number.

        Args:
            ctx: MCP context for user communication.
            failure_threshold: Indicator value considered as failure, in the
                same units as the feature values. No universal default is
                imposed — but when the indicator is broadband VELOCITY RMS
                in mm/s, the ISO 10816-3:2009 zone C/D boundary for the
                machine is the standard choice: 4.5 (group 2 rigid), 7.1
                (group 1 rigid / group 2 flexible), or 11.0 mm/s (group 1
                flexible) — the same table assess_severity uses.
            timestamps: Measurement times in `time_unit`, strictly
                increasing (e.g. hours since first measurement).
            feature_values: Indicator values, one per measurement
                (mutually exclusive with signal_ids).
            signal_ids: Stored signal IDs, one per measurement session
                (mutually exclusive with feature_values).
            feature_name: Time-domain feature used to reduce each signal
                (default: "rms"). Ignored for feature_values input.
            method: "linear" (default), "exponential", or "kalman"
                (kalman needs approximately uniform measurement spacing).
            time_unit: Label for the time axis; RUL and
                observation_horizon are expressed in this unit.

        Returns:
            RULEstimationResult with status, rul (only when estimated),
            fit_r_squared (goodness of fit — NOT a confidence),
            observation_horizon, and a plain-language message.
        """
    # --- Input route validation -----------------------------------
    if method not in ("linear", "exponential", "kalman"):
        raise ValueError(
            f"Unknown method '{method}' — use 'linear', 'exponential', "
                "or 'kalman'."
        )

    if (feature_values is None) == (signal_ids is None):
        raise ValueError(
            "Provide exactly one of feature_values or signal_ids — "
                "feature_values for externally measured indicator values, "
                "signal_ids for stored measurement recordings."
        )

    n = len(feature_values if feature_values is not None else signal_ids)  # type: ignore[arg-type]
    if n < 3:
        raise ValueError(_MULTI_MEASURE_ERROR.format(n=n))

    if feature_values is not None:
        values = [float(v) for v in feature_values]
    else:
        assert signal_ids is not None
        values = _measurement_series_from_signals(signal_ids, feature_name)

    if len(timestamps) != n:
        raise ValueError(
            f"Got {n} measurements but {len(timestamps)} timestamps — "
                "provide one timestamp per measurement."
        )
    t = [float(x) for x in timestamps]
    if any(b <= a for a, b in zip(t, t[1:])):
        raise ValueError(
            "timestamps must be strictly increasing — sort the "
                "measurements chronologically and remove duplicates."
        )

    horizon = t[-1] - t[0]
    current_value = values[-1]

    await ctx.info(
        f"Estimating RUL ({method}) from {n} measurements spanning "
            f"{horizon:g} {time_unit} ..."
    )

    common = dict(
        method=method,
        feature_name=feature_name,
        num_measurements=n,
        observation_horizon=round(horizon, 6),
        time_unit=time_unit,
        failure_threshold=failure_threshold,
        current_value=round(current_value, 6),
    )

    # --- Gate 1: already at/above the failure threshold -----------
    if current_value >= failure_threshold:
        return RULEstimationResult(
            status="threshold_already_exceeded",
            trend_p_value=None,
            message=(
                f"The most recent measurement ({current_value:g}) is at or "
                    f"above the failure threshold ({failure_threshold:g}) — "
                    "no remaining life to estimate. Inspect the machine now."
            ),
            **common,
        )

    # --- Gate 2: statistically significant increasing trend -------
    trend = analyze_trend(values, timestamps=t, significance_level=TREND_ALPHA)
    p_value = trend["p_value"]

    if p_value is None or p_value >= TREND_ALPHA or trend["slope"] <= 0:
        if trend["slope"] <= 0 and p_value is not None and p_value < TREND_ALPHA:
            reason = (
                "the indicator is decreasing/flat, moving away from the "
                    "failure threshold"
            )
        else:
            reason = (
                "no statistically significant increasing trend "
                    f"(slope p-value = {p_value if p_value is not None else 'n/a'})"
            )
        return RULEstimationResult(
            status="no_degradation_trend",
            trend_p_value=p_value,
            fit_r_squared=round(trend["r_squared"], 4),
            message=(
                f"No degradation trend detected: {reason}. The machine "
                    f"appears stable over the {horizon:g} {time_unit} "
                    "observed — keep collecting measurements and re-run "
                    "estimate_rul as the series grows."
            ),
            **common,
        )

    # --- Fit the requested degradation model ----------------------
    extra: dict = {}
    if method == "linear":
        result = estimate_rul_linear(values, t, failure_threshold)
    elif method == "exponential":
        result = estimate_rul_exponential(values, t, failure_threshold)
    else:  # kalman
        dts = np.diff(t)
        mean_dt = float(np.mean(dts))
        spread = float((np.max(dts) - np.min(dts)) / mean_dt)
        if spread > KALMAN_MAX_SPACING_SPREAD:
            raise ValueError(
                "The kalman method assumes uniformly spaced measurements, "
                    f"but the intervals vary by {spread * 100:.0f}% — use "
                    "method='linear' or 'exponential', or resample the series "
                    "to a regular interval."
            )
        result = estimate_rul_kalman(
            values, failure_threshold, sampling_interval=mean_dt
        )
        if result is not None:
            extra["rul_interval_95"] = [
                round(x, 4) for x in result["rul_interval_95"]
            ]
            extra["precision_heuristic"] = round(
                result["precision_heuristic"], 4
            )
            extra["estimated_rate"] = round(result["estimated_rate"], 6)

    if result is None:
        return RULEstimationResult(
            status="no_degradation_trend",
            trend_p_value=p_value,
            fit_r_squared=round(trend["r_squared"], 4),
            message=(
                f"The fitted {method} degradation curve does not reach the "
                    f"failure threshold ({failure_threshold:g}) after the last "
                    "measurement — no finite RUL. Keep collecting measurements."
            ),
            **common,
        )

    rul = float(result["rul"])
    if method == "linear":
        extra["estimated_rate"] = round(result["estimated_rate"], 6)
    fit_r2 = result.get("fit_r_squared")

    message = (
        f"RUL estimated at {rul:g} {time_unit} from {n} measurements "
            f"spanning {horizon:g} {time_unit}. This is an extrapolation of "
            "the observed trend — treat it as planning input, not a guarantee."
    )
    if rul > horizon:
        message += (
            f" Caution: the estimate extends {rul / horizon:.1f}x beyond "
                "the observation horizon, so its reliability is low."
        )

    return RULEstimationResult(
        status="estimated",
        trend_p_value=p_value,
        rul=round(rul, 4),
        fit_r_squared=round(fit_r2, 4) if fit_r2 is not None else None,
        message=message,
        **common,
        **extra,
    )

async def analyze_signal_trend(
    ctx: Context,
    signal_id: str,
    feature_name: str = "rms",
    segment_duration: float = 0.1,
    overlap_ratio: float = 0.5,
    onset_threshold_sigma: float = 3.0,
) -> TrendAnalysisResult:
    """Within-recording screening: feature trend + degradation onset.

        THE unified screening tool: feature trend AND degradation
        onset in one call. Segments a single recording
        (seconds of data), extracts the requested feature per segment,
        tests whether the per-segment values show a statistically
        significant trend (slope p < 0.05), and detects the first segment
        AFTER the baseline window (first half of the series) whose value
        exceeds baseline mean + onset_threshold_sigma standard deviations.
        Onset inside the baseline window cannot be detected (the baseline
        defines "normal"). Requires the signal loaded via load_signal()
        first; the sampling rate comes from the stored signal metadata.

        This is a SCREENING tool, not a prognosis: a trend inside seconds
        of signal says whether the recording is stationary, not how long
        the machine will live. For Remaining Useful Life, collect repeated
        measurements over days/weeks (one recording per session) and pass
        them to estimate_rul — this tool returns the per-segment feature
        series so each recording can be reduced to one measurement point.

        Args:
            ctx: MCP context for user communication.
            signal_id: ID of the stored signal (from load_signal).
            feature_name: Time-domain feature to analyze (default: "rms").
            segment_duration: Duration of each segment in seconds.
            overlap_ratio: Overlap between segments (0-1).
            onset_threshold_sigma: Baseline standard deviations above the
                baseline mean that trigger onset detection (default: 3.0).

        Returns:
            TrendAnalysisResult with slope, direction (p-value based),
            fit quality, the (truncated) per-segment feature series, and
            the onset-detection outcome (onset_detected,
            onset_segment_index, onset_time_s, baseline_segments).

        Raises:
            ValueError: If the signal_id is not loaded, or the stored
                signal has no sampling rate.
        """
    signal_data, info = resolve_signal(signal_id)
    sr = info.sampling_rate
    await ctx.info(
        f"Screening within-recording trend of '{feature_name}' in "
            f"'{signal_id}' ..."
    )

    feature_series, segment_times = _extract_feature_series(
        signal_data, feature_name, sr, segment_duration, overlap_ratio,
    )
    await ctx.info(
        f"Extracted {len(feature_series)} segments, fitting trend ..."
    )

    trend = analyze_trend(feature_series, timestamps=segment_times)
    series_out, times_out, truncated = _truncate_series(
        feature_series, segment_times
    )

    # Onset detection (merged detect_signal_degradation_onset): scan only
    # AFTER the baseline window — the baseline defines "normal".
    onset_index = detect_degradation_onset(
        feature_series, onset_threshold_sigma
    )
    onset_time_s = (
        round(segment_times[onset_index], 4) if onset_index is not None else None
    )

    return TrendAnalysisResult(
        feature_name=feature_name,
        slope=trend["slope"],
        intercept=trend["intercept"],
        r_squared=trend["r_squared"],
        trend_direction=trend["trend_direction"],
        p_value=trend["p_value"],
        num_segments=len(feature_series),
        analysis_scope="within_recording_screening",
        feature_series=series_out,
        segment_times_s=times_out,
        series_truncated=truncated,
        onset_detected=onset_index is not None,
        onset_segment_index=onset_index,
        onset_time_s=onset_time_s,
        onset_threshold_sigma=onset_threshold_sigma,
        baseline_segments=len(feature_series) // 2,
    )


def register(mcp: FastMCP) -> None:
    """Register prognostics MCP tools on *mcp*."""
    mcp.tool()(estimate_rul)
    mcp.tool()(analyze_signal_trend)
