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
- ``forecast_asset_indicator`` projects one caller-supplied indicator
  series with a statistical reference method. It is a projection of the
  indicator, not a prognosis, and shares nothing with the RUL path.
Logging note
------------
Every tool here takes a ``ctx`` parameter it never uses. That is deliberate,
not leftover: ``tests/fixtures/tool_inventory.json`` pins ``context_kwarg``
per tool, so dropping the parameter would be a protocol-visible change to
the tool surface.

What changed in 0.12.0 is *how* progress is emitted, not whether tools
accept a context. SEP-2577 deprecated the MCP logging capability with no
in-protocol replacement, so narration goes to this module's logger, which
``server.configure_logging`` binds to stderr — stdout is the stdio
transport's JSON-RPC channel. Clients no longer receive progress
notifications; any fact a caller needs is carried by the return value.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Literal, Optional

import numpy as np
import pydantic_core
from mcp import types
from mcp.server.mcpserver import MCPServer, Context

from ..signal_processing.features import extract_time_domain_features
from ..signal_acquisition.repository import get_repository
from ._utils import refusal_boundary, resolve_signal
from ..prognostics import (
    estimate_rul_linear,
    estimate_rul_exponential,
    analyze_trend,
    detect_degradation_onset,
)
from ..prognostics.kalman_rul import estimate_rul_kalman
from ..prognostics import indicator_forecast
from ..prognostics.indicator_forecast import forecast_indicator
from ..signal_validation import InvalidInputError
from ..models import (
    ForecastAggregation,
    ForecastBackend,
    IndicatorForecastResult,
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
    idx = np.unique(np.linspace(0, len(values) - 1, max_points).round().astype(int))
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
        ctx: MCP context. Unused — see this module's docstring on logging.
        failure_threshold: Indicator value considered as failure, in the
            same units as the feature values. No universal default is
            imposed — but when the indicator is broadband VELOCITY RMS
            in mm/s, the standard choice is the ISO 10816-3:2009 zone
            C/D boundary that assess_severity / get_zone_boundaries()
            reports for the machine's group and support (single source
            of truth — no boundaries restated here).
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
            f"Unknown method '{method}' — use 'linear', 'exponential', " "or 'kalman'."
        )

    if (feature_values is None) == (signal_ids is None):
        raise ValueError(
            "Provide exactly one of feature_values or signal_ids — "
            "feature_values for externally measured indicator values, "
            "signal_ids for stored measurement recordings."
        )

    n = len(feature_values if feature_values is not None else signal_ids)
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

    logger.info(
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
            extra["rul_interval_95"] = [round(x, 4) for x in result["rul_interval_95"]]
            extra["precision_heuristic"] = round(result["precision_heuristic"], 4)
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
        ctx: MCP context. Unused — see this module's docstring on logging.
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
    logger.info(
        f"Screening within-recording trend of '{feature_name}' in " f"'{signal_id}' ..."
    )

    feature_series, segment_times = _extract_feature_series(
        signal_data,
        feature_name,
        sr,
        segment_duration,
        overlap_ratio,
    )
    logger.info(f"Extracted {len(feature_series)} segments, fitting trend ...")

    trend = analyze_trend(feature_series, timestamps=segment_times)
    series_out, times_out, truncated = _truncate_series(feature_series, segment_times)

    # Onset detection (merged detect_signal_degradation_onset): scan only
    # AFTER the baseline window — the baseline defines "normal".
    onset_index = detect_degradation_onset(feature_series, onset_threshold_sigma)
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


async def forecast_asset_indicator(
    ctx: Context,
    observed_at: list[str],
    values: list[Optional[float]],
    indicator: str,
    unit: str,
    resolution: str,
    horizon_steps: int,
    backend: ForecastBackend = "naive",
    aggregation: ForecastAggregation = "mean",
    season_length_steps: Optional[int] = None,
) -> IndicatorForecastResult:
    """Project one condition indicator's future values from its own history.

    Takes an indicator series the caller already has, for example velocity
    RMS from a data collector or a bearing temperature exported from a
    historian, and projects it forward on a regular UTC grid with a
    statistical reference method, returning P10, P50 and P90 per step. It
    reads nothing the server holds, no signal_id and no asset history: only
    the series passed in the call. The result is a projection of the
    indicator, not a fault detection, a failure probability or a
    remaining-useful-life estimate. Do not pass its values to estimate_rul
    or assess_severity.

    Args:
        ctx: MCP context. Unused: see this module's docstring on logging.
        observed_at: ISO 8601 instants, one per value, each with 'Z' or a
            UTC offset (for example '2026-10-24T08:00:00Z'), strictly
            increasing; at most 2048. An instant before 1971 or more than
            one day after the server clock is refused. A missing
            observation is expressed by omitting its instant and its value,
            never by null, zero or an interpolated value.
        values: The indicator's measured values, one finite number per
            instant. The schema accepts null only so that the server can
            refuse it with an explanation.
        indicator: What the values measure, for example 'velocity RMS'.
        unit: The unit recorded at the source, for example 'mm/s' or
            'degC'. Never inferred: ask the user when it is unknown.
        resolution: Grid step of at most 30 days, a whole number followed
            by s, min, h, d or w, for example '10min' or '1h'. Bins are
            aligned to the Unix epoch in UTC, so '1d' bins are UTC days and
            '1w' bins start on Thursdays; the series may span at most
            100000 bins.
        horizon_steps: Steps to project, 1 to 256. A horizon longer than
            30 days is refused as a call error; a horizon longer than the
            observed history span (last instant minus first instant)
            returns insufficient_history with the largest supported
            horizon.
        backend: 'naive' (default; repeats the last observed bin) or
            'seasonal_naive' (repeats the same phase of the declared
            season).
        aggregation: How several values in one bin combine: 'mean'
            (default), 'median', 'max' or 'last'.
        season_length_steps: Season in grid steps as the user states it,
            required for 'seasonal_naive' (for example 24 when the user
            says the pattern repeats daily at '1h') and never inferred;
            omit it for 'naive'.

    Returns:
        IndicatorForecastResult with status 'forecast' (quantiles per step,
        anchor steps with absolute times, warnings, history counts,
        provenance and fixed interpretation texts) or
        'insufficient_history' (reason_code, available and required
        amounts, max_supported_horizon_steps and a remedy). Quote
        interpretation, method_note and interval_note when reporting a
        projection. The history needs at least 10 difference pairs;
        seasonal naive also needs two full seasons. A call whose band is too
        narrow to report at the level of its values is refused, with a
        remedy. A forecast result exceeding the server's conservative
        32,000-byte serialized-output limit is refused. When possible, the
        refusal suggests a shorter horizon that has been recomputed and
        checked to fit.
    """
    from .. import __version__  # lazy: the package imports this module

    now = datetime.now(timezone.utc)

    def project(steps: int) -> IndicatorForecastResult:
        return IndicatorForecastResult.model_validate(
            forecast_indicator(
                observed_at=observed_at,
                values=values,
                indicator=indicator,
                unit=unit,
                resolution=resolution,
                horizon_steps=steps,
                backend=backend,
                aggregation=aggregation,
                season_length_steps=season_length_steps,
                now=now,
                server_version=__version__,
            )
        )

    result = project(horizon_steps)
    _refuse_oversized_result(result, horizon_steps, project)
    return result


#: Room for the server metadata newer protocol versions add to a tool result:
#: from the 2026-07-28 handshake on, a ``_meta`` serverInfo block naming the
#: server and its version (about 92 bytes for this server). Counted so that no
#: protocol version receives a result above the budget.
RESULT_METADATA_ALLOWANCE = 256

#: Room kept when naming a shorter horizon, for what a later call of the same
#: request adds as the server clock moves on: a stale_history warning in its
#: longest form and longer step counts (at most 470 bytes measured, even 900
#: years later at a one-second grid). Without it the named horizon, which sits
#: within one step of the budget, would be refused again once a bin boundary
#: passes.
RETRY_RESERVE = 512


def call_tool_result_bytes(result: IndicatorForecastResult) -> int:
    """Bytes of the CallToolResult that MCPServer builds for this result.

    Built the way mcp 2.0.0 to 2.3.0 build it: the structured content and its
    indented JSON text copy in one CallToolResult, without null fields. The
    SDK writes it in two encodings: pydantic's on stdio, SSE and the legacy
    HTTP transport, and the standard library's on the modern HTTP transport,
    which escapes every non-ASCII character and writes floats in Python's
    form. The larger of the two is returned.
    """
    text = pydantic_core.to_json(result, fallback=str, indent=2).decode()
    call = types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        structured_content=result.model_dump(mode="json", by_alias=True),
    )
    pydantic_form = call.model_dump_json(by_alias=True, exclude_none=True)
    stdlib_form = json.dumps(
        call.model_dump(mode="json", by_alias=True, exclude_none=True),
        separators=(",", ":"),
    )
    return max(len(pydantic_form.encode()), len(stdlib_form.encode()))


def client_result_bytes(result: IndicatorForecastResult) -> int:
    """Most bytes a client receives for this result, on any protocol version.

    The CallToolResult plus RESULT_METADATA_ALLOWANCE. The JSON-RPC envelope
    around it, whose size depends on the client's request id, is not counted.
    """
    return call_tool_result_bytes(result) + RESULT_METADATA_ALLOWANCE


def _refuse_oversized_result(
    result: IndicatorForecastResult,
    horizon_steps: int,
    project: Callable[[int], IndicatorForecastResult],
) -> None:
    """Refuse a result over MAX_RESULT_BYTES, naming a horizon checked to fit.

    The shorter horizon is found by recomputing and measuring the result for
    each candidate, so the one named has itself been measured, with
    RETRY_RESERVE to spare; nothing is shortened or dropped from the
    requested result to make it fit. Every candidate is a forecast: a shorter
    horizon passes the sufficiency and precision checks the requested one
    passed.

    The binary search finds the longest horizon that fits only when the size
    grows with the horizon, which is not guaranteed: the isolated_origin
    warning is carried only up to the number of empty bins before the last
    observed bin, the stale_history text is longer while every projected step
    is past, and the anchors move with the horizon. The size can fall across
    such a horizon, so the horizon named is checked to fit but can be a few
    steps shorter than the longest that does; the refusal never claims it is
    the longest. For the same reason, when none of the horizons the search
    measured fits, a horizon it skipped still can, so that refusal says only
    that no shorter horizon was verified to fit.

    Raises:
        InvalidInputError: When the result exceeds the budget.
    """
    budget = indicator_forecast.MAX_RESULT_BYTES
    size = client_result_bytes(result)
    if size <= budget:
        return
    fit: Optional[tuple[int, int]] = None
    low, high = 1, horizon_steps - 1
    while low <= high:
        steps = (low + high) // 2
        steps_size = client_result_bytes(project(steps))
        if steps_size <= budget - RETRY_RESERVE:
            fit, low = (steps, steps_size), steps + 1
        else:
            high = steps - 1
    problem = (
        f"The result for horizon_steps {horizon_steps} has a conservative "
        f"serialized size of {size:,} bytes, over the server's {budget:,}-byte "
        f"output-size limit. The size is an upper bound for the result across "
        f"the serializations the server supports, so a particular transport may "
        f"carry fewer bytes for it. Its size depends on the number of forecast "
        f"steps, the digits needed to represent the quantile bands, and the length "
        f"of the indicator and unit labels repeated in the response."
    )
    kept = (
        "No quantile values are truncated, further rounded or omitted to fit "
        "the limit."
    )
    if fit is None:
        raise InvalidInputError(
            f"{problem} No shorter horizon was verified to fit within the "
            f"output-size limit for these inputs. If the indicator or unit labels "
            f"are long, shorten them before retrying. {kept}"
        )
    raise InvalidInputError(
        f"{problem} Call again with horizon_steps of {fit[0]}: that result was "
        f"recomputed and checked to fit, at a conservative size of {fit[1]:,} "
        f"bytes. {kept}"
    )


def register(mcp: MCPServer) -> None:
    """Register prognostics MCP tools on *mcp*."""
    mcp.tool()(refusal_boundary(estimate_rul))
    mcp.tool()(refusal_boundary(analyze_signal_trend))
    mcp.tool()(refusal_boundary(forecast_asset_indicator))
