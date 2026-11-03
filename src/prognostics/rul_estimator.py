"""
ISO 13374 Block 5 — Prognostic Assessment: RUL Estimation.

Remaining Useful Life estimation by fitting a degradation curve to a
series of feature measurements taken over time.

RUL is only physically meaningful when computed from repeated
measurements of the same machine collected over a real time span
(days/weeks/months). Fitting a curve inside a single stationary
recording produces segment-to-segment noise, not a degradation trend.
The MCP tool layer enforces this precondition; the estimators here
assume a valid timestamped series.

Both estimators report ``fit_r_squared`` — the coefficient of
determination of the fitted curve on the observed data. It measures
goodness of fit only; it is NOT a probability that the extrapolated
RUL is correct.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def _validate_series(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert and sanity-check a timestamped series.

    Returns:
        Tuple ``(y, t)`` of float arrays, or *None* when the series is
        too short for a fit (fewer than 2 points).

    Raises:
        ValueError: If lengths differ or timestamps are not strictly
            increasing.
    """
    y = np.asarray(feature_series, dtype=float)
    t = np.asarray(timestamps, dtype=float)

    if len(y) != len(t):
        raise ValueError(
            f"feature_series has {len(y)} values but timestamps has "
            f"{len(t)} — provide one timestamp per measurement."
        )
    if len(y) < 2:
        return None
    if np.any(np.diff(t) <= 0):
        raise ValueError(
            "timestamps must be strictly increasing — sort the "
            "measurements chronologically and remove duplicates."
        )
    return y, t


def _time_after_last(crossing_time: float | None, t: np.ndarray) -> float | None:
    """Time from the last measurement to the crossing; *None* without one."""
    return None if crossing_time is None else crossing_time - float(t[-1])


def validate_positive_for_exponential(feature_series: Iterable[float]) -> None:
    """Refuse a series the log-linear fit cannot use whole.

    Fitting on the positive values only would drop measurements without
    saying so, so the first value that is not positive is named instead.

    Raises:
        ValueError: If any value is zero, negative or not a number.
    """
    for index, value in enumerate(feature_series):
        if not value > 0:
            raise ValueError(
                "The exponential method fits the logarithm of the values and needs "
                "every measurement to be positive, but the measurement at index "
                f"{index} (counting from 0) is {value:g}. No measurement is "
                "dropped: use method='linear' or "
                "method='kalman', or pass an indicator that is positive."
            )


def fit_linear_degradation(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
    failure_threshold: float,
) -> dict | None:
    """Fit a straight line and locate where it meets *failure_threshold*.

    Fits ``y = intercept + slope * t`` on the measurement timestamps. The
    crossing is reported wherever it falls, including at or before the
    last measurement, so a caller can tell a fit that has already passed
    the threshold from one that never rises.

    Args:
        feature_series: Degradation indicator values, one per measurement.
        timestamps: Measurement times, strictly increasing.
        failure_threshold: Indicator limit the crossing is located for.

    Returns:
        Dict with ``slope``, ``intercept``, ``fit_r_squared`` (goodness of
        fit, NOT a confidence), ``fitted_last_value`` (the line at the last
        timestamp), ``crossing_time`` (on the timestamps' axis) and
        ``time_to_crossing`` (crossing time minus the last timestamp, zero
        or negative when the line reached the threshold by then); both
        *None* when the slope is not positive. *None* when the series has
        fewer than 2 points.

    Raises:
        ValueError: If series/timestamps lengths differ or timestamps
            are not strictly increasing.
    """
    validated = _validate_series(feature_series, timestamps)
    if validated is None:
        return None
    y, t = validated

    result = stats.linregress(t, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue**2)
    if math.isnan(r_squared):
        r_squared = 0.0

    crossing_time = (failure_threshold - intercept) / slope if slope > 0 else None
    return {
        "slope": slope,
        "intercept": intercept,
        "fit_r_squared": r_squared,
        "fitted_last_value": intercept + slope * float(t[-1]),
        "crossing_time": crossing_time,
        "time_to_crossing": _time_after_last(crossing_time, t),
    }


def estimate_rul_linear(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
    failure_threshold: float,
) -> dict | None:
    """Estimate RUL by linearly extrapolating to *failure_threshold*.

    Fits ``y = intercept + slope * t`` on the measurement timestamps and
    extrapolates to the time at which the line crosses the threshold.

    Args:
        feature_series: Degradation indicator values, one per
            measurement (assumed to rise with degradation).
        timestamps: Measurement times, strictly increasing. Units are
            the caller's choice (hours, days, ...); the RUL is returned
            in the same units.
        failure_threshold: Indicator value at which the component is
            considered failed.

    Returns:
        Dict with ``rul`` (time from the last measurement to the
        threshold crossing), ``fit_r_squared`` (goodness of fit of the
        line, NOT a confidence), ``estimated_rate`` (slope in feature
        units per time unit), and ``method`` (``"linear"``). Returns
        *None* when the fitted line does not rise, or reaches the
        threshold at or before the last measurement
        (``fit_linear_degradation`` tells the two apart).

    Raises:
        ValueError: If series/timestamps lengths differ or timestamps
            are not strictly increasing.
    """
    fit = fit_linear_degradation(feature_series, timestamps, failure_threshold)
    if fit is None or fit["time_to_crossing"] is None or fit["time_to_crossing"] <= 0:
        return None

    return {
        "rul": float(fit["time_to_crossing"]),
        "fit_r_squared": fit["fit_r_squared"],
        "estimated_rate": fit["slope"],
        "method": "linear",
    }


def fit_exponential_degradation(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
    failure_threshold: float,
) -> dict | None:
    """Fit ``y = a * exp(b * t)`` and locate where it meets *failure_threshold*.

    Log-linear regression on every value of *feature_series*, which must
    all be positive. As for the linear fit, the crossing is reported
    wherever it falls.

    Args:
        feature_series: Degradation indicator values, one per
            measurement, all positive.
        timestamps: Measurement times, strictly increasing.
        failure_threshold: Indicator limit the crossing is located for.
            Must be positive.

    Returns:
        Dict with ``growth_rate`` (b), ``log_intercept`` (ln a),
        ``fit_r_squared`` (of the log-linear fit, NOT a confidence),
        ``fitted_last_value`` (the curve at the last timestamp),
        ``crossing_time`` and ``time_to_crossing`` (as for the linear fit;
        both *None* when b is not positive). *None* when the series has
        fewer than 2 points.

    Raises:
        ValueError: If series/timestamps lengths differ, timestamps are
            not strictly increasing, a value is not positive, or the
            threshold is not positive.
    """
    validated = _validate_series(feature_series, timestamps)
    if validated is None:
        return None
    y, t = validated

    if failure_threshold <= 0:
        raise ValueError(
            "failure_threshold must be positive for the exponential method, got "
            f"{failure_threshold:g}: the curve a * exp(b * t) never reaches a "
            "threshold at or below zero."
        )
    validate_positive_for_exponential(y)

    result = stats.linregress(t, np.log(y))
    b = float(result.slope)
    log_a = float(result.intercept)
    r_squared = float(result.rvalue**2)
    if math.isnan(r_squared):
        r_squared = 0.0

    crossing_time = (math.log(failure_threshold) - log_a) / b if b > 0 else None
    return {
        "growth_rate": b,
        "log_intercept": log_a,
        "fit_r_squared": r_squared,
        "fitted_last_value": float(np.exp(log_a + b * float(t[-1]))),
        "crossing_time": crossing_time,
        "time_to_crossing": _time_after_last(crossing_time, t),
    }


def estimate_rul_exponential(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
    failure_threshold: float,
) -> dict | None:
    """Estimate RUL by fitting an exponential degradation curve.

    Fits ``y = a * exp(b * t)`` via log-linear regression on the values
    of *feature_series*, which must all be positive, using the
    measurement timestamps as the time axis.

    Args:
        feature_series: Degradation indicator values, one per
            measurement (assumed to rise with degradation), all positive.
        timestamps: Measurement times, strictly increasing.
        failure_threshold: Indicator value at which the component is
            considered failed. Must be positive.

    Returns:
        Dict with ``rul``, ``fit_r_squared`` (goodness of the
        log-linear fit, NOT a confidence), and ``method``
        (``"exponential"``). Returns *None* when the threshold is not
        positive, the curve does not rise, or it reaches the threshold
        at or before the last measurement.

    Raises:
        ValueError: If series/timestamps lengths differ, timestamps are
            not strictly increasing, or a value is not positive.
    """
    validated = _validate_series(feature_series, timestamps)
    if validated is None or failure_threshold <= 0:
        return None

    fit = fit_exponential_degradation(feature_series, timestamps, failure_threshold)
    if fit is None or fit["time_to_crossing"] is None or fit["time_to_crossing"] <= 0:
        return None

    return {
        "rul": float(fit["time_to_crossing"]),
        "fit_r_squared": fit["fit_r_squared"],
        "method": "exponential",
    }
