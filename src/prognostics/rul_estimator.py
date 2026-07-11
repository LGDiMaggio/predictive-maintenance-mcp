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
from typing import Sequence

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
        *None* when the fitted line never reaches the threshold after
        the last measurement.

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

    if slope <= 0:
        return None

    crossing_time = (failure_threshold - intercept) / slope
    rul = crossing_time - float(t[-1])

    if rul <= 0:
        # Already past (or at) the threshold according to the fit.
        return None

    return {
        "rul": float(rul),
        "fit_r_squared": r_squared,
        "estimated_rate": slope,
        "method": "linear",
    }


def estimate_rul_exponential(
    feature_series: Sequence[float],
    timestamps: Sequence[float],
    failure_threshold: float,
) -> dict | None:
    """Estimate RUL by fitting an exponential degradation curve.

    Fits ``y = a * exp(b * t)`` via log-linear regression on the
    strictly positive values of *feature_series*, using the measurement
    timestamps as the time axis.

    Args:
        feature_series: Degradation indicator values, one per
            measurement (assumed to rise with degradation).
        timestamps: Measurement times, strictly increasing.
        failure_threshold: Indicator value at which the component is
            considered failed. Must be positive.

    Returns:
        Dict with ``rul``, ``fit_r_squared`` (goodness of the
        log-linear fit, NOT a confidence), and ``method``
        (``"exponential"``). Returns *None* when the fit is not
        feasible or the curve never reaches the threshold after the
        last measurement.

    Raises:
        ValueError: If series/timestamps lengths differ or timestamps
            are not strictly increasing.
    """
    validated = _validate_series(feature_series, timestamps)
    if validated is None:
        return None
    y, t = validated

    if failure_threshold <= 0:
        return None

    # Keep only strictly positive values for the log-linear fit.
    mask = y > 0
    if mask.sum() < 2:
        return None

    t_pos = t[mask]
    log_y = np.log(y[mask])

    result = stats.linregress(t_pos, log_y)
    b = float(result.slope)
    log_a = float(result.intercept)
    r_squared = float(result.rvalue**2)
    if math.isnan(r_squared):
        r_squared = 0.0

    if b <= 0:
        return None

    # Solve a * exp(b * t_cross) = F  →  t_cross = (ln F − ln a) / b
    crossing_time = (math.log(failure_threshold) - log_a) / b
    rul = crossing_time - float(t[-1])

    if rul <= 0:
        return None

    return {
        "rul": float(rul),
        "fit_r_squared": r_squared,
        "method": "exponential",
    }
