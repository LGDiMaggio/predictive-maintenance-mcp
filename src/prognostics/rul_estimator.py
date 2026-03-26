"""
ISO 13374 Block 5 — Prognostic Assessment: RUL Estimation.

Remaining Useful Life estimation using degradation curve fitting.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def estimate_rul_linear(
    feature_series: list[float],
    failure_threshold: float,
    sampling_interval: float = 1.0,
) -> dict | None:
    """Estimate RUL by linearly extrapolating to *failure_threshold*.

    Args:
        feature_series: Ordered degradation indicator values.
        failure_threshold: Value at which the component is considered
            failed.
        sampling_interval: Time between successive samples (arbitrary
            units — hours, cycles, etc.).

    Returns:
        Dict with ``rul``, ``confidence`` (R-squared of the linear
        fit), and ``method`` (``"linear"``).  Returns *None* when the
        extrapolated line never reaches the threshold.
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 2:
        return None

    x = np.arange(n, dtype=float)
    result = stats.linregress(x, y)
    slope = result.slope
    intercept = result.intercept
    r_squared = float(result.rvalue ** 2)

    if slope == 0:
        return None

    # Index at which the line crosses the threshold.
    crossing_index = (failure_threshold - intercept) / slope

    # RUL is the distance from the *last* observation to the crossing.
    remaining_indices = crossing_index - (n - 1)

    if remaining_indices <= 0:
        # Already past the threshold (or at it).
        return None

    rul = float(remaining_indices * sampling_interval)
    return {"rul": rul, "confidence": r_squared, "method": "linear"}


def estimate_rul_exponential(
    feature_series: list[float],
    failure_threshold: float,
    sampling_interval: float = 1.0,
) -> dict | None:
    """Estimate RUL by fitting an exponential degradation curve.

    Fits ``y = a * exp(b * t)`` via log-linear regression on the
    positive values of *feature_series*.

    Args:
        feature_series: Ordered degradation indicator values.
        failure_threshold: Value at which the component is considered
            failed.
        sampling_interval: Time between successive samples.

    Returns:
        Dict with ``rul``, ``confidence``, and ``method``
        (``"exponential"``).  Returns *None* when the fit is not
        feasible or the curve never reaches the threshold.
    """
    y = np.asarray(feature_series, dtype=float)

    # Keep only strictly positive values for log-linear fit.
    mask = y > 0
    if mask.sum() < 2:
        return None

    x_pos = np.arange(len(y), dtype=float)[mask]
    y_pos = y[mask]
    log_y = np.log(y_pos)

    result = stats.linregress(x_pos, log_y)
    b = result.slope
    log_a = result.intercept
    r_squared = float(result.rvalue ** 2)
    a = math.exp(log_a)

    if b == 0 or a <= 0:
        return None

    if failure_threshold <= 0:
        return None

    # Solve a * exp(b * t_cross) = failure_threshold  →  t_cross = ln(F/a) / b
    try:
        crossing_index = math.log(failure_threshold / a) / b
    except (ValueError, ZeroDivisionError):
        return None

    remaining_indices = crossing_index - (len(y) - 1)

    if remaining_indices <= 0:
        return None

    rul = float(remaining_indices * sampling_interval)
    return {"rul": rul, "confidence": r_squared, "method": "exponential"}
