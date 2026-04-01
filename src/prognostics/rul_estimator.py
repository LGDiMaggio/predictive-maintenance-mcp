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


def estimate_rul_weibull(
    feature_series: list[float],
    failure_threshold: float,
    sampling_interval: float = 1.0,
) -> dict | None:
    """Estimate RUL by fitting a Weibull degradation curve.

    Fits a Weibull distribution to *feature_series* using
    ``scipy.stats.weibull_min.fit()`` and extrapolates when the
    degradation curve reaches the *failure_threshold*.

    Args:
        feature_series: Ordered degradation indicator values.
        failure_threshold: Value at which the component is considered
            failed.
        sampling_interval: Time between successive samples.

    Returns:
        Dict with ``rul``, ``confidence``, ``method`` (``"weibull"``),
        ``shape`` (k), and ``scale`` (lambda).  Returns *None* when the
        fit is not feasible or the curve never reaches the threshold.
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 3:
        return None

    # Weibull fit requires strictly positive values.
    if np.any(y <= 0):
        return None

    # Flat series cannot be meaningfully fitted.
    if np.ptp(y) < 1e-12:
        return None

    # Check if the last value already exceeds the threshold.
    if y[-1] >= failure_threshold:
        return None

    try:
        shape, loc, scale = stats.weibull_min.fit(y, floc=0)
    except Exception:
        return None

    if shape <= 0 or scale <= 0:
        return None

    # Evaluate the Weibull CDF at each sample index to build a
    # degradation curve scaled so it passes through the last observed point.
    x = np.arange(n, dtype=float)
    cdf_last = stats.weibull_min.cdf(float(n - 1), shape, loc=0, scale=scale)
    if cdf_last <= 1e-12:
        return None
    fit_scale = float(y[-1]) / cdf_last
    fitted = stats.weibull_min.cdf(x, shape, loc=0, scale=scale) * fit_scale

    # Compute R-squared as a confidence proxy.
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r_squared = max(0.0, r_squared)

    # Extrapolate forward to find when the degradation curve crosses
    # the threshold.  We scale the CDF so that it passes through the last
    # observed point (preserving monotonicity) and can exceed data_max.
    max_future = n * 10
    cdf_at_last = stats.weibull_min.cdf(float(n - 1), shape, loc=0, scale=scale)
    if cdf_at_last <= 1e-12:
        return None
    scale_factor = float(y[-1]) / cdf_at_last
    future_x = np.arange(n, n + max_future, dtype=float)
    projected = stats.weibull_min.cdf(future_x, shape, loc=0, scale=scale) * scale_factor
    crossings = np.where(projected >= failure_threshold)[0]

    if len(crossings) == 0:
        return None

    first_crossing = int(future_x[crossings[0]])
    remaining_indices = first_crossing - (n - 1)
    rul = float(remaining_indices * sampling_interval)
    return {
        "rul": rul,
        "confidence": r_squared,
        "method": "weibull",
        "shape": float(shape),
        "scale": float(scale),
    }
