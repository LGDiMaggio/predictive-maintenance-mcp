"""
ISO 13374 Block 5 — Prognostic Assessment: Trend Analysis.

Detects degradation trends in time-series feature data using
statistical methods (linear regression, change detection).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def analyze_trend(
    feature_series: list[float],
    timestamps: list[float] | None = None,
    significance_level: float = 0.05,
) -> dict:
    """Fit a linear trend to a feature time series.

    Uses ordinary least-squares regression. The trend direction is
    decided by the regression's own significance test (two-sided
    p-value of the slope), not by an arbitrary R² cutoff: a direction
    is only reported when ``p_value < significance_level``.

    Args:
        feature_series: Ordered feature values (e.g., RMS velocity
            collected over successive measurements).
        timestamps: Optional time axis.  When *None*, indices
            ``0 .. N-1`` are used.
        significance_level: Alpha for the slope significance test
            (default: 0.05).

    Returns:
        Dictionary with keys:
            - slope (float)
            - intercept (float)
            - r_squared (float)
            - trend_direction (str): ``"increasing"``, ``"decreasing"``,
              or ``"stable"``
            - p_value (float | None): Two-sided p-value of the slope.
              *None* when the test is not computable (fewer than 3
              points or degenerate data), in which case the direction
              is reported as ``"stable"``.
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 2:
        return {
            "slope": 0.0,
            "intercept": float(y[0]) if n == 1 else 0.0,
            "r_squared": 0.0,
            "trend_direction": "stable",
            "p_value": None,
        }

    if timestamps is not None:
        x = np.asarray(timestamps, dtype=float)
    else:
        x = np.arange(n, dtype=float)

    result = stats.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue**2)
    if math.isnan(r_squared):
        r_squared = 0.0

    p_value: float | None = float(result.pvalue)
    if n < 3 or math.isnan(p_value):
        # With 2 points (zero degrees of freedom) or degenerate data no
        # significance test exists — do not fabricate one.
        p_value = None

    if p_value is not None and p_value < significance_level and slope > 0:
        direction = "increasing"
    elif p_value is not None and p_value < significance_level and slope < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "trend_direction": direction,
        "p_value": p_value,
    }


def detect_degradation_onset(
    feature_series: list[float],
    threshold_sigma: float = 3.0,
) -> int | None:
    """Detect the first post-baseline index exceeding baseline statistics.

    The *first half* of the series establishes a baseline (mean and
    standard deviation). The function returns the first index **after
    the baseline window** whose value exceeds
    ``mean + threshold_sigma * std``. Indices inside the baseline are
    never reported: the baseline defines "normal", so flagging onset
    inside it would be circular. If degradation starts within the
    baseline window itself, this detector cannot see it — it inflates
    its own baseline statistics instead (a longer healthy history is
    needed).

    Args:
        feature_series: Ordered feature values.
        threshold_sigma: Number of standard deviations above the
            first-half mean to trigger degradation onset.

    Returns:
        Index of degradation onset (always ``>= len(series) // 2``), or
        *None* if not detected.
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 2:
        return None

    half = n // 2
    if half < 1:
        return None

    baseline = y[:half]
    mean = float(np.mean(baseline))
    std = float(np.std(baseline, ddof=0))

    if std == 0.0:
        # Constant baseline — any deviation is significant.
        for i in range(half, n):
            if y[i] > mean:
                return i
        return None

    threshold = mean + threshold_sigma * std
    for i in range(half, n):
        if y[i] > threshold:
            return i
    return None
