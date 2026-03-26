"""
ISO 13374 Block 5 — Prognostic Assessment: Trend Analysis.

Detects degradation trends in time-series feature data using
statistical methods (linear regression, change detection).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def analyze_trend(
    feature_series: list[float],
    timestamps: list[float] | None = None,
) -> dict:
    """Fit a linear trend to a feature time series.

    Uses ordinary least-squares regression to quantify the direction
    and strength of a monotonic trend in the data.

    Args:
        feature_series: Ordered feature values (e.g., RMS velocity
            collected over successive measurements).
        timestamps: Optional time axis.  When *None*, indices
            ``0 .. N-1`` are used.

    Returns:
        Dictionary with keys:
            - slope (float)
            - intercept (float)
            - r_squared (float)
            - trend_direction (str): ``"increasing"``, ``"decreasing"``,
              or ``"stable"``
            - p_value (float)
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 2:
        return {
            "slope": 0.0,
            "intercept": float(y[0]) if n == 1 else 0.0,
            "r_squared": 0.0,
            "trend_direction": "stable",
            "p_value": 1.0,
        }

    if timestamps is not None:
        x = np.asarray(timestamps, dtype=float)
    else:
        x = np.arange(n, dtype=float)

    result = stats.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue ** 2)
    p_value = float(result.pvalue)

    if r_squared > 0.3 and slope > 0:
        direction = "increasing"
    elif r_squared > 0.3 and slope < 0:
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
    """Detect the first index where values exceed baseline statistics.

    A rolling window over the *first half* of the series establishes a
    baseline (mean and standard deviation).  The function returns the
    first index whose value exceeds ``mean + threshold_sigma * std``.

    Args:
        feature_series: Ordered feature values.
        threshold_sigma: Number of standard deviations above the
            first-half mean to trigger degradation onset.

    Returns:
        Index of degradation onset, or *None* if not detected.
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
    for i in range(n):
        if y[i] > threshold:
            return i
    return None
