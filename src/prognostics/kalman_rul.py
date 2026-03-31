"""
ISO 13374 Block 5 — Prognostic Assessment: Kalman Filter RUL Estimation.

Adaptive Remaining Useful Life estimation using a linear Kalman filter
with state vector [degradation_level, degradation_rate].
"""

from __future__ import annotations

import numpy as np


def estimate_rul_kalman(
    feature_series: list[float],
    failure_threshold: float,
    sampling_interval: float = 1.0,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> dict | None:
    """Estimate RUL using a 2-state linear Kalman filter.

    The state vector is ``[degradation_level, degradation_rate]``.  The
    filter tracks degradation dynamics and extrapolates linearly from
    the final filtered state to compute Remaining Useful Life.

    Args:
        feature_series: Ordered degradation indicator values.
        failure_threshold: Value at which the component is considered
            failed.
        sampling_interval: Time between successive samples (arbitrary
            units — hours, cycles, etc.).
        process_noise: Process noise scaling factor for the state
            transition covariance matrix *Q*.
        measurement_noise: Measurement noise variance *R*.

    Returns:
        Dict with ``rul``, ``confidence``, ``method`` (``"kalman"``),
        ``confidence_interval`` (``[lower, upper]``),
        ``estimated_rate``, and ``final_level``.  Returns *None* when
        the estimate is not feasible.
    """
    y = np.asarray(feature_series, dtype=float)
    n = len(y)

    if n < 3:
        return None

    dt = sampling_interval

    # --- State-space model ---
    # State transition matrix.
    F = np.array([[1.0, dt], [0.0, 1.0]])

    # Measurement matrix (observe degradation level only).
    H = np.array([[1.0, 0.0]])

    # Process noise covariance.
    Q = process_noise * np.array([
        [dt ** 3 / 3.0, dt ** 2 / 2.0],
        [dt ** 2 / 2.0, dt],
    ])

    # Measurement noise covariance (scalar).
    R = np.array([[measurement_noise]])

    # --- Initialisation ---
    # Initial state: first observation as level, crude rate from first
    # two observations.
    x = np.array([y[0], (y[1] - y[0]) / dt])
    P = np.eye(2) * 1.0  # Initial uncertainty.

    # --- Kalman filter forward pass ---
    for k in range(n):
        z = np.array([y[k]])

        # Prediction.
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update.
        S = float((H @ P_pred @ H.T + R)[0, 0])  # Scalar innovation covariance.
        K = (P_pred @ H.T) / S  # Kalman gain (scalar division).
        innovation = z - H @ x_pred
        x = x_pred + (K @ innovation).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

    final_level = float(x[0])
    final_rate = float(x[1])

    # Threshold already exceeded.
    if final_level >= failure_threshold:
        return None

    # Cannot estimate RUL if degradation is not progressing.
    if final_rate <= 0:
        return None

    # Extrapolate: RUL = (threshold - level) / rate.
    rul = (failure_threshold - final_level) / final_rate

    # --- Confidence interval from state covariance P ---
    # Variance of RUL ≈ Var(level)/rate² + level²·Var(rate)/rate⁴
    # (first-order delta-method approximation).
    var_level = float(P[0, 0])
    var_rate = float(P[1, 1])
    gap = failure_threshold - final_level

    rul_variance = var_level / (final_rate ** 2) + (gap ** 2) * var_rate / (final_rate ** 4)
    rul_std = float(np.sqrt(max(rul_variance, 0.0)))

    ci_lower = max(0.0, rul - 1.96 * rul_std)
    ci_upper = rul + 1.96 * rul_std

    # Confidence: use normalised innovation-based heuristic.
    # Map RUL standard deviation relative to RUL itself into [0, 1].
    if rul > 0:
        confidence = max(0.0, min(1.0, 1.0 - rul_std / rul))
    else:
        confidence = 0.0

    return {
        "rul": float(rul),
        "confidence": confidence,
        "method": "kalman",
        "confidence_interval": [ci_lower, ci_upper],
        "estimated_rate": final_rate,
        "final_level": final_level,
    }
