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
        Dict with:
            - ``rul``: Estimated remaining useful life (time units).
            - ``rul_std``: Delta-method standard deviation of the RUL.
            - ``rul_interval_95``: ``[lower, upper]`` approximate 95%
              interval from the delta-method variance. Coverage has not
              been validated — treat as an order-of-magnitude band.
            - ``precision_heuristic``: Heuristic in [0, 1] computed as
              ``1 − rul_std / rul`` (clipped). It is NOT a statistical
              confidence or a probability of correctness — only a rough
              indication of how noisy the extrapolation is.
            - ``estimated_rate``, ``final_level``, ``method``
              (``"kalman"``).
        Returns *None* when the estimate is not feasible (fewer than 3
        points, non-positive filtered rate, or threshold already
        exceeded).
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
    Q = process_noise * np.array(
        [
            [dt**3 / 3.0, dt**2 / 2.0],
            [dt**2 / 2.0, dt],
        ]
    )

    # Measurement noise covariance (scalar).
    R = np.array([[measurement_noise]])

    # --- Initialisation ---
    # Initial state: first observation as level, crude rate from first
    # two observations.
    x = np.array([y[0], (y[1] - y[0]) / dt])
    P = np.eye(2) * 1.0  # Initial uncertainty.

    # --- Kalman filter forward pass ---
    # The state is already seeded from y[0] and y[1], so the measurement
    # loop starts at k=2. Re-consuming y[0] and y[1] as measurements here
    # (range(n)) would double-use the two points that defined the initial
    # state — a subtle information leak that biased the filter toward the
    # first samples.
    for k in range(2, n):
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

    # --- Uncertainty interval from state covariance P ---
    # First-order delta method for g(L, r) = (T − L) / r with gap = T − L:
    #   ∂g/∂L = −1/r,  ∂g/∂r = −gap/r²
    #   Var(RUL) ≈ Var(L)/r² + gap²·Var(r)/r⁴ + 2·gap·Cov(L, r)/r³
    # The cross-covariance term Cov(L, r) = P[0, 1] must be included:
    # level and rate estimates from the same filter are correlated.
    var_level = float(P[0, 0])
    var_rate = float(P[1, 1])
    cov_level_rate = float(P[0, 1])
    gap = failure_threshold - final_level

    rul_variance = (
        var_level / (final_rate**2)
        + (gap**2) * var_rate / (final_rate**4)
        + 2.0 * gap * cov_level_rate / (final_rate**3)
    )
    rul_std = float(np.sqrt(max(rul_variance, 0.0)))

    interval_lower = max(0.0, rul - 1.96 * rul_std)
    interval_upper = rul + 1.96 * rul_std

    # Heuristic precision indicator: RUL standard deviation relative to
    # the RUL itself, mapped into [0, 1]. This is a heuristic, NOT a
    # statistical confidence — do not present it as a probability.
    if rul > 0:
        precision_heuristic = max(0.0, min(1.0, 1.0 - rul_std / rul))
    else:
        precision_heuristic = 0.0

    return {
        "rul": float(rul),
        "rul_std": rul_std,
        "rul_interval_95": [interval_lower, interval_upper],
        "precision_heuristic": precision_heuristic,
        "method": "kalman",
        "estimated_rate": final_rate,
        "final_level": final_level,
    }
