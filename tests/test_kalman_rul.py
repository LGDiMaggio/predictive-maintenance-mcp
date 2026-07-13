"""Tests for prognostics.kalman_rul — Kalman filter RUL estimation."""

import math

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.kalman_rul import estimate_rul_kalman


def _reference_filter_covariance(
    series: list[float],
    sampling_interval: float = 1.0,
    process_noise: float = 0.01,
    measurement_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent re-implementation of the 2-state filter.

    Returns the final state vector and covariance matrix so tests can
    verify the delta-method variance formula analytically.
    """
    y = np.asarray(series, dtype=float)
    dt = sampling_interval
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = process_noise * np.array(
        [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
    )
    R = np.array([[measurement_noise]])

    x = np.array([y[0], (y[1] - y[0]) / dt])
    P = np.eye(2)

    # Match the engine: state seeded from y[0], y[1], measurements from k=2
    # (no double-use of the two points that defined the initial state).
    for k in range(2, len(y)):
        z = np.array([y[k]])
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        S = float((H @ P_pred @ H.T + R)[0, 0])
        K = (P_pred @ H.T) / S
        innovation = z - H @ x_pred
        x = x_pred + (K @ innovation).flatten()
        P = (np.eye(2) - K @ H) @ P_pred

    return x, P


class TestKalmanRUL:
    """Tests for Kalman filter-based RUL estimation."""

    def test_linear_ramp(self):
        """Perfect linear ramp should yield accurate RUL."""
        # y = 2*t for t in [0..49], threshold at y=200 (t=100).
        series = [2.0 * t for t in range(50)]
        result = estimate_rul_kalman(series, failure_threshold=200.0)

        assert result is not None
        assert result["method"] == "kalman"
        # Expected: (200 - 98) / 2 = 51 time units.
        assert pytest.approx(result["rul"], abs=3.0) == 51.0
        assert result["estimated_rate"] > 0
        assert 0.0 <= result["precision_heuristic"] <= 1.0

    def test_noisy_ramp(self):
        """Noisy linear ramp should still produce a reasonable RUL."""
        rng = np.random.default_rng(42)
        series = [1.0 * t + rng.normal(0, 0.5) for t in range(60)]
        result = estimate_rul_kalman(series, failure_threshold=100.0)

        assert result is not None
        assert result["rul"] > 0
        assert result["estimated_rate"] > 0

    def test_exponential_growth(self):
        """Exponential growth should still produce a finite RUL."""
        series = [math.exp(0.05 * t) for t in range(40)]
        threshold = math.exp(0.05 * 60)

        result = estimate_rul_kalman(series, failure_threshold=threshold)

        assert result is not None
        assert result["rul"] > 0

    def test_flat_series(self):
        """Flat series has zero rate — should return None."""
        series = [5.0] * 20
        result = estimate_rul_kalman(series, failure_threshold=100.0)
        assert result is None

    def test_insufficient_data(self):
        """Fewer than 3 points should return None."""
        result = estimate_rul_kalman([1.0, 2.0], failure_threshold=100.0)
        assert result is None

    def test_single_point(self):
        """Single data point should return None."""
        result = estimate_rul_kalman([5.0], failure_threshold=100.0)
        assert result is None

    def test_custom_noise_params(self):
        """Custom noise parameters should not crash and produce a result."""
        series = [float(i) for i in range(30)]
        result = estimate_rul_kalman(
            series,
            failure_threshold=100.0,
            process_noise=0.1,
            measurement_noise=1.0,
        )
        assert result is not None
        assert result["rul"] > 0

    def test_interval_validity(self):
        """Uncertainty interval should be well-ordered and non-negative."""
        series = [float(i) for i in range(50)]
        result = estimate_rul_kalman(series, failure_threshold=200.0)

        assert result is not None
        interval = result["rul_interval_95"]
        assert len(interval) == 2
        assert interval[0] >= 0
        assert interval[1] >= interval[0]
        assert interval[0] <= result["rul"] <= interval[1]

    def test_threshold_already_exceeded(self):
        """If current level exceeds threshold, return None."""
        series = [float(i) for i in range(100)]
        result = estimate_rul_kalman(series, failure_threshold=50.0)
        assert result is None

    def test_sampling_interval_scaling(self):
        """RUL should scale with the sampling interval."""
        series = [float(i) for i in range(50)]
        result_1 = estimate_rul_kalman(series, failure_threshold=200.0, sampling_interval=1.0)
        result_2 = estimate_rul_kalman(series, failure_threshold=200.0, sampling_interval=2.0)

        assert result_1 is not None and result_2 is not None
        # With 2x sampling interval, RUL (in time units) should roughly double.
        assert result_2["rul"] > result_1["rul"] * 1.5

    def test_no_confidence_key(self):
        """Output exposes honest names only — no 'confidence' number."""
        series = [float(i) for i in range(30)]
        result = estimate_rul_kalman(series, failure_threshold=100.0)
        assert result is not None
        assert "confidence" not in result
        assert "confidence_interval" not in result
        assert "precision_heuristic" in result
        assert "rul_interval_95" in result

    def test_variance_includes_covariance_term(self):
        """Delta-method variance must include the level-rate cross term.

        Compares rul_std against the analytic delta-method formula
        computed from an independent filter run:
            Var(RUL) = Var(L)/r² + gap²·Var(r)/r⁴ + 2·gap·Cov(L,r)/r³
        and asserts the covariance term actually changes the result.
        """
        rng = np.random.default_rng(7)
        series = [0.5 * t + rng.normal(0, 0.3) for t in range(40)]
        threshold = 50.0

        result = estimate_rul_kalman(series, failure_threshold=threshold)
        assert result is not None

        x, P = _reference_filter_covariance(series)
        level, rate = float(x[0]), float(x[1])
        var_l, var_r = float(P[0, 0]), float(P[1, 1])
        cov_lr = float(P[0, 1])
        gap = threshold - level

        full_var = (
            var_l / rate**2
            + gap**2 * var_r / rate**4
            + 2.0 * gap * cov_lr / rate**3
        )
        without_cov = var_l / rate**2 + gap**2 * var_r / rate**4

        expected_std = math.sqrt(max(full_var, 0.0))
        assert result["rul_std"] == pytest.approx(expected_std, rel=1e-6)

        # The cross-covariance is non-zero for this filter, so omitting it
        # must give a measurably different standard deviation.
        assert abs(cov_lr) > 0
        assert result["rul_std"] != pytest.approx(
            math.sqrt(without_cov), rel=1e-3
        )
