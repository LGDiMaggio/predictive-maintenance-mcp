"""Tests for prognostics.kalman_rul — Kalman filter RUL estimation."""

import math

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.kalman_rul import (
    estimate_rul_kalman,
    kalman_degradation_state,
    kalman_rul_from_state,
)


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
    Q = process_noise * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
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
        result_1 = estimate_rul_kalman(
            series, failure_threshold=200.0, sampling_interval=1.0
        )
        result_2 = estimate_rul_kalman(
            series, failure_threshold=200.0, sampling_interval=2.0
        )

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
            var_l / rate**2 + gap**2 * var_r / rate**4 + 2.0 * gap * cov_lr / rate**3
        )
        without_cov = var_l / rate**2 + gap**2 * var_r / rate**4

        expected_std = math.sqrt(max(full_var, 0.0))
        assert result["rul_std"] == pytest.approx(expected_std, rel=1e-6)

        # The cross-covariance is non-zero for this filter, so omitting it
        # must give a measurably different standard deviation.
        assert abs(cov_lr) > 0
        assert result["rul_std"] != pytest.approx(math.sqrt(without_cov), rel=1e-3)


class TestKalmanDegradationState:
    """The filtered state behind every Kalman outcome, including those with no RUL."""

    def test_state_matches_an_independent_filter(self):
        rng = np.random.default_rng(11)
        series = [1.0 + 0.3 * t + rng.normal(0, 0.2) for t in range(25)]
        state = kalman_degradation_state(series, sampling_interval=2.0)
        x, P = _reference_filter_covariance(series, sampling_interval=2.0)
        assert state["final_level"] == pytest.approx(float(x[0]))
        assert state["final_rate"] == pytest.approx(float(x[1]))
        np.testing.assert_allclose(state["covariance"], P)

    def test_state_reports_a_level_past_the_threshold(self):
        """The case estimate_rul_kalman answers with None is visible in the state."""
        series = [6.0, 7.1, 8.0, 9.2, 10.4, 9.7]
        state = kalman_degradation_state(series)
        assert state["final_level"] >= 10.0 > series[-1]
        assert estimate_rul_kalman(series, failure_threshold=10.0) is None

    def test_state_reports_a_falling_rate(self):
        series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 6.0, 4.0]
        state = kalman_degradation_state(series)
        assert state["final_rate"] < 0
        assert estimate_rul_kalman(series, failure_threshold=20.0) is None

    def test_too_short_has_no_state(self):
        assert kalman_degradation_state([1.0, 2.0]) is None

    def test_rul_from_state_extrapolates_the_filtered_level(self):
        rng = np.random.default_rng(5)
        series = [0.4 * t + rng.normal(0, 0.2) for t in range(30)]
        x, _ = _reference_filter_covariance(series)
        result = kalman_rul_from_state(
            kalman_degradation_state(series), failure_threshold=40.0
        )
        assert result["rul"] == pytest.approx((40.0 - x[0]) / x[1])

    @pytest.mark.parametrize(
        "level, rate", [(10.0, 0.5), (12.0, 0.5), (5.0, 0.0), (5.0, -0.1)]
    )
    def test_no_rul_at_the_threshold_or_without_a_rising_rate(self, level, rate):
        """The boundaries estimate_rul's own outcome predicates copy."""
        state = {"final_level": level, "final_rate": rate, "covariance": np.eye(2)}
        assert kalman_rul_from_state(state, failure_threshold=10.0) is None
