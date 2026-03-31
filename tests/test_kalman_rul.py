"""Tests for prognostics.kalman_rul — Kalman filter RUL estimation."""

import math

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.kalman_rul import estimate_rul_kalman


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
        assert result["confidence"] > 0.5

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

    def test_confidence_interval_validity(self):
        """Confidence interval should be well-ordered and non-negative."""
        series = [float(i) for i in range(50)]
        result = estimate_rul_kalman(series, failure_threshold=200.0)

        assert result is not None
        ci = result["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] >= 0
        assert ci[1] >= ci[0]
        assert ci[0] <= result["rul"] <= ci[1]

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
