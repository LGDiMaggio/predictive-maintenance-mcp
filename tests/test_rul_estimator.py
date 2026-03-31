"""Tests for prognostics.rul_estimator — ISO 13374 Block 5 RUL estimation."""

import math

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.rul_estimator import (
    estimate_rul_linear,
    estimate_rul_exponential,
    estimate_rul_weibull,
)


class TestLinearRUL:
    """Tests for linear RUL estimation."""

    def test_linear_rul_increasing(self):
        """Known linear ramp should yield predictable RUL."""
        # y = 1 * x  → reaches 100 at x=100; last point is x=49
        series = [float(i) for i in range(50)]
        result = estimate_rul_linear(series, failure_threshold=100.0)

        assert result is not None
        assert result["method"] == "linear"
        # Expected RUL: (100 - 49) * 1.0 = 51 time units
        assert pytest.approx(result["rul"], abs=1.0) == 51.0
        assert result["confidence"] > 0.99

    def test_linear_rul_below_threshold(self):
        """Decreasing series should never reach an upper threshold."""
        series = [100.0 - i for i in range(50)]
        result = estimate_rul_linear(series, failure_threshold=200.0)
        assert result is None

    def test_rul_none_for_flat_series(self):
        """Flat series has zero slope — no crossing possible."""
        series = [5.0] * 20
        result = estimate_rul_linear(series, failure_threshold=100.0)
        assert result is None

    def test_sampling_interval_scaling(self):
        """RUL should scale with the sampling interval."""
        series = [float(i) for i in range(50)]
        result_1 = estimate_rul_linear(series, failure_threshold=100.0, sampling_interval=1.0)
        result_2 = estimate_rul_linear(series, failure_threshold=100.0, sampling_interval=2.0)

        assert result_1 is not None and result_2 is not None
        assert pytest.approx(result_2["rul"], rel=0.01) == result_1["rul"] * 2


class TestExponentialRUL:
    """Tests for exponential RUL estimation."""

    def test_exponential_rul(self):
        """Exponential growth should yield a finite RUL."""
        # y = exp(0.1 * t) for t in [0..29]
        series = [math.exp(0.1 * t) for t in range(30)]
        threshold = math.exp(0.1 * 50)  # threshold at t=50

        result = estimate_rul_exponential(series, failure_threshold=threshold)

        assert result is not None
        assert result["method"] == "exponential"
        # Expected remaining indices: 50 - 29 = 21
        assert pytest.approx(result["rul"], abs=2.0) == 21.0
        assert result["confidence"] > 0.99

    def test_exponential_negative_values_handled(self):
        """Series with all negative values cannot be log-fitted."""
        series = [-5.0, -4.0, -3.0, -2.0, -1.0]
        result = estimate_rul_exponential(series, failure_threshold=10.0)
        assert result is None


class TestWeibullRUL:
    """Tests for Weibull degradation RUL estimation."""

    def test_weibull_increasing_degradation(self):
        """Monotonically increasing degradation should yield a finite RUL."""
        series = [0.1 + 0.5 * (i ** 1.5) for i in range(1, 31)]
        # Threshold above current max — extrapolation should reach it.
        threshold = max(series) * 1.5

        result = estimate_rul_weibull(series, failure_threshold=threshold)

        assert result is not None
        assert result["method"] == "weibull"
        assert result["rul"] >= 0
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["shape"] > 0
        assert result["scale"] > 0

    def test_weibull_flat_series(self):
        """Flat series cannot be fitted — should return None."""
        series = [5.0] * 20
        result = estimate_rul_weibull(series, failure_threshold=100.0)
        assert result is None

    def test_weibull_threshold_already_exceeded(self):
        """If the last value already exceeds the threshold, return None."""
        series = [float(i) for i in range(1, 51)]
        result = estimate_rul_weibull(series, failure_threshold=10.0)
        assert result is None

    def test_weibull_short_series(self):
        """Series with fewer than 3 points should return None."""
        series = [1.0, 2.0]
        result = estimate_rul_weibull(series, failure_threshold=100.0)
        assert result is None

    def test_weibull_negative_values(self):
        """Series with negative values cannot be Weibull-fitted."""
        series = [-3.0, -2.0, -1.0, 0.0, 1.0]
        result = estimate_rul_weibull(series, failure_threshold=10.0)
        assert result is None
