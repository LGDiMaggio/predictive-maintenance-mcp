"""Tests for prognostics.rul_estimator — ISO 13374 Block 5 RUL estimation."""

import math

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.rul_estimator import (
    estimate_rul_linear,
    estimate_rul_exponential,
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
