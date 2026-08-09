"""Tests for prognostics.rul_estimator — ISO 13374 Block 5 RUL estimation.

The estimators fit degradation curves on explicit measurement timestamps
and report ``fit_r_squared`` (goodness of fit, never "confidence").
"""

import math

import pytest

import predictive_maintenance_mcp.prognostics as prognostics_pkg
import predictive_maintenance_mcp.prognostics.rul_estimator as rul_estimator_module
from predictive_maintenance_mcp.prognostics.rul_estimator import (
    estimate_rul_linear,
    estimate_rul_exponential,
)


class TestLinearRUL:
    """Tests for linear RUL estimation."""

    def test_linear_rul_increasing(self):
        """Known linear ramp should yield predictable RUL."""
        # y = 1 * t → reaches 100 at t=100; last measurement at t=49.
        timestamps = [float(i) for i in range(50)]
        series = [float(i) for i in range(50)]
        result = estimate_rul_linear(series, timestamps, failure_threshold=100.0)

        assert result is not None
        assert result["method"] == "linear"
        # Expected RUL: 100 − 49 = 51 time units.
        assert pytest.approx(result["rul"], abs=1.0) == 51.0
        assert result["fit_r_squared"] > 0.99
        assert pytest.approx(result["estimated_rate"], abs=1e-6) == 1.0

    def test_linear_rul_non_uniform_timestamps(self):
        """Irregularly spaced measurements are fitted on real time, not index."""
        timestamps = [0.0, 1.0, 3.0, 7.0, 10.0]
        series = [2.0 * t for t in timestamps]  # y = 2t
        result = estimate_rul_linear(series, timestamps, failure_threshold=30.0)

        assert result is not None
        # Crossing at t=15, last measurement at t=10 → RUL = 5.
        assert pytest.approx(result["rul"], abs=0.01) == 5.0
        assert result["fit_r_squared"] > 0.99

    def test_linear_rul_decreasing_series(self):
        """Decreasing series moves away from an upper threshold — no RUL."""
        timestamps = [float(i) for i in range(50)]
        series = [100.0 - i for i in range(50)]
        result = estimate_rul_linear(series, timestamps, failure_threshold=200.0)
        assert result is None

    def test_rul_none_for_flat_series(self):
        """Flat series has zero slope — no crossing possible."""
        timestamps = [float(i) for i in range(20)]
        series = [5.0] * 20
        result = estimate_rul_linear(series, timestamps, failure_threshold=100.0)
        assert result is None

    def test_threshold_already_passed(self):
        """If the fit crosses the threshold before the last point, return None."""
        timestamps = [float(i) for i in range(50)]
        series = [float(i) for i in range(50)]
        result = estimate_rul_linear(series, timestamps, failure_threshold=10.0)
        assert result is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="timestamp"):
            estimate_rul_linear([1.0, 2.0, 3.0], [0.0, 1.0], failure_threshold=10.0)

    def test_non_increasing_timestamps_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            estimate_rul_linear(
                [1.0, 2.0, 3.0], [0.0, 2.0, 2.0], failure_threshold=10.0
            )

    def test_no_confidence_key(self):
        """Output must expose fit_r_squared, never a 'confidence' number."""
        timestamps = [float(i) for i in range(10)]
        series = [float(i) for i in range(10)]
        result = estimate_rul_linear(series, timestamps, failure_threshold=100.0)
        assert result is not None
        assert "confidence" not in result
        assert "fit_r_squared" in result


class TestExponentialRUL:
    """Tests for exponential RUL estimation."""

    def test_exponential_rul(self):
        """Exponential growth should yield a finite RUL."""
        timestamps = [float(t) for t in range(30)]
        series = [math.exp(0.1 * t) for t in range(30)]
        threshold = math.exp(0.1 * 50)  # threshold at t=50

        result = estimate_rul_exponential(
            series, timestamps, failure_threshold=threshold
        )

        assert result is not None
        assert result["method"] == "exponential"
        # Expected remaining time: 50 − 29 = 21.
        assert pytest.approx(result["rul"], abs=2.0) == 21.0
        assert result["fit_r_squared"] > 0.99
        assert "confidence" not in result

    def test_exponential_negative_values_handled(self):
        """Series with all negative values cannot be log-fitted."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        series = [-5.0, -4.0, -3.0, -2.0, -1.0]
        result = estimate_rul_exponential(series, timestamps, failure_threshold=10.0)
        assert result is None

    def test_exponential_decay_returns_none(self):
        """Decaying series never reaches an upper threshold."""
        timestamps = [float(t) for t in range(20)]
        series = [math.exp(-0.1 * t) for t in range(20)]
        result = estimate_rul_exponential(series, timestamps, failure_threshold=10.0)
        assert result is None

    def test_non_positive_threshold_returns_none(self):
        timestamps = [0.0, 1.0, 2.0]
        series = [1.0, 2.0, 4.0]
        assert (
            estimate_rul_exponential(series, timestamps, failure_threshold=0.0) is None
        )


class TestWeibullRemoved:
    """The statistically incoherent Weibull estimator must be gone."""

    def test_weibull_removed_from_module(self):
        assert not hasattr(rul_estimator_module, "estimate_rul_weibull")

    def test_weibull_removed_from_package(self):
        assert not hasattr(prognostics_pkg, "estimate_rul_weibull")
