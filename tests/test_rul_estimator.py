"""Tests for prognostics.rul_estimator — ISO 13374 Block 5 RUL estimation.

The estimators fit degradation curves on explicit measurement timestamps
and report ``fit_r_squared`` (goodness of fit, never "confidence").
"""

import math

import numpy as np
import pytest

import predictive_maintenance_mcp.prognostics as prognostics_pkg
import predictive_maintenance_mcp.prognostics.rul_estimator as rul_estimator_module
from predictive_maintenance_mcp.prognostics.rul_estimator import (
    estimate_rul_linear,
    estimate_rul_exponential,
    fit_exponential_degradation,
    fit_linear_degradation,
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

    @pytest.mark.parametrize(
        "series, index",
        [
            ([-5.0, -4.0, -3.0, -2.0, -1.0], 0),
            ([-1.0, 0.0, 0.5, 1.0, 2.0], 0),
            ([0.5, 1.0, 0.0, 1.5, 2.0], 2),
        ],
    )
    def test_exponential_refuses_non_positive_values(self, series, index):
        """A log-linear fit on a subset would hide the dropped measurements."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        with pytest.raises(ValueError, match=f"index {index} \\(counting from 0\\) is"):
            estimate_rul_exponential(series, timestamps, failure_threshold=10.0)

    def test_exponential_refuses_a_value_that_is_not_a_number(self):
        with pytest.raises(ValueError, match="index 1 \\(counting from 0\\) is nan"):
            estimate_rul_exponential(
                [1.0, float("nan"), 2.0], [0.0, 1.0, 2.0], failure_threshold=10.0
            )

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


class TestThresholdCrossingFits:
    """The fits report where the curve meets the threshold, even before the last point."""

    VALUES = [6.0, 7.1, 8.0, 9.2, 10.4, 9.7]
    TIMESTAMPS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_linear_fit_locates_a_crossing_before_the_last_measurement(self):
        fit = fit_linear_degradation(
            self.VALUES, self.TIMESTAMPS, failure_threshold=10.0
        )
        slope, intercept = np.polyfit(self.TIMESTAMPS, self.VALUES, 1)
        assert fit["slope"] == pytest.approx(slope)
        assert fit["crossing_time"] == pytest.approx((10.0 - intercept) / slope)
        assert fit["crossing_time"] < self.TIMESTAMPS[-1]
        assert fit["fitted_last_value"] == pytest.approx(intercept + slope * 5.0)
        assert fit["time_to_crossing"] == pytest.approx(fit["crossing_time"] - 5.0)
        assert fit["time_to_crossing"] < 0
        assert estimate_rul_linear(self.VALUES, self.TIMESTAMPS, 10.0) is None

    def test_linear_fit_has_no_crossing_without_a_rise(self):
        fit = fit_linear_degradation([5.0] * 6, self.TIMESTAMPS, failure_threshold=10.0)
        assert fit["crossing_time"] is None and fit["time_to_crossing"] is None

    def test_linear_fit_locates_a_crossing_after_the_last_measurement(self):
        """y = 2t reaches 30 at t = 15, five after the last measurement at t = 10."""
        timestamps = [0.0, 1.0, 3.0, 7.0, 10.0]
        series = [2.0 * t for t in timestamps]
        fit = fit_linear_degradation(series, timestamps, failure_threshold=30.0)
        assert fit["crossing_time"] == pytest.approx(15.0)
        assert fit["time_to_crossing"] == pytest.approx(5.0)

    def test_exponential_fit_locates_a_crossing_before_the_last_measurement(self):
        fit = fit_exponential_degradation(
            self.VALUES, self.TIMESTAMPS, failure_threshold=10.0
        )
        b, log_a = np.polyfit(self.TIMESTAMPS, np.log(self.VALUES), 1)
        assert fit["growth_rate"] == pytest.approx(b)
        assert fit["crossing_time"] == pytest.approx((math.log(10.0) - log_a) / b)
        assert fit["crossing_time"] < self.TIMESTAMPS[-1]
        assert fit["fitted_last_value"] == pytest.approx(math.exp(log_a + b * 5.0))

    def test_exponential_fit_refuses_a_non_positive_threshold(self):
        with pytest.raises(ValueError, match="failure_threshold must be positive"):
            fit_exponential_degradation(self.VALUES, self.TIMESTAMPS, 0.0)


class TestWeibullRemoved:
    """The statistically incoherent Weibull estimator must be gone."""

    def test_weibull_removed_from_module(self):
        assert not hasattr(rul_estimator_module, "estimate_rul_weibull")

    def test_weibull_removed_from_package(self):
        assert not hasattr(prognostics_pkg, "estimate_rul_weibull")
