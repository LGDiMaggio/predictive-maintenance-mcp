"""Tests for prognostics.trend_analyzer — ISO 13374 Block 5 trend analysis."""

import numpy as np
import pytest

from predictive_maintenance_mcp.prognostics.trend_analyzer import (
    analyze_trend,
    detect_degradation_onset,
)


class TestAnalyzeTrend:
    """Tests for the linear trend analyser."""

    def test_linear_increasing_trend(self):
        """A clear upward ramp should be classified as increasing."""
        series = [float(i) for i in range(50)]
        result = analyze_trend(series)

        assert result["slope"] > 0
        assert result["r_squared"] > 0.99
        assert result["trend_direction"] == "increasing"
        assert result["p_value"] < 0.01

    def test_linear_decreasing_trend(self):
        """A clear downward ramp should be classified as decreasing."""
        series = [100.0 - 2.0 * i for i in range(50)]
        result = analyze_trend(series)

        assert result["slope"] < 0
        assert result["r_squared"] > 0.99
        assert result["trend_direction"] == "decreasing"

    def test_stable_no_trend(self):
        """Random noise around a constant should be classified as stable."""
        rng = np.random.default_rng(42)
        series = (5.0 + rng.normal(0, 0.1, 200)).tolist()
        result = analyze_trend(series)

        assert result["trend_direction"] == "stable"
        assert result["r_squared"] < 0.3

    def test_with_timestamps(self):
        """Explicit timestamps should produce correct slope units."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        result = analyze_trend(values, timestamps=timestamps)

        assert pytest.approx(result["slope"], abs=1e-6) == 2.0
        assert pytest.approx(result["intercept"], abs=1e-6) == 10.0
        assert result["trend_direction"] == "increasing"

    def test_short_series(self):
        """Two points should still produce a valid result."""
        result = analyze_trend([1.0, 3.0])
        assert result["slope"] > 0
        assert result["r_squared"] > 0.99

    def test_single_point_series(self):
        """A single-point series cannot have a trend."""
        result = analyze_trend([42.0])
        assert result["trend_direction"] == "stable"
        assert result["r_squared"] == 0.0
        assert result["intercept"] == 42.0


class TestDetectDegradationOnset:
    """Tests for degradation onset detection."""

    def test_degradation_onset_detected(self):
        """A jump at index 50 should be detected."""
        baseline = [1.0] * 50
        degraded = [10.0] * 50
        series = baseline + degraded

        onset = detect_degradation_onset(series, threshold_sigma=3.0)
        assert onset is not None
        assert onset == 50

    def test_no_degradation_onset(self):
        """A flat series has no degradation onset."""
        series = [5.0] * 100
        onset = detect_degradation_onset(series, threshold_sigma=3.0)
        assert onset is None

    def test_short_series_returns_none(self):
        """Series too short for meaningful analysis."""
        assert detect_degradation_onset([1.0]) is None
        assert detect_degradation_onset([]) is None
