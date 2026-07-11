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

    def test_direction_uses_p_value_not_r_squared(self):
        """Three points with R² > 0.3 but p > 0.05 are NOT a trend.

        The old implementation used an arbitrary R² > 0.3 cutoff, which
        called any 3 vaguely-aligned points "increasing". The direction
        must come from the slope significance test instead.
        """
        series = [0.0, 2.0, 3.0]  # R² ≈ 0.96, but p ≈ 0.12 with df=1
        result = analyze_trend(series)

        assert result["r_squared"] > 0.3
        assert result["p_value"] is not None
        assert result["p_value"] > 0.05
        assert result["trend_direction"] == "stable"

    def test_with_timestamps(self):
        """Explicit timestamps should produce correct slope units."""
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        result = analyze_trend(values, timestamps=timestamps)

        assert pytest.approx(result["slope"], abs=1e-6) == 2.0
        assert pytest.approx(result["intercept"], abs=1e-6) == 10.0
        assert result["trend_direction"] == "increasing"

    def test_short_series(self):
        """Two points fit a line but cannot support a significance test."""
        result = analyze_trend([1.0, 3.0])
        assert result["slope"] > 0
        assert result["r_squared"] > 0.99
        # No degrees of freedom → no p-value → no claimed direction.
        assert result["p_value"] is None
        assert result["trend_direction"] == "stable"

    def test_single_point_series(self):
        """A single-point series cannot have a trend."""
        result = analyze_trend([42.0])
        assert result["trend_direction"] == "stable"
        assert result["r_squared"] == 0.0
        assert result["intercept"] == 42.0
        assert result["p_value"] is None

    def test_constant_series_is_stable(self):
        """Exactly constant series: degenerate stats must not leak NaN."""
        result = analyze_trend([5.0] * 20)
        assert result["trend_direction"] == "stable"
        assert result["r_squared"] == 0.0


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

    def test_onset_never_inside_baseline_window(self):
        """A spike inside the baseline must not be reported as onset.

        The first half of the series IS the baseline that defines
        "normal"; scanning it for onset is circular. The old code
        scanned from index 0 and could flag its own baseline.
        """
        series = [1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        # With sigma=1 the spike at index 4 exceeds the baseline
        # threshold, but index 4 lies inside the baseline window (0-4).
        onset = detect_degradation_onset(series, threshold_sigma=1.0)
        assert onset is None

    def test_onset_index_at_least_baseline_length(self):
        """Any reported onset index must lie after the baseline window."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            n = int(rng.integers(6, 60))
            series = rng.normal(1.0, 0.5, n).tolist()
            # Add a late jump sometimes
            if n > 8:
                series[-2] += 20.0
            onset = detect_degradation_onset(series, threshold_sigma=2.0)
            if onset is not None:
                assert onset >= n // 2

    def test_degradation_starting_at_series_start_not_flagged(self):
        """Degradation from the very start inflates the baseline itself.

        The detector cannot separate 'always degraded' from 'normal',
        so it must return None (or a post-baseline index), never an
        index inside its own baseline.
        """
        series = [float(10 + i) for i in range(20)]  # degrading from t=0
        onset = detect_degradation_onset(series, threshold_sigma=3.0)
        assert onset is None or onset >= 10

    def test_short_series_returns_none(self):
        """Series too short for meaningful analysis."""
        assert detect_degradation_onset([1.0]) is None
        assert detect_degradation_onset([]) is None
